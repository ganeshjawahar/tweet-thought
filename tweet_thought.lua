--[[

Class for TweetThought. 

--]]

local TweetThought = torch.class("TweetThought")
local utils = require 'utils'
local model_utils = require 'model_utils'

-- Lua Constructor
function TweetThought:__init(config)
	-- data
	self.data = config.data
	self.pre_train = config.pre_train
	self.pre_train_dir = config.pre_train_dir
	self.data_db = lmdb.env{Path = './dataDB', Name = 'dataDB'}
	self.topic_db = lmdb.env{Path = './ntmDB', Name = 'ntmDB'}
	-- model params (general)
	self.word_dim = config.word_dim
	self.min_freq = config.min_freq
	self.model = config.model
	self.num_layers = config.num_layers
	self.mem_dim = config.mem_dim
	self.context_size = config.context_size
	self.num_topics = config.num_topics
	-- char model params
	self.char_dim = config.char_dim
	self.char_feature_maps = utils.computeArray(config.char_feature_maps)
	self.char_kernels = utils.computeArray(config.char_kernels)
	self.char_highway_layers = config.char_highway_layers
	self.char_final_out_size = config.char_final_out_size
	-- rntn model params
	self.rntn_out_size = config.rntn_out_size
	self.sc_rank = config.sc_rank
	self.rntn_in_size = config.rntn_in_size
	-- optimization
	self.learning_rate = config.learning_rate
	self.grad_clip = config.grad_clip
	self.batch_size = config.batch_size
	self.max_epochs = config.max_epochs
	self.dropout = config.dropout
	self.softmaxtree = config.softmaxtree
	self.param_init = config.param_init
	-- GPU/CPU
	self.gpu = config.gpu
	self.cudnn = config.cudnn
	self.num_threads = config.num_threads
	self.prefix = config.prefix
	self.opt = config

    -- Build vocabulary
	utils.buildVocab(self)

	if self.softmaxtree == 1 then
		-- Create frequency based tree
		require 'nnx'
		self.tree, self.root = utils.create_frequency_tree(utils.create_word_map(self.vocab, self.index2word), torch.sqrt(#self.index2word))
		if self.gpu == 1 then
			require 'cunnx'
		end
	end

	-- initialize the topic model	
    self.topic_layer = NTM(config, self.index2word, self.word2index, self.num_words)

	-- build the net
    self:build_model()

	-- generate batches
	self:create_batches()

	--[[
	local f = sys.clock()
	self.rntn_in_size = 100
	self.rntn_out_size = 25
	model = nn.RNTN(self.rntn_in_size, self.rntn_out_size, 5, 1):cuda()
	for i = 1, 10 do
		f1 = sys.clock()
		local input = torch.Tensor(self.rntn_in_size):cuda()
		local fwd = model:forward(input)
		print(string.format('f: %.2f',(sys.clock()-f1)))
		f1 = sys.clock()
		model:backward(input, torch.Tensor(1,self.rntn_out_size):cuda())
		print(string.format('b: %.2f',(sys.clock()-f1)))
	end
	print(string.format('total : %.2f',(sys.clock()-f)))
	]]--

end

-- Function to train the model
function TweetThought:train()	
	print('main: training...')
	local start = sys.clock()
	self.protos.model:training()
	self:define_feval()
	self.data_db:open()
	local reader = self.data_db:txn(true)
	local num_batches = reader:get('num_batches')
	for epoch = 1, self.max_epochs do
		local epoch_start = sys.clock()
		local epoch_loss, epoch_iteration = 0, 0
		xlua.progress(1, num_batches)
		local indices = torch.randperm(num_batches)
		for i = 1, num_batches do
			self.mini_batch = reader:get('batch_'..indices[i])
			local _, loss = optim.rmsprop(self.feval, self.params, self.optim_state)
			epoch_loss = epoch_loss + loss[1]
			epoch_iteration = epoch_iteration + 1
			self.mini_batch = nil
			xlua.progress(i, num_batches)
			if i % 5 == 0 then
				collectgarbage()
			end
		end
		xlua.progress(num_batches, num_batches)
		self:save_model(self.prefix..epoch)
		if epoch ~= 1 then os.execute('rm '..self.prefix..(epoch - 1)) end
		print(string.format("Epoch %d done in %.2f minutes. Loss = %f\n",epoch,((sys.clock() - epoch_start) / 60), (epoch_loss / epoch_iteration)))		
	end
	self.data_db:close()
	print(string.format("Done in %.2f minutes.", (sys.clock() - start)/60))
end

-- Function to train the model
function TweetThought:train_parallel()	
	print('main: training...')
	local start = sys.clock()
	self.data_db:open()
	local reader = self.data_db:txn(true)
	self.protos.model:training()

	local function threadedTrain(reader, params)
		local num_batches = reader:get('num_batches')
		local weights = params.protos.model:parameters()
		THREADS.serialization('threads.sharedserialize')
		for epoch = 1, params.max_epochs do
			local epoch_start = sys.clock()
			local epoch_loss, epoch_iteration = 0, 0
			xlua.progress(1, num_batches)
			local indices = torch.randperm(num_batches)

			local pool = THREADS.Threads(
				params.num_threads, 
				function()
					require 'nn'
					require 'nngraph'
					require 'cutorch'
					require 'cunn'
					require 'cudnn'	
					require 'lmdb'				
					require 'rnn'
					require 'model.Squeeze'
					require 'model.Diagonal'
					require 'model.RNTN'
					require 'model.LSTMEncoder'
					require 'model.LSTMDecoder'
					include('tweet_thought.lua')
					include('ntm.lua')
					include('auto_encoder.lua')
				end,

				function()
					local model = params.protos.model:clone('weight', 'bias')
					local word_model = model:get(1)
					local rntn = model:get(2)
					local rnn_input = model:get(3)
					local encoder = model:get(4)
					local decoders = {}
					for i = 1, (2 * params.context_size) do
						table.insert(decoders, model:get(4 + i))
					end
					local wt, dwt = model:parameters()
					local word_dim = params.word_dim
					local rntn_out_size = params.rntn_out_size
					local context_size = params.context_size

					function gupdate(mini_batch)
						local loss=0
						for it, tuples in ipairs(mini_batch) do
							model:zeroGradParameters()
							local enc_topic, enc_input_word, enc_input_char, enc_target_word = unpack(tuples[1])
							local enc_res, enc_out = {}, {}
							local enc_word_tensor = torch.Tensor(#enc_input_word, word_dim + rntn_out_size):cuda()
							for wi, word in ipairs(enc_input_word) do
								enc_res[wi] = word_model:forward({word, enc_input_char[wi], enc_topic})
								enc_out[wi] = rntn:forward(enc_res[wi])
								enc_word_tensor[wi] = rnn_input:forward({word, enc_out[wi]})
							end
							local enc_out, loss0 = encoder:forward(enc_word_tensor, enc_target_word)
							loss = loss + loss0
							local enc_final_state = {}
							for i = 2, #enc_out, 2 do table.insert(enc_final_state, enc_out[i]) end

							local enc_grad = encoder:backward(enc_word_tensor, enc_target_word)
							for wi, word in ipairs(enc_input_word) do
								local rntn_grads = rnn_input:backward({word, enc_out[wi]}, enc_grad[wi]:view(1, word_dim + rntn_out_size))
								local word_grads = rntn:backward(enc_res[wi], rntn_grads[2]:view(1, rntn_out_size))
								word_model:backward({word, enc_input_char[wi], enc_topic}, word_grads)
							end

							--  Do decoding
							for di = 2, (2 * context_size + 1) do				
								local dec_topic, dec_input_word, dec_input_char, dec_target_word = unpack(tuples[di])				
								local dec_res, dec_out = {}, {}
								local dec_word_tensor = torch.Tensor(#dec_input_word, word_dim + rntn_out_size):cuda()
								for wi, word in ipairs(dec_input_word) do
									dec_res[wi] = word_model:forward({word, dec_input_char[wi], dec_topic})
									dec_out[wi] = rntn:forward(dec_res[wi])
									dec_word_tensor[wi] = rnn_input:forward({word, dec_out[wi]})
								end
								local dec_out, loss0 = decoders[di - 1]:forward({dec_word_tensor, unpack(enc_final_state)}, dec_target_word)
								loss = loss + loss0
								local dec_grad = decoders[di - 1]:backward({dec_word_tensor, unpack(enc_final_state)}, dec_target_word)
								local dec_grad = dec_grad[1]
								for wi, word in ipairs(dec_input_word) do
									local rntn_grads = rnn_input:backward({word, enc_out[wi]}, dec_grad[wi]:view(1, word_dim + rntn_out_size))
									local word_grads = rntn:backward(enc_res[wi], rntn_grads[2]:view(1, rntn_out_size))
									word_model:backward({word, dec_input_char[wi], dec_topic}, word_grads)
								end				
							end
						end
						return loss, dwt
					end
				end
			)

			for i = 1, num_batches do
				local mini_batch = reader:get('batch_'..indices[i])
				pool:addjob(
					function(mini_batch)
						return gupdate(mini_batch)
					end,

					function(err, dwt)
						epoch_loss = epoch_loss + err
						for j = 1, #weights do weights[j]:add(-1 * params.learning_rate, dwt[j]) end
						epoch_iteration = epoch_iteration + 1
						xlua.progress(epoch_iteration, num_batches)
						if epoch_iteration % 5 == 0 then
							collectgarbage()
						end
					end,
					mini_batch
				)
			end
			pool:synchronize()

			xlua.progress(num_batches, num_batches)
			self:save_model('thought_'..epoch)
			if epoch ~= 1 then os.execute('rm thought_'..(epoch - 1)) end
			print(string.format("Epoch %d done in %.2f minutes. Loss = %f\n",epoch,((sys.clock() - epoch_start) / 60), (epoch_loss / epoch_iteration)))
		end
		pool:terminate()
	end
	threadedTrain(reader, self)

	self.data_db:close()
	print(string.format("Done in %.2f minutes.", (sys.clock() - start)/60))
end

-- Function to define the feval
function TweetThought:define_feval()
	self.optim_state = {learningRate = self.learning_rate}
	-- self.params, self.grad_params = self.protos.model:getParameters()
	self.feval = function(x)
		-- Get new params
		if x ~= self.params then
			self.params:copy(x)
		end 

		-- Reset gradients
		self.grad_params:zero()

		-- loss is average of all criterions
		local loss=0
		for it, tuples in ipairs(self.mini_batch) do
			-- Do encoding
			local f1 = sys.clock()
			local enc_topic, enc_input_word, enc_input_char, enc_target_word = unpack(tuples[1])
			local enc_res, enc_out = {}, {}
			local enc_word_tensor = torch.Tensor(#enc_input_word, self.word_dim + self.rntn_out_size)
			enc_word_tensor = self:cuda_it(enc_word_tensor)
			--print('1')
			for wi, word in ipairs(enc_input_word) do
				enc_res[wi] = self.protos.word_model:forward({word, enc_input_char[wi], enc_topic})
				--print(string.format('f1: %.2f',(sys.clock()-f1)))
				f1 = sys.clock()
				enc_out[wi] = self.protos.rntn:forward(enc_res[wi])
				--print(string.format('f2: %.2f',(sys.clock()-f1)))
				f1 = sys.clock()
				enc_word_tensor[wi] = self.protos.rnn_input:forward({word, enc_out[wi]})
				--print(string.format('f3: %.2f',(sys.clock()-f1)))
				f1 = sys.clock()
			end
			--print(string.format('encoding forward: %.2f',(sys.clock()-f1)))
			f1 = sys.clock()
			--print('2')
			local enc_out, loss0 = self.protos.encoder:forward(enc_word_tensor, enc_target_word)
			--print('3')
			--print(string.format('ef complete : %.2f',(sys.clock()-f1)))
			f1 = sys.clock()
			loss = loss + loss0
			local enc_final_state = {}
			for i = 2, #enc_out, 2 do table.insert(enc_final_state, enc_out[i]) end
			local enc_grad = self.protos.encoder:backward(enc_word_tensor, enc_target_word)
			-- print(string.format('eb complete : %.2f',(sys.clock()-f1)))
			for wi, word in ipairs(enc_input_word) do
				f1 = sys.clock()
				local rntn_grads = self.protos.rnn_input:backward({word, enc_out[wi]}, enc_grad[wi]:view(1, self.word_dim + self.rntn_out_size))
				-- print(string.format('b1: %.2f',(sys.clock()-f1)))
				f1 = sys.clock()
				local word_grads = self.protos.rntn:backward(enc_res[wi], rntn_grads[2]:view(1, self.rntn_out_size))
				-- print(string.format('b2: %.2f',(sys.clock()-f1)))
				f1 = sys.clock()
				self.protos.word_model:backward({word, enc_input_char[wi], enc_topic}, word_grads)
				-- print(string.format('b3: %.2f',(sys.clock()-f1)))
				f1 = sys.clock()
			end
			-- print(string.format('encoding backward: %.2f',(sys.clock()-f1)))
			f1 = sys.clock()

			--  Do decoding
			for di = 2, (2 * self.context_size + 1) do				
				local dec_topic, dec_input_word, dec_input_char, dec_target_word = unpack(tuples[di])				
				local dec_res, dec_out = {}, {}
				local dec_word_tensor = torch.Tensor(#dec_input_word, self.word_dim + self.rntn_out_size)
				dec_word_tensor = self:cuda_it(dec_word_tensor)
				for wi, word in ipairs(dec_input_word) do
					dec_res[wi] = self.protos.word_model:forward({word, dec_input_char[wi], dec_topic})
					dec_out[wi] = self.protos.rntn:forward(dec_res[wi])
					dec_word_tensor[wi] = self.protos.rnn_input:forward({word, dec_out[wi]})
				end
				-- print(string.format('decoding forward: %.2f',(sys.clock()-f1)))
				f1 = sys.clock()
				local dec_out, loss0 = self.protos.decoders[di - 1]:forward({dec_word_tensor, unpack(enc_final_state)}, dec_target_word)
				loss = loss + loss0
				local dec_grad = self.protos.decoders[di - 1]:backward({dec_word_tensor, unpack(enc_final_state)}, dec_target_word)
				local dec_grad = dec_grad[1]
				for wi, word in ipairs(dec_input_word) do
					local rntn_grads = self.protos.rnn_input:backward({word, enc_out[wi]}, dec_grad[wi]:view(1, self.word_dim + self.rntn_out_size))
					local word_grads = self.protos.rntn:backward(enc_res[wi], rntn_grads[2]:view(1, self.rntn_out_size))
					self.protos.word_model:backward({word, dec_input_char[wi], dec_topic}, word_grads)
				end				
				-- print(string.format('decoding backward: %.2f',(sys.clock()-f1)))
				f1 = sys.clock()
			end
			collectgarbage()
			-- print(string.format('gc: %.2f',(sys.clock()-f1)))
			-- print('4')
		end
		loss = loss / #self.mini_batch

		self.grad_params:clamp(-self.grad_clip, self.grad_clip)
		--[[
		-- If the gradients explode, scale down the gradients
		if self.grad_params:norm() >= self.grad_clip then
			self.grad_params:mul(self.grad_clip / self.grad_params:norm())
		end
		]]--

		return loss, self.grad_params
	end
end

-- Function to save the model
function TweetThought:save_model(file)
	print('main: saving the model...')
	local start = sys.clock()
	local checkpoint = {}
	checkpoint.index2word = self.index2word
	checkpoint.word2index = self.word2index
	checkpoint.index2char = self.index2char
	checkpoint.char2index = self.char2index
	checkpoint.encoder = self.protos.encoder
	checkpoint.word_model = self.protos.word_model
	checkpoint.rntn = self.protos.rntn
	checkpoint.rnn_input = self.protos.rnn_input
	checkpoint.rntn_out_size = self.rntn_out_size
	checkpoint.opt = self.opt
	checkpoint.max_word_l = self.max_word_l
	if self.softmaxtree == 1 then
		checkpoint.tree = self.tree
		checkpoint.root = self.root
	end
	checkpoint.gpu = self.gpu
	torch.save(file, checkpoint)
	print(string.format("Done in %.2f minutes.", (sys.clock() - start)/60))
end

-- Function to create character tensor
function TweetThought:create_character_tensor(word)
	local char_ids = {}
	table.insert(char_ids, self.char2index['<START>'])
	for ch in word:gmatch"." do
		table.insert(char_ids, self.char2index[ch])	
	end
	table.insert(char_ids, self.char2index['<END>'])
	local char_tensor = torch.ones(self.max_word_l)
	for i, id in ipairs(char_ids) do
		char_tensor[i] = id
	end
	char_tensor = self:cuda_it(char_tensor)
	return char_tensor
end

-- Function to generate batches
function TweetThought:create_batches()
	print('main: creating batches...')
	local start = sys.clock()
	self.data_db:open()
	local data_writer = self.data_db:txn()
	-- self.topic_db:open()
	-- local topic_reader = self.topic_db:txn(true)
	local topic_vectors = torch.load('topic_vectors.t7')

	local fptr = io.open(self.data, 'r')
	local seq_no, c = 0, 0
	xlua.progress(1, self.chat_count)
	local data_batches = {}
	while true do
		local line = fptr:read()
		if line == nil then
			break
		end
		local chat_id, chat_size = unpack(utils.splitByChar(line, '\t'))
		local chat_tensors = {}
		c = c + 1
		for i = 1, chat_size do
			local record = fptr:read()
			local tweet_id, user_id, date, tweet_text = unpack(utils.splitByChar(record, '\t'))
			local tweet_tokens = (tweet_text == nil) and {} or utils.splitByChar(tweet_text, ' ')
			tweet_tokens = utils.padTokens(tweet_tokens)

			local input_word_list, input_char_list, target_word_list = {}, {}, {}
			for i, word in ipairs(tweet_tokens) do
				local idx = nil
				if self.word2index[word] == nil then
					idx = self.word2index['<UK>']
				else
					idx = self.word2index[word]
				end
				-- lstm input words
				if 1 <= i and i < #tweet_tokens then
					local tensor = torch.IntTensor{idx}
					tensor = self:cuda_it(tensor)
					table.insert(input_word_list, tensor)
					table.insert(input_char_list, self:create_character_tensor(word))
				end
				-- lstm targets
				if 1 < i and i <= #tweet_tokens then
					local tensor = torch.IntTensor{idx}
					tensor = self:cuda_it(tensor)
					table.insert(target_word_list, tensor)
				end
			end
			local topic_tensor = topic_vectors.topic_weights[topic_vectors.tweet2index[tweet_id]]
			topic_tensor = self:cuda_it(topic_tensor)
			table.insert(chat_tensors, {topic_tensor, input_word_list, input_char_list, target_word_list})
		end
		for i = (1 + self.context_size), (#chat_tensors - self.context_size) do
			local input = {}
			table.insert(input, chat_tensors[i])
			for j = (i - self.context_size), (i + self.context_size) do
				table.insert(input, chat_tensors[j])
			end
			table.insert(data_batches, input)
			if #data_batches == self.batch_size then
				seq_no = seq_no + 1
				data_writer:put('batch_'..seq_no, data_batches)
				data_batches = nil
				data_batches = {}
			end
			if seq_no % 10 == 0 then
				collectgarbage()
			end
			if seq_no % 100 == 0 then
				data_writer:commit()
				data_writer = self.data_db:txn()
			end
			input = nil
		end
		chat_tensors = nil
		if c % 100 == 0 then
			xlua.progress(c, self.chat_count)
		end
	end
	if #data_batches ~= 0 then
		seq_no = seq_no + 1
		data_writer:put('batch_'..seq_no, data_batches)
		data_batches = nil
		data_batches = {}
	end
	topic_vectors = nil
	collectgarbage()
	data_writer:put('num_batches', seq_no)
	data_writer:commit()
	-- topic_reader:abort()
	-- self.topic_db:close()
	self.data_db:close()
	xlua.progress(self.chat_count, self.chat_count)
	print(string.format("Done in %.2f minutes.", (sys.clock() - start)/60))
end

-- Function to shif an module/tensor to GPU
function TweetThought:cuda_it(input)
	if self.gpu == 1 then
		input = input:cuda()
	end
	return input
end

-- Function to ship stuffs to GPU
function TweetThought:ship_to_gpu()
	if self.gpu == 1 then
		--[[
		for k, v in pairs(self.protos) do
			if type(v) == 'table' then
				for _, i in ipairs(v) do
					i = i:cuda()
				end
			else
				v = v:cuda()
			end
		end
		]]--
		self.protos.word_model = self.protos.word_model:cuda()
		self.protos.rntn = self.protos.rntn:cuda()
		self.protos.rnn_input = self.protos.rnn_input:cuda()
		self.protos.encoder = self.protos.encoder:cuda()
		for i = 1, (2 * self.context_size) do
			self.protos.decoders[i] = self.protos.decoders[i]:cuda()
		end
	end
end

-- Function to build the Tweet-Thought model
function TweetThought:build_model()
	print('main: creating the model...')
	self.protos = {} -- contains all the info need to be cuda()'d	

	-- Define the proposed word model
	self.char_conv_out_size = utils.computeSum(self.char_feature_maps)
	--self.rntn_in_size = self.word_dim + self.char_conv_out_size + self.num_topics
	self.protos.word_model = WORD.word(#self.index2word, self.word_dim, #self.index2char, self.char_dim, self.max_word_l, self.char_feature_maps, self.char_kernels, self.cudnn, self.char_conv_out_size, self.char_final_out_size, self.num_topics, self.rntn_in_size, self.char_highway_layers, nil, nil)

	-- Define the semantic composition layer
	self.protos.rntn = nn.RNTN(self.rntn_in_size, self.rntn_out_size, self.sc_rank, self.gpu)
	self.protos.rntn = self.protos.rntn:cuda()
	local word_lookup = nil	
    for _, node in ipairs(self.protos.word_model.forwardnodes) do
        if node.data.annotations.name == "word_lookup" then
        	word_lookup = node.data.module
        end
    end

    -- Define the input layer for LSTMs
    self.protos.rnn_input = nn.Sequential()
    self.protos.rnn_input:add(nn.ParallelTable())
    self.protos.rnn_input.modules[1]:add(word_lookup:clone("weight", "bias", "gradWeight", "gradBias"))
    self.protos.rnn_input.modules[1]:add(nn.Identity())
    self.protos.rnn_input:add(nn.JoinTable(2))

    -- Define the encoder
    local encode_config = {
		in_dim = self.word_dim + self.rntn_out_size,
		mem_dim = self.mem_dim,
		num_layers = self.num_layers,
		gpu = self.gpu,
		dropout = self.dropout,
		vocab_size = #self.index2word,
		tree = self.tree,
		root = self.root,
		softmaxtree = self.softmaxtree
	}
	self.protos.encoder = nn.LSTMEncoder(encode_config)

	-- Define the decoder
	local decode_config = {
		in_dim = self.word_dim + self.rntn_out_size,
		mem_dim = self.mem_dim,
		num_layers = self.num_layers,
		gpu = self.gpu,
		dropout = self.dropout,
		vocab_size = #self.index2word,
		tree = self.tree,
		root = self.root,
		softmaxtree = self.softmaxtree
	}
    self.protos.decoders = {}
    for i = 1, self.context_size do     	 
		local decoder_l, decoder_r = nn.LSTMDecoder(decode_config), nn.LSTMDecoder(decode_config)
		table.insert(self.protos.decoders, decoder_l)
		table.insert(self.protos.decoders, decoder_r)
    end

	self.protos.model = nn.Parallel() -- container of all the models whose parameters need to be learnt
	self.protos.model:add(self.protos.word_model)
	self.protos.model:add(self.protos.rntn)
	self.protos.model:add(self.protos.rnn_input)
	self.protos.model:add(self.protos.encoder)
	for i = 1, (2 * self.context_size) do
		self.protos.model:add(self.protos.decoders[i])
	end

    -- push stuffs to GPU
	self:ship_to_gpu()
	
	-- parameter initialization	
	self.params, self.grad_params = self.protos.model:getParameters()
	self.params:uniform(-self.param_init, self.param_init) -- small numbers uniform

    -- Initialize the word embeddings with pre-trained word vectors.
    if self.pre_train == 1 then
    	word_lookup = utils.initWordWeights(self.word2index, self.index2word, word_lookup, self.pre_train_dir..'glove.twitter.27B.'..self.word_dim..'d.txt.gz')
    end

	--[[
	print(26, self.char_dim, self.char_feature_maps, self.char_kernels, self.cudnn)
	local char_model = CHARCONV.conv(26, self.char_dim, self.char_feature_maps, self.char_kernels, self.cudnn)
	char_model=char_model:cuda()
	res=char_model:forward(torch.Tensor(26,self.char_dim):cuda())
	local hi_model = HIGHWAY.mlp(1100, 2)
	hi_model=hi_model:cuda()
	print(hi_model:forward(res:cuda()))

	input = nn.Identity()()
	r=nn.RNTN(100,20,3,1)(input)
	m=nn.gModule({input},{r})
	print(m:forward(torch.Tensor(100)))
	os.exit(0)
	--r:backward(r:forward(torch.Tensor(100)), torch.Tensor(100))
	]]--

	--[[
	local encode_config = {
		in_dim = self.word_dim,
		mem_dim = self.mem_dim,
		num_layers = self.num_layers,
		gpu = self.gpu,
		dropout = self.dropout,
		vocab_size = #self.index2word,
		tree = self.tree,
		root = self.root,
		softmaxtree = self.softmaxtree
	}
	encoder=nn.LSTMEncoder(encode_config)

	local decode_config = {
		in_dim = self.word_dim,
		mem_dim = self.mem_dim,
		num_layers = self.num_layers,
		gpu = self.gpu,
		dropout = self.dropout,
		vocab_size = #self.index2word,
		tree = self.tree,
		root = self.root,
		softmaxtree = self.softmaxtree
	}
	decoder=nn.LSTMDecoder(decode_config)


	-- encoding
	inputs = self.protos.word_vecs:forward(torch.Tensor{1, 2})
	outputs = {torch.Tensor{2},torch.Tensor{3}}	
	enc_out = encoder:forward(inputs, outputs)
	enc_final_state = {}
	for i = 2, #enc_out, 2 do
		table.insert(enc_final_state, enc_out[i])
	end
	e_grad=encoder:backward(inputs, outputs)
	self.protos.word_vecs:backward(torch.Tensor{1, 2},e_grad)	

	--[[

	self.decodeInputModel=nn.ParallelTable()
	self.clone_word_vecs=self.protos.word_vecs:clone("weight","bias","gradWeight","gradBias")
	self.decodeInputModel:add(self.clone_word_vecs)
	for i=1, self.num_layers do
		self.decodeInputModel:add(nn.Identity())
	end
	inputs = self.decodeInputModel:forward({torch.Tensor{4, 5}, unpack(enc_final_state)})
	outputs = {torch.Tensor{5},torch.Tensor{6}}	
	dec_out=decoder:forward(inputs, outputs)
	grad=decoder:backward(inputs, outputs)
	self.decodeInputModel:backward({torch.Tensor{4, 5}, unpack(enc_final_state)}, grad)

	local f1_char = torch.Tensor{1,2,3}
	local f2_char = torch.Tensor{3,4}
	local s1_char = torch.Tensor{4,5,6}
	local s2_char = torch.Tensor{8,9}
	local t1_char = torch.Tensor{7,8,9}
	local t2_char = torch.Tensor{11}
	local fword = torch.Tensor{1,4}
	local sword = torch.Tensor{2,3}
	local tword = torch.Tensor{10,11}
	--local ftopic, stopic, ttopic = torch.Tensor(self.num_topics), torch.Tensor(self.num_topics), torch.Tensor(self.num_topics)
	--]]

	--[[
	-- encoder
	mod = WORD.word(#self.index2word, self.word_dim, #self.index2char, self.char_dim, 26, self.char_feature_maps, self.char_kernels, self.cudnn, 1145, 2):cuda() --, nil, nil, 1145, 300, 3, self.gpu):cuda()
	rntn = nn.RNTN(1145,50,3,1):cuda()
	local wlook_clone = nil	
    for _, node in ipairs(mod.forwardnodes) do
        if node.data.annotations.name == "word_lookup" then
        	-- wlook_clone = node.data.module:clone("weight", "bias", "gradWeight", "gradBias")
        	word_lookup = node.data.module
        end
    end
    rnn_in = nn.Sequential()
    rnn_in:add(nn.ParallelTable())
    rnn_in.modules[1]:add(word_lookup:clone("weight", "bias", "gradWeight", "gradBias"))
    rnn_in.modules[1]:add(nn.Identity())
    rnn_in:add(nn.JoinTable(2))
    rnn_in = rnn_in:cuda()
	local encode_config = {
		in_dim = self.word_dim + 50,
		mem_dim = self.mem_dim,
		num_layers = self.num_layers,
		gpu = self.gpu,
		dropout = self.dropout,
		vocab_size = #self.index2word,
		tree = self.tree,
		root = self.root,
		softmaxtree = self.softmaxtree
	}
	encoder=nn.LSTMEncoder(encode_config)
	encoder = encoder:cuda()
    words = { torch.Tensor{1}:cuda(), torch.Tensor{2}:cuda(), torch.Tensor{3}:cuda() }
    topic = torch.Tensor(20):cuda()
    chars = {torch.Tensor{1,2,4,5,1,1,1,1,1,11,1,1,1,2,4,5,1,1,1,1,1,11,1,2,3,4}:cuda(), torch.Tensor{1,2,4,5,1,1,1,1,1,11,1,1,1,2,4,5,1,1,1,1,1,11,1,2,3,4}:cuda(), torch.Tensor{1,2,4,5,1,1,1,1,1,11,1,1,1,2,4,5,1,1,1,1,1,11,1,2,3,4}:cuda()}
    word_outs = { torch.Tensor{2}:cuda(), torch.Tensor{3}:cuda(), torch.Tensor{4}:cuda() }
    word_tensor = torch.Tensor(#words, self.word_dim+50):cuda()
    res, out = {}, {}
    for i, word in ipairs(words) do
    	print({word, chars[i], topic })
		res[i] = mod:forward({word, chars[i], topic })
		out[i] = rntn:forward(res[i])
		print({word, out[i]})
		word_tensor[i] = rnn_in:forward({word, out[i]})
    end

    enc_out = encoder:forward(word_tensor, word_outs)
    enc_final_state = {}
	for i = 2, #enc_out, 2 do
		table.insert(enc_final_state, enc_out[i])
	end
	e_grad=encoder:backward(word_tensor, word_outs)
    for i, word in ipairs(words) do
    	rntn_grads = rnn_in:backward({word, out[i]}, e_grad[i]:view(1, self.word_dim + 50))
    	mod_grads = rntn:backward(res[i], rntn_grads[2]:view(1, 50) )
    	mod:backward({word, chars[i], topic}, mod_grads)
    end

    --[[
	local decode_config = {
		in_dim = self.word_dim + 50,
		mem_dim = self.mem_dim,
		num_layers = self.num_layers,
		gpu = self.gpu,
		dropout = self.dropout,
		vocab_size = #self.index2word,
		tree = self.tree,
		root = self.root,
		softmaxtree = self.softmaxtree
	}
    decoder=nn.LSTMDecoder(decode_config)

    words = { torch.Tensor{1}, torch.Tensor{2}, torch.Tensor{3} }
    topic = torch.Tensor(20)
    chars = {torch.Tensor{1,2,4,5,1,1,1,1,1,11,1,1,1,2,4,5,1,1,1,1,1,11,1,2,3,4}, torch.Tensor{1,2,4,5,1,1,1,1,1,11,1,1,1,2,4,5,1,1,1,1,1,11,1,2,3,4}, torch.Tensor{1,2,4,5,1,1,1,1,1,11,1,1,1,2,4,5,1,1,1,1,1,11,1,2,3,4}}
    word_outs = { torch.Tensor{2}, torch.Tensor{3}, torch.Tensor{4} }
    word_tensor = torch.Tensor(#words, self.word_dim+50)
    res, out = {}, {}
    for i, word in ipairs(words) do
		res[i] = mod:forward({word, chars[i], topic })
		out[i] = rntn:forward(res[i])
		word_tensor[i] = rnn_in:forward({word, out[i]})
    end

    enc_out = decoder:forward({word_tensor,  unpack(enc_final_state)}, word_outs)
	e_grad=decoder:backward({word_tensor,  unpack(enc_final_state)}, word_outs)
    e_grad=e_grad[1]
    for i, word in ipairs(words) do
    	rntn_grads = rnn_in:backward({word, out[i]}, e_grad[i]:view(1, self.word_dim + 50))
    	mod_grads = rntn:backward(res[i], rntn_grads[2]:view(1, 50) )
    	mod:backward({word, chars[i], topic}, mod_grads)
    end
    ]]--

end