--[[

Class for Neural Topic Model. 

--]]

local NTM = torch.class("NTM")
local utils = require 'utils'

-- Lua Constructor
function NTM:__init(config, index2word, word2index, num_words)
	-- data
	self.data = config.data
	self.pre_train = config.pre_train
	self.pre_train_dir = config.pre_train_dir
	self.index2word = index2word
	self.word2index = word2index
	self.num_words = num_words
	self.db = lmdb.env{Path = './ntmDB', Name = 'ntmDB'}
	-- model params (general)
	self.word_dim = config.word_dim
	self.num_topics = config.num_topics
	self.min_freq = config.min_freq
	self.neg_samples = 2
	-- optimization	
	self.learning_rate = 0.1
	self.reg = 0.001
	self.batch_size = 1024
	self.max_epochs = 5
	self.create_batch = true
	-- GPU/CPU
	self.gpu = config.gpu

	-- Build vocabulary
	utils.buildTweetVocab(self)

	-- Build the model
	self:build_model()

	-- Call auto-encoder
	self:call_auto_encoder(config)

	-- Create batches
	if self.create_batch == true then
		self:create_batches()
	end
	self:print_db_status()

	-- Push stuffs to GPU
	self:ship_to_gpu()

	-- Start training
	self:train_model()

	-- Save the topic vectors
	self:save_topic_vectors()

	-- Clean the memory
	self:clean_memory()
end

-- Function to train the model
function NTM:train_model()
	print('ntm: training...')
	local start = sys.clock()
	self.cur_batch = nil
	local optim_state = {learningRate = self.learning_rate}
	params, grad_params = self.protos.model:getParameters()
	feval = function(x)
		-- Get new params
		if x ~= params then params:copy(x) end

		-- Reset gradients
		grad_params:zero()

		-- loss is average of all criterions
		local input = self.protos.model:forward(self.cur_batch)
		local loss = self.protos.criterion:forward(input, self.protos.labels)
		local grads = self.protos.criterion:backward(input, self.protos.labels)
		self.protos.model:backward(self.cur_batch, grads)

		loss = loss / #self.cur_batch
		grad_params:div(#self.cur_batch)

		return loss, grad_params
	end

	self.db:open()
	local reader = self.db:txn(true)
	for epoch = 1, self.max_epochs do
		local epoch_start = sys.clock()
		local indices = torch.randperm(self.batch_count)
		local epoch_loss = 0
		local epoch_iteration = 0
		xlua.progress(1, self.batch_count)
		for i = 1, self.batch_count do
			self.cur_batch = reader:get('batch_'..indices[i])
			local _, loss = optim.adam(feval, params, optim_state)
			epoch_loss = epoch_loss + loss[1]
			epoch_iteration = epoch_iteration + 1
			if epoch_iteration % 10 == 0 then
				xlua.progress(i, self.batch_count)	
				collectgarbage()
			end
			self.cur_batch = nil
		end
		xlua.progress(self.batch_count, self.batch_count)
		print(string.format("Epoch %d done in %.2f minutes. loss=%f\n", epoch, ((sys.clock() - epoch_start)/60), (epoch_loss / epoch_iteration)))
	end
	self.db:close()
	print(string.format("Done in %.2f seconds.", sys.clock() - start))	
end

-- Function to save the topic vectors
function NTM:save_topic_vectors()
	print('ntm: saving topic vectors...')
	local start = sys.clock()	
	-- self.db:open()
	-- local txn = self.db:txn()
	-- xlua.progress(1, 100000)
	-- self.doc_vecs = nn.LookupTable(100000, self.num_topics)
	--[[
	for i = 1, 100000 do
		txn:put(tostring(i), self.doc_vecs.weight[i])
		if i % 1000 == 0 then
			collectgarbage()
			txn:commit()
			txn = self.db:txn()
		end
	end	
	]]--
	local topic_vectors = {}
	topic_vectors.tweet2index = self.tweet2index
	topic_vectors.topic_weights = self.doc_vecs.weight
	torch.save('topic_vectors.t7', topic_vectors)
	-- txn:put('num_docs', 100000)
	-- txn:commit()
	-- self.db:close()
	local topic_model = {}
	topic_model.protos = self.protos
	topic_model.word2tweets = self.word2tweets
	topic_model.tweet2index = self.tweet2index
	topic_model.index2tweet = self.index2tweet
	torch.save('topic_model.t7', topic_model)
	-- xlua.progress(100000, 100000)
	print(string.format("Done in %.2f minutes.", (sys.clock() - start)/60))
end

-- Function to remove useless stuffs from memory
function NTM:clean_memory()
	print('ntm: cleaning the memory...')
	for k, v in ipairs(self.protos) do
		value = nil
	end
	self.word_vecs = nil
	self.doc_vecs = nil
	self.word2tweets = nil	
	self.tweet2index = nil
	self.index2tweet = nil
	collectgarbage()
end

-- Function to build the NTM model
function NTM:build_model()
	self.protos = {} -- modules to be cuda()'d
	self.protos.labels = {}
	for i = 1, self.batch_size do 
		local tensor = torch.Tensor(1 + self.neg_samples):fill(0); 
		tensor[1] = 1; 
		table.insert(self.protos.labels, tensor)
	end
	-- Define the lookups
	self.word_vecs = nn.LookupTable(#self.index2word, self.word_dim) -- 3000 for new words found during testing
	self.doc_vecs = nn.LookupTable(#self.index2tweet + 50000, self.num_topics)
	-- Define the topic - word model
	self.protos.word_topic_model = nn.Sequential()
	self.protos.word_topic_model:add(self.word_vecs)
	self.protos.word_topic_model:add(nn.Linear(self.word_dim, self.num_topics))
	-- Define the topic - document model
	self.protos.topic_document_model = nn.Sequential()
	self.protos.topic_document_model:add(self.doc_vecs)
	self.protos.topic_document_model:add(nn.SoftMax())
	-- Merge both the above models
	self.protos.model = nn.Sequential()
	self.protos.model:add(nn.ParallelTable())
	self.protos.model.modules[1]:add(self.protos.topic_document_model)
	self.protos.model.modules[1]:add(self.protos.word_topic_model)
	self.protos.model:add(nn.MM(false, true))
	self.protos.model:add(nn.Sigmoid())
	self.protos.model = nn.Sequencer(self.protos.model)
	-- Define the criterion
	self.protos.criterion = nn.SequencerCriterion(nn.MarginCriterion())
end

-- Function to call auto-encoder
function NTM:call_auto_encoder(config)
	if not path.exists('auto_ntm_w2.t7') then
		local ae_model = AutoEncoder(config, self.index2word, self.word2index, self.num_words)
		ae_model = nil
		collectgarbage()
	end
	-- update the W2 matrix with pre_trained weights
	local w2 = torch.load('auto_ntm_w2.t7')
	self.protos.word_topic_model:get(2).weight:copy(w2)
	collectgarbage()
end

-- Function to print no. of batches in the NTM DB.
function NTM:print_db_status()
	self.db:open()
	local reader = self.db:txn(true)
	self.batch_count = reader:get('num_batches')
	print(self.batch_count..' batches found in NTM DB.')
	self.db:close()
end

-- Function to create batches
function NTM:create_batches()
	print('ntm: creating ntm batches...')
	local start = sys.clock()
	self.db:open()
	local txn = self.db:txn()
	self.batch_count = 0
	local cur_batch = {}
	xlua.progress(1, self.num_words)
	local pc = 1
	for word, tweet_id_list in pairs(self.word2tweets) do
		for tweet_id, _ in pairs(tweet_id_list) do
			local doc_tensor, word_tensor = self:sample_negative_context(word, tweet_id, tweet_id_list)
			table.insert(cur_batch, {doc_tensor, word_tensor})
			if #cur_batch == self.batch_size then
				self.batch_count = self.batch_count + 1
				txn:put('batch_'..self.batch_count, cur_batch)
				cur_batch = nil
				cur_batch = {}
			end
			if self.batch_count % 1000 == 0 then
				txn:commit()
				txn = self.db:txn()
			end
		end
		if pc % 1000 == 0 then
			collectgarbage()
		end
		if pc % 100 == 0 then
			xlua.progress(pc, self.num_words)
		end
		pc = pc + 1
	end
	if #cur_batch ~= 0 then
		self.batch_count = self.batch_count + 1
		txn:put('batch_'..self.batch_count, cur_batch)
		cur_batch = nil
		collectgarbage()
	end
	txn:put('num_batches', self.batch_count)
	xlua.progress(self.num_words, self.num_words)
	txn:commit()
	self.db:close()
	print(string.format("Done in %.2f minutes.", (sys.clock() - start)/60))
end

-- Function to create negative samples
function NTM:sample_negative_context(word, tweet_id, tweet_id_list)
	local word_tensor = torch.Tensor{word}
	local doc_tensor = torch.Tensor(1 + self.neg_samples)
	doc_tensor[1] = tweet_id
	local i = 0
	while i < self.neg_samples do
		local rand_tweet_id = torch.random(#self.index2tweet)
		if tweet_id_list[rand_tweet_id] == nil then
			doc_tensor[i + 2] = rand_tweet_id
			i = i + 1
		end
	end
	if self.gpu == 1 then
		doc_tensor = doc_tensor:cuda()
		word_tensor = word_tensor:cuda()
	end
	return doc_tensor, word_tensor
end

-- Function to ship stuffs to GPU
function NTM:ship_to_gpu()
	if self.gpu == 1 then
		--[[
		for k, v in ipairs(self.protos) do
			if type(v) == 'table' then
				for _, i in ipairs(v) do
					i = i:cuda()
				end
			else
				v = v:cuda()
			end
		end
		]]--
		self.protos.model = self.protos.model:cuda()
		for i = 1, #self.protos.labels do self.protos.labels[i] = self.protos.labels[i]:cuda() end
		self.protos.criterion = self.protos.criterion:cuda()
	end
end