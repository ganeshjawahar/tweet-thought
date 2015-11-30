--[[

Class for Neural Topic Model. 

--]]

local NTM = torch.class("NTM")
local utils = require 'utils'

-- Lua Constructor
function NTM:__init(config, index2word, word2index)
	-- data
	self.data = config.data
	self.pre_train = config.pre_train
	self.pre_train_dir = config.pre_train_dir
	self.index2word = index2word
	self.word2index = word2index
	self.db = lmdb.env{Path = './ntmDB', Name = 'ntmDB'}
	-- model params (general)
	self.word_dim = config.word_dim
	self.num_topics = config.num_topics
	self.min_freq = config.min_freq
	self.neg_samples = 2
	-- optimization	
	self.learning_rate = 0.01
	self.reg = 0.001
	self.batch_size = 2
	self.max_epochs = 10
	self.create_batch = false
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
end

-- Function to train the model
function NTM:train_model()
	print('NTM training...')
	local start = sys.clock()
	self.cur_batch = nil
	local optim_state = {learningRate = self.learning_rate}
	params, grad_params = self.protos.model:getParameters()
	feval = function(x)
		-- Get new params
		params:copy(x)

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
	self.word_vecs = nn.LookupTable(#self.index2word, self.word_dim)
	self.doc_vecs = nn.LookupTable(#self.index2tweet, self.num_topics)
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
		local ae_model = AutoEncoder(config, self.index2word, self.word2index)
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
	self.batch_count = reader:get('size')[1]
	print(self.batch_count..' batches found in NTM DB.')
	self.db:close()
end

-- Function to create batches
function NTM:create_batches()
	print('creating ntm batches...')
	local start = sys.clock()
	self.db:open()
	local txn = self.db:txn()
	self.batch_count = 0
	local cur_batch = {}
	xlua.progress(1, #self.index2word)
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
				xlua.progress(pc, #self.index2word)
			end
		end
		if pc % 1000 == 0 then
			collectgarbage()
		end
		pc = pc + 1
	end
	if #cur_batch ~= 0 then
		self.batch_count = self.batch_count + 1
		txn:put('batch_'..self.batch_count, cur_batch)
		cur_batch = nil
		collectgarbage()
	end
	txn:put('size', torch.IntTensor{self.batch_count})
	xlua.progress(#self.index2word, #self.index2word)
	txn:commit()
	local reader = self.db:txn(true)
	reader:abort()
	self.db:close()
	print(string.format("Done in %.2f seconds.", sys.clock() - start))
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
		for k, v in ipairs(self.protos) do
			if type(v) == 'table' then
				for _, i in ipairs(v) do
					i = i:cuda()
				end
			else
				v = v:cuda()
			end
		end
	end
end