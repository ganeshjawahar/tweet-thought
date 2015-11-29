--[[

Class for Auto-Encoder Model. 

--]]

local AutoEncoder = torch.class("AutoEncoder")
local utils = require 'utils'

-- Lua Constructor
function AutoEncoder:__init(config, index2word, word2index)
	-- data
	self.data = config.data	
	self.pre_train = config.pre_train
	self.pre_train_dir = config.pre_train_dir
	self.word2index = word2index
	self.index2word = index2word
	-- model params (general)
	self.word_dim = config.word_dim
	self.min_freq = config.min_freq
	self.num_topics = config.num_topics
	-- optimization	
	self.learning_rate = 0.01
	self.batch_size = 50
	self.max_epochs = 10
	-- GPU/CPU
	self.gpu = config.gpu
	if self.pre_train ~= 1 then
		error('No pre-trained word vectors to use auto-encoder.')
	end

	-- Build the model
	self:build_model()

	-- Initialize the pre-trained word vectors
	self:initialize_word_vectors()

	-- Push stuffs to GPU
	self:ship_to_gpu()

	-- Start training
	self:train_model()

	-- Save the W2 Matrix
	self:save_weights()
end

-- Function to save the matrix
function AutoEncoder:save_weights()
	print('Saving weight...')
	torch.save('auto_ntm_w2.t7', self.protos.model:get(1):get(1).weight)
end

-- Function to start training
function AutoEncoder:train_model()
	print('Auto encoding...')
	local start = sys.clock()
	self.cur_batch = nil
	local optim_state = {learningRate = self.learning_rate}
	params, grad_params = self.protos.model:getParameters()
	feval=function(x)
		-- Get new params
		params:copy(x)

		-- Reset gradients
		grad_params:zero()

		-- loss is average of all criterions
		local input = self.protos.model:forward(self.cur_batch)
		local loss = self.protos.criterion:forward(input, self.cur_batch)
		local grads = self.protos.criterion:backward(input, self.cur_batch)
		self.protos.model:backward(self.cur_batch, grads)

		loss = loss / #self.cur_batch
		grad_params:div(#self.cur_batch)

		return loss, grad_params
	end	

	for epoch = 1, self.max_epochs do
		local epoch_start = sys.clock()
		local indices = torch.randperm(#self.index2word)
		local epoch_loss = 0
		local epoch_iteration = 0
		xlua.progress(1, #self.index2word)
		for i = 1, #self.index2word, self.batch_size do
			local batch_end = math.min(i + self.batch_size - 1, #self.index2word) - i + 1
			self.cur_batch = {}
			for j = 1, batch_end do
				table.insert(self.cur_batch, self.protos.word_vecs.weight[indices[i + j - 1]])
			end
			local _, loss = optim.sgd(feval, params, optim_state)
			epoch_loss = epoch_loss + loss[1]
			epoch_iteration = epoch_iteration + 1
			if epoch_iteration % 10 == 0 then
				xlua.progress(i, #self.index2word)	
				collectgarbage()
			end
			self.cur_batch = nil
		end
		xlua.progress(#self.index2word, #self.index2word)
		print(string.format("Epoch %d done in %.2f minutes. loss=%f\n", epoch, ((sys.clock() - epoch_start)/60), (epoch_loss / epoch_iteration)))
	end
	print(string.format("Done in %.2f seconds.", sys.clock() - start))	
end

-- Function to build the model
function AutoEncoder:build_model()
	self.protos = {} -- modules to be cuda()'d
	self.protos.word_vecs = nn.LookupTable(#self.index2word, self.word_dim)
	self.protos.model = nn.Sequential()
	self.protos.model:add(nn.Linear(self.word_dim, self.num_topics))
	self.protos.model:add(nn.Linear(self.num_topics, self.word_dim))
	self.protos.model = nn.Sequencer(self.protos.model)
	self.protos.criterion = nn.SequencerCriterion(nn.MSECriterion())
end

-- Function to initialize the word vectors
function AutoEncoder:initialize_word_vectors()
	utils.initWordWeights(self.word2index, self.index2word, self.protos.word_vecs, self.pre_train_dir..'glove.twitter.27B.'..self.word_dim..'d.txt.gz')
end

-- Function to ship stuffs to GPU
function AutoEncoder:ship_to_gpu()
	if self.gpu == 1 then
		for k, v in ipairs(self.protos) do
			v = v:cuda()
		end
	end
end