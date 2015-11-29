--[[

Class for TweetThought. 

--]]

local TweetThought = torch.class("TweetThought")
local utils = require 'utils'

-- Lua Constructor
function TweetThought:__init(config)
	-- data
	self.data = config.data
	self.pre_train = config.pre_train
	self.pre_train_dir = config.pre_train_dir
	-- model params (general)
	self.word_dim = config.word_dim
	self.char_dim = config.char_dim
	self.min_freq = config.min_freq
	self.model = config.model
	self.num_layers = config.num_layers
	self.mem_dim = config.mem_dim
	self.context_size = config.context_size
	-- optimization
	self.learning_rate = config.learning_rate
	self.grad_clip = config.grad_clip
	self.batch_size = config.batch_size
	self.max_epochs = config.max_epochs
	self.dropout = config.dropout
	self.softmaxtree = config.softmaxtree
	-- GPU/CPU
	self.gpu = config.gpu

    -- Build vocabulary
	utils.buildVocab(self)

	if self.softmaxtree == 1 then
		-- Create frequency based tree
		require 'nnx'
		self.tree, self.root = utils.create_frequency_tree(utils.create_word_map(self.vocab, self.index2word))
		if self.gpu == 1 then
			require 'cunnx'
		end
	end

	-- build the net
    self:build_model()

    m=NTM(config, self.index2word, self.word2index)
end

-- Function to build the Tweet-Thought model
function TweetThought:build_model()
	self.protos = {} -- contains all the info need to be cuda()'d
	-- Define the lookups
	self.protos.word_vecs = nn.LookupTable(#self.index2word, self.word_dim)
	self.protos.char_vecs = nn.LookupTable(#self.index2char, self.char_dim)

	-- Define the topic model
	self.protos.topic_model = nn.Sequential()





end