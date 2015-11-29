--[[

Tweet-Thought: Learning Tweet Representations using Skip-Thought Vectors

]]--

require 'torch'
require 'io'
require 'nn'
require 'nngraph'
require 'sys'
require 'optim'
require 'os'
require 'xlua'
require 'lfs'
require 'rnn'
require 'lmdb'
include('tweet_thought.lua')
include('ntm.lua')
include('auto_encoder.lua')

cmd = torch.CmdLine()
cmd:text()
cmd:text('Tweet-Thought: Learning Tweet Representations using Skip-Thought Vectors')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-data', 'data/chat-data-pp.tsv', 'Document Directory.')
cmd:option('-pre_train', 1, 'initialize word embeddings with pre-trained vectors?')
cmd:option('-pre_train_dir', '/home/ganesh/Documents/t2v/proj/wuit/data/Glove/', 'Directory for accesssing the pre-trained word embeddings')
-- model params (general)
cmd:option('-word_dim', 25, 'dimensionality of word embeddings')
cmd:option('-min_freq', 5, 'words that occur less than <int> times will not be taken for training')
cmd:option('-model', 'lstm', 'LSTM variant to train (lstm, bi-lstm)')
cmd:option('-num_layers', 3, 'number of layers in LSTM')
cmd:option('-mem_dim', 150, 'LSTM memory dimensions')
cmd:option('-context_size', 1, 'Skip-Thought Context size')
-- char model params
cmd:option('-char_dim', 100, 'dimensionality of character embeddings')
cmd:option('-char_width', 3, 'char. convolutional model filter width i.e. dimensionality of one sequence element')
cmd:option('-conv_out_size', 10, 'char. convolution output size i.e number of derived features for one sequence element')
-- topic model params
cmd:option('-num_topics', 20, 'k - no. of unique topics in the dataset')
-- optimization
cmd:option('-learning_rate', 0.01, 'learning rate')
cmd:option('-grad_clip', 5, 'clip gradients at this value')
cmd:option('-batch_size', 10, 'number of sequences to train on in parallel')
cmd:option('-max_epochs', 5, 'number of full passes through the training data')
cmd:option('-dropout', 0.5, 'dropout for regularization, used after each LSTM hidden layer. 0 = no dropout')
cmd:option('-softmaxtree', 1, 'use SoftmaxTree instead of the inefficient (full) softmax')
cmd:option('-reg', 0.001, 'L2 norm regularization parameter')
-- GPU/CPU
cmd:option('-gpu', 0, '1=use gpu; 0=use cpu;')
-- Book-keeping
cmd:option('-print_params', 0, 'output the parameters in the console. 0=dont print; 1=print;')

-- parse input params
params = cmd:parse(arg)

if params.print_params == 1 then
	-- output the parameters	
	for param, value in pairs(params) do
	    print(param ..' : '.. tostring(value))
	end
end

-- load cuda libraries
if params.gpu == 1 then
	require 'cunn'
	require 'cutorch'
end

model=TweetThought(params)
--model:train()
--model:save_model()