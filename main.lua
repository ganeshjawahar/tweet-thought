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
require 'model.Squeeze'
require 'model.Diagonal'
require 'model.RNTN'
require 'model.LSTMEncoder'
require 'model.LSTMDecoder'
include('tweet_thought.lua')
include('ntm.lua')
include('auto_encoder.lua')
CHARCONV = require 'model.CharConvolution'
HIGHWAY = require 'model.HighwayMLP'
WORD = require 'model.Word'
THREADS = require 'threads'
UTF8 = require 'lua-utf8'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Tweet-Thought: Learning Tweet Representations using Skip-Thought Vectors')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-data', 'data/chat-data-pp.tsv', 'Document Directory.')
cmd:option('-pre_train', 1, 'initialize word embeddings with pre-trained vectors?')
cmd:option('-pre_train_dir', 'data/glove/', 'Directory for accesssing the pre-trained word embeddings')
-- model opt (general)
cmd:option('-word_dim', 25, 'dimensionality of word embeddings')
cmd:option('-min_freq', 5, 'words that occur less than <int> times will not be taken for training')
cmd:option('-model', 'lstm', 'LSTM variant to train (lstm, bi-lstm)')
cmd:option('-num_layers', 3, 'number of layers in LSTM')
cmd:option('-mem_dim', 150, 'LSTM memory dimensions')
cmd:option('-context_size', 1, 'Skip-Thought Context size')
-- char model opt
cmd:option('-char_dim', 20, 'dimensionality of character embeddings')
cmd:option('-char_feature_maps', '50', 'number of feature maps in the CNN')
cmd:option('-char_kernels', '2', 'conv net kernel widths')
cmd:option('-char_highway_layers', 2, 'number of highway layers')
cmd:option('-char_highway_output_size', 50, 'output layer after information flows through highway layers')
cmd:option('-char_final_out_size', 500, 'output size after char hw.')
-- topic model opt
cmd:option('-num_topics', 50, 'k - no. of unique topics in the dataset')
-- rntn mode opt
cmd:option('-rntn_in_size', 50, 'input size for rntn (semantic composition).')
cmd:option('-rntn_out_size', 50, 'semantic compositionality (rntn) output size')
cmd:option('-sc_rank', 3, 'rank for approximation of tensor weight matrix for each slice')
-- optimization
cmd:option('-learning_rate', 0.1, 'learning rate')
cmd:option('-grad_clip', 5, 'clip gradients at this value')
cmd:option('-batch_size', 128, 'number of sequences to train on in parallel')
cmd:option('-max_epochs', 15, 'number of full passes through the training data')
cmd:option('-dropout', 0.5, 'dropout for regularization, used after each LSTM hidden layer. 0 = no dropout')
cmd:option('-softmaxtree', 1, 'use SoftmaxTree instead of the inefficient (full) softmax')
cmd:option('-reg', 0.001, 'L2 norm regularization parameter')
cmd:option('-param_init', 0.05, 'initialize parameters at')
-- GPU/CPU
cmd:option('-num_threads', 2, 'no. of asynchronous threads to train parallel')
cmd:option('-asgd', 0, 'use asynchronous gradient descent for faster training')
cmd:option('-gpu', 1, '1=use gpu; 0=use cpu;')
cmd:option('-cudnn', 1,'use cudnn (1=yes). this should greatly speed up convolutions')
-- Book-keeping
cmd:option('-seed', 3435, 'torch manual random number generator seed')
cmd:option('-print_opt', 0, 'output the parameters in the console. 0=dont print; 1=print;')
cmd:option('-prefix', 'thought_', 'prefix for the files to be saved')

-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

if opt.print_opt == 1 then
	-- output the parameters	
	for param, value in pairs(opt) do
	    print(param ..' : '.. tostring(value))
	end
end

-- load cuda libraries
if opt.gpu == 1 then
	require 'cunn'
	require 'cutorch'
end
if opt.cudnn == 1 then
	require 'cudnn'
end

model = TweetThought(opt)
if opt.asgd == 1 then
	model:train_parallel()
else
	model:train()
end