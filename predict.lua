--[[

Get the representations for each tweet.

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
require 'cunn'
require 'cutorch'
require 'cudnn'
require 'nnx'
require 'cunnx'
require 'model.Squeeze'
require 'model.Diagonal'
require 'model.RNTN'
require 'model.LSTMEncoder'
require 'model.LSTMDecoder'
require 'model.HLogSoftMax'
include('tweet_thought.lua')
include('ntm.lua')
include('auto_encoder.lua')
CHARCONV = require 'model.CharConvolution'
HIGHWAY = require 'model.HighwayMLP'
WORD = require 'model.Word'
UTF8 = require 'lua-utf8'
UTILS = require 'utils'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Tweet-Thought: Get representations by running the trained encoder')
cmd:text()
cmd:text('Options')
cmd:option('-data', 'prediction/test_pp', 'File containing tweets whose representations have to be learned')
cmd:option('-out', 'test_out', 'Output file containing representations')
cmd:option('-thought_model', 'prediction/thought_2', 'Tweet-Thought model')
cmd:option('-topic_model', 'prediction/topic_model.t7', 'NTM Topic model')
cmd:option('-char_model', 'prediction/lm_char_epoch1.00.t7', 'LSTM-CHAR-CNN model')
cmd:option('-seed', 3435, 'torch manual random number generator seed')
cmd:option('-save_prev_state', 1, 'save the state of the previous test sample')
cmd:option('-uk_topic', 3, '0=skip 1=1st_layer 2=2nd_layer 3=sum 4=avg')
cmd:option('-batch_size', 12, 'topic model batch size')
cmd:option('-neg_samples', 5, 'topic model neg samples')
cmd:option('-learning_rate', 0.1, 'topic model learning rate')
cmd:option('-max_epochs', 1, 'topic model max epochs')

opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

thought_model = torch.load(opt.thought_model)
topic_model = torch.load(opt.topic_model)

char_model = torch.load(opt.char_model)
char_model.protos.rnn:evaluate()
init_state = {}
for i = 1, char_model.opt.num_layers do
	table.insert(init_state, torch.zeros(2, char_model.opt.rnn_size):cuda())
	table.insert(init_state, torch.zeros(2, char_model.opt.rnn_size):cuda())
end
char_model_idx2char = char_model.vocab[3]
char_model_char2idx = char_model.vocab[4]
function get_char_input_tensor(word)
	local input_tensor = torch.ones(2, char_model.max_word_l):cuda()
	local len = UTF8.len(word)
	local char_id_list = {}
	table.insert(char_id_list, char_model_char2idx['<start>'])
	for _, char in UTF8.next, word do
		char = UTF8.char(char)
		local c_id = char_model_char2idx[char]
		if c_id == nil then c_id = char_model_char2idx['<unk>'] end
		table.insert(char_id_list, c_id)
	end
	table.insert(char_id_list, char_model_char2idx['<end>'])
	for i = 1, math.min(#char_id_list, char_model.max_word_l) do
		input_tensor[1][i] = char_id_list[i]
	end
	return input_tensor
end
rnn_state = {[0] = init_state}
join_model = nn.JoinTable(1):cuda()
function get_word_rep(tweet_tokens)
	local word_rep_tensors = {}
	if opt.save_prev_state ~= 1 then rnn_state[0] = init_state end
	for i, word in ipairs(tweet_tokens) do
		local rnn_input = {}
		table.insert(rnn_input, get_char_input_tensor(word))
		for _, state in pairs(rnn_state[0]) do
			table.insert(rnn_input, state)
		end
		local lst = char_model.protos.rnn:forward(rnn_input)
		rnn_state[0] = {}
		local final_out = {}
		for j = 1, (#lst - 1) do
			rnn_state[0][j] = lst[j]
			if j%2 == 0 then table.insert(final_out, rnn_state[0][j][1]) end
		end
		table.insert(word_rep_tensors, join_model:forward(final_out))
	end
	return word_rep_tensors
end

print('creating word-tweet maps...')
tweet2index = topic_model.tweet2index
index2tweet = topic_model.index2tweet
word2tweets = topic_model.word2tweets
index2word = thought_model.index2word
word2index = thought_model.word2index
word_vecs = topic_model.protos.word_topic_model:get(1)
word_dim = word_vecs.weight:size(2)
word_lookup = nil	
for _, node in ipairs(thought_model.word_model.forwardnodes) do
    if node.data.annotations.name == "word_lookup" then
    	word_lookup = node.data.module
    end
end
test_words = {}
uk_count = 0
test_tweet_index = #index2tweet + 1
test_word_index = thought_model.num_words
for line in io.lines(opt.data) do
	local tweet_id, tweet_text = unpack(UTILS.splitByChar(line, '\t'))
	local tweet_tokens = (tweet_text == nil) and {} or UTILS.splitByChar(tweet_text, ' ')
	index2tweet[#index2tweet + 1] = tweet_id
	tweet2index[tweet_id] = #index2tweet
	for _, word in ipairs(UTILS.padTokens(tweet_tokens)) do
		local w_id = word2index[word]
		if w_id ~= nil then
			if word2tweets[w_id] == nil then
				word2tweets[w_id] = {}
			end
			word2tweets[w_id][tweet2index[tweet_id]] = 1
			test_words[word] = 1
		else
			uk_count = uk_count + 1
			if opt.uk_topic ~= 0 then
				local word_rep = get_word_rep({word})[1]
				test_word_index = test_word_index + 1
				index2word[test_word_index] = word
				word2index[word] = test_word_index
				w_id = word2index[word]
				if opt.uk_topic == 1 then
					for i = 1, word_dim do
						word_vecs.weight[w_id][i] = word_rep[i]
						word_lookup.weight[w_id][i] = word_rep[i]
					end
				elseif opt.uk_topic == 2 then
					for i = 1, word_dim do
						word_vecs.weight[w_id][i] = word_rep[word_dim + i]
						word_lookup.weight[w_id][i] = word_rep[word_dim + i]
					end
				elseif opt.uk_topic == 3 then
					local tensor1 = torch.Tensor(word_dim):cuda()
					for i = 1, word_dim do
						tensor1[i] = word_rep[i]
					end
					local tensor2 = torch.Tensor(word_dim):cuda()
					for i = 1, word_dim do
						tensor2[i] = word_rep[word_dim + i]
					end
					local sum = torch.zeros(word_dim):cuda()
					torch.add(sum, tensor1, tensor2)
					for i = 1, word_dim do
						word_vecs.weight[w_id][i] = sum[i]
						word_lookup.weight[w_id][i] = sum[i]
					end
				elseif opt.uk_topic == 4 then
					local tensor1 = torch.Tensor(word_dim):cuda()
					for i = 1, word_dim do
						tensor1[i] = word_rep[i]
					end
					local tensor2 = torch.Tensor(word_dim):cuda()
					for i = 1, word_dim do
						tensor2[i] = word_rep[word_dim + i]
					end
					local sum = torch.zeros(word_dim):cuda()
					torch.add(sum, tensor1, tensor2)
					sum:div(2)
					for i = 1, word_dim do
						word_vecs.weight[w_id][i] = sum[i]						
						word_lookup.weight[w_id][i] = sum[i]
					end
				end
				if word2tweets[w_id] == nil then
					word2tweets[w_id] = {}
				end
				word2tweets[w_id][tweet2index[tweet_id]] = 1
				test_words[word] = 1
			end
		end		
	end
end
print('# unknown words = '..uk_count)

function sample_negative_context(word, tweet_id, tweet_id_list, index2tweet, neg_samples)
	local word_tensor = torch.Tensor{word}:cuda()
	local doc_tensor = torch.Tensor(1 + neg_samples):cuda()
	doc_tensor[1] = tweet_id
	local i = 0
	while i < neg_samples do
		local rand_tweet_id = torch.random(#index2tweet)
		if tweet_id_list[rand_tweet_id] == nil then
			doc_tensor[i + 2] = rand_tweet_id
			i = i + 1
		end
	end
	return doc_tensor, word_tensor
end

print('creating ntm batches...')
batches = {}
cur_batch = {}
pc = 0
for word, _ in pairs(test_words) do
	word = word2index[word]
	tweet_id_list = word2tweets[word]
	for tweet_id, _ in pairs(tweet_id_list) do
		if tweet_id >= test_tweet_index then
			local doc_tensor, word_tensor = sample_negative_context(word, tweet_id, tweet_id_list, index2tweet, opt.neg_samples)
			table.insert(cur_batch, {doc_tensor, word_tensor})
			if #cur_batch == opt.batch_size then
				table.insert(batches, cur_batch)
				cur_batch = nil
				cur_batch = {}
			end
		end
	end
	if pc % 1000 == 0 then
		collectgarbage()
	end
	pc = pc + 1
end
if #cur_batch ~= 0 then
	table.insert(batches, cur_batch)
	cur_batch = nil
	collectgarbage()
end
print(#batches..' batches found')

print('ntm training')
labels = {}
for i = 1, opt.batch_size do 
	local tensor = torch.Tensor(1 + opt.neg_samples):fill(0):cuda()
	tensor[1] = 1
	table.insert(labels, tensor)
end
function ntm_train(opt, topic_model, labels, batches)
	optim_state = {learningRate = opt.learning_rate}
	params, grad_params = topic_model.protos.model:getParameters()
	cur_batch = {}
	feval = function(x)
		-- Get new params
		if x ~= params then params:copy(x) end

		-- Reset gradients
		grad_params:zero()

		-- loss is average of all criterions
		local input = topic_model.protos.model:forward(cur_batch)
		local loss = topic_model.protos.criterion:forward(input, labels)
		local grads = topic_model.protos.criterion:backward(input, labels)
		topic_model.protos.model:backward(cur_batch, grads)

		loss = loss / #cur_batch
		grad_params:div(#cur_batch)

		return loss, grad_params
	end

	for epoch = 1, opt.max_epochs do
		local epoch_start = sys.clock()
		local indices = torch.randperm(#batches)
		local epoch_loss = 0
		local epoch_iteration = 0
		xlua.progress(1, #batches)
		for i = 1, #batches do
			cur_batch = batches[indices[i]]
			local _, loss = optim.adam(feval, params, optim_state)
			epoch_loss = epoch_loss + loss[1]
			epoch_iteration = epoch_iteration + 1
			if epoch_iteration % 10 == 0 then
				xlua.progress(i, #batches)	
				collectgarbage()
			end
			cur_batch = nil
		end
		xlua.progress(#batches, #batches)
		print(string.format("Epoch %d done in %.2f minutes. loss=%f\n", epoch, ((sys.clock() - epoch_start)/60), (epoch_loss / epoch_iteration)))
	end
	return topic_model.protos.topic_document_model:get(1)
end
doc_vecs = ntm_train(opt, topic_model, labels, batches)
batches = nil
word2tweets = nil
collectgarbage()

print('getting representations...')
out_fptr = io.open(opt.out, 'w')
for line in io.lines(opt.data) do
	local tweet_id, tweet_text = unpack(UTILS.splitByChar(line, '\t'))
	local tweet_tokens = (tweet_text == nil) and {} or UTILS.splitByChar(tweet_text, ' ')
	local words, chars = {}, {}
	for _, word in ipairs(UTILS.padTokens(tweet_tokens)) do
		local w_id = word2index[word]
		if w_id == nil then w_id = word2index['<UK>'] end
		table.insert(words, torch.Tensor{w_id}:cuda())
		local len = UTF8.len(word)
		local char_tensor = torch.ones(thought_model.max_word_l):cuda()
		local i = 1
		for _, char in UTF8.next, word do
			char = UTF8.char(char)
			c_id = thought_model.char2index[char]
			if c_id == nil then c_id = thought_model.char2index['<UK>'] end
			char_tensor[i] = c_id
			i = i + 1
		end
		table.insert(chars, char_tensor)
	end
	local src_words, target_words = {}, {}
	for i, word in ipairs(words) do
		if i < #words then table.insert(src_words, word) end
		if i > 1 then table.insert(target_words, word) end
	end
	local topic = doc_vecs.weight[tweet2index[tweet_id]]
	local enc_word_tensor = torch.Tensor(#words, thought_model.opt.word_dim + thought_model.opt.rntn_out_size):cuda()
	for i, word in ipairs(src_words) do
		local word_outs = thought_model.word_model:forward({word, chars[i], topic})
		local rntn_outs = thought_model.rntn:forward(word_outs)
		enc_word_tensor[i] = thought_model.rnn_input:forward({word, rntn_outs})
	end
	local enc_out = thought_model.encoder:forward(enc_word_tensor, target_words)
	local enc_final_state = {}
	for i = 2, #enc_out, 2 do
		table.insert(enc_final_state, enc_out[i])
	end
	local final_tensor = join_model:forward(enc_final_state)
	out_fptr:write(tweet_id)
	for i = 1, (#final_tensor)[2] do
		out_fptr:write('\t'..final_tensor[1][i])
	end
	out_fptr:write('\n')
	break
end
out_fptr:close()