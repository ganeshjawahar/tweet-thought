--[[
Utility function used by TweetThought class.
--]]

local utils = {}

-- Function to trim the string
function utils.trim(s)
  return (s:gsub("^%s*(.-)%s*$",  "%1"))
end

-- Function to pad tokens.
function utils.padTokens(tokens)
	local res = {}

	-- Append begin token
	table.insert(res, '<bos>')

	for _, word in ipairs(tokens) do
		table.insert(res, word)
	end

	-- Append end tokens
	table.insert(res, '<eos>')

	return res
end

-- Function to split a string by given char.
function utils.splitByChar(str, inSplitPattern)
	str = utils.trim(str)
	outResults = {}
	local theStart  =  1
	local theSplitStart, theSplitEnd = string.find(str, inSplitPattern, theStart)
	while theSplitStart do
		table.insert(outResults, string.sub(str, theStart, theSplitStart-1))
		theStart = theSplitEnd + 1
		theSplitStart, theSplitEnd = string.find(str, inSplitPattern, theStart)
	end
	table.insert(outResults, string.sub(str, theStart))
	return outResults
end

-- Function to build vocabulary from the corpus
function utils.buildVocab(config)
	print('Building character, word maps...')
	local start = sys.clock()
	config.vocab = {} -- word frequency map
	config.index2word = {}
	config.word2index = {}
	config.index2char = {[1] = '<ZERO>'}
	config.char2index = {['ZERO'] = 1}
	config.max_word_l = 0 -- maximum length of a word in the corpus

	local word_count, chat_count, max_word = 0, 0, nil
	local fptr = io.open(config.data, 'r')
	while true do
		local line = fptr:read()
		if line == nil then
			break
		end
		local chat_id, chat_size = unpack(utils.splitByChar(line, '\t'))
		for i = 1, chat_size do
			local record = fptr:read()
			local tweet_id, user_id, date, tweet_text = unpack(utils.splitByChar(record, '\t'))
			local tweet_tokens = (tweet_text == nil) and {} or utils.splitByChar(tweet_text, ' ')
			for _, word in ipairs(utils.padTokens(tweet_tokens)) do
				-- Fill word vocab.
				if config.vocab[word] == nil then
					config.vocab[word] = 1
				else
					config.vocab[word] = config.vocab[word] + 1
				end
				word_count = word_count + 1
				-- Fill character vocab.
				local char_count = 0
				local len = UTF8.len(word)
				for _, char in UTF8.next, word do
					ch = UTF8.char(char)
					if config.char2index[ch] == nil then
						config.index2char[#config.index2char + 1] = ch
						config.char2index[ch] = #config.index2char
					end
					char_count = char_count + 1
				end
				--[[
				for ch in word:gmatch"." do
					if config.char2index[ch] == nil then
						config.index2char[#config.index2char + 1] = ch
						config.char2index[ch] = #config.index2char
					end
					char_count = char_count + 1
				end
				]]--			
				if char_count > config.max_word_l then
					config.max_word_l = char_count
					max_word = word
				end
			end
		end
		chat_count = chat_count + 1
	end
	io.close(fptr)
	config.chat_count = chat_count	
	config.max_word_l = config.max_word_l + 2 -- one each for begin and end tokens

	-- Discard the words that doesn't meet minimum frequency and create indices.
	for word, count in pairs(config.vocab) do
		if count < config.min_freq then
			config.vocab[word] = nil
		else
			config.index2word[#config.index2word + 1] = word
			config.word2index[word] = #config.index2word
		end
	end

	-- Add unknown word
	config.vocab['<UK>'] = 1
	config.index2word[#config.index2word + 1] = '<UK>'
	config.word2index['<UK>'] = #config.index2word
	config.num_words = #config.index2word
	for i = 1, 500 do		
		config.index2word[#config.index2word + 1] = 'SPECIAL-'..i
		config.word2index['SPECIAL-'..i] = #config.index2word
	end
	-- Add special characters (Start/End of a word markers)
	config.index2char[#config.index2char + 1] = '<START>'
	config.char2index['<START>'] = #config.index2char
	config.index2char[#config.index2char + 1] = '<END>'
	config.char2index['<END>'] = #config.index2char
	config.index2char[#config.index2char + 1] = '<UK>'
	config.char2index['<UK>'] = #config.index2char

	print('Maximum word length is: '..config.max_word_l..' ('..max_word..')')
	print(string.format("Vocab size after eliminating words occuring less than %d times: %d", config.min_freq, config.num_words))
	print(string.format("%d characters, %d words, %d chats processed in %.2f minutes.", #config.index2char, word_count, chat_count, ((sys.clock() - start) / 60)))
end

-- Function to build frequency-based tree for Hierarchical Softmax
function utils.create_frequency_tree(freq_map, binSize)
	binSize = 100
	print('Creating frequency tree with '..binSize..' as bin size...')
	local start = sys.clock()
	local ft = torch.IntTensor(freq_map)
	local vals, indices = ft:sort()
	local tree = {}
	local id = indices:size(1)
	function recursiveTree(indices)
		if indices:size(1) < binSize then
			id = id + 1
			tree[id] = indices
			return
		end
		local parents = {}
		for start = 1, indices:size(1), binSize do
			local stop = math.min(indices:size(1), start + binSize - 1)
			local bin = indices:narrow(1, start, stop - start + 1)
			assert(bin:size(1) <= binSize)
			id = id + 1
			table.insert(parents, id)
			tree[id] = bin
		end
		recursiveTree(indices.new(parents))
	end
	recursiveTree(indices)	
	print(string.format('Done in %.2f minutes', ((sys.clock() - start) / 60)))
	return tree, id
end

-- Function to create word map (for Softmaxtree)
function utils.create_word_map(vocab, index2word)
	word_map = {}
	for i=1, #index2word do
		word_map[i] = vocab[index2word[i]]
	end
	return word_map
end

-- Function to initalize word weights
function utils.initWordWeights(word2index, index2word, word_vecs, file)
	print('initializing the pre-trained embeddings...')
	local start = sys.clock()
	local ic = 0
	for line in io.lines(file) do
		local content = utils.splitByChar(line, ' ')
		local word = content[1]
		if word2index[word] ~= nil then
			local tensor = torch.Tensor(#content-1)
			for i = 2, #content do
				tensor[i - 1] = tonumber(content[i])
			end
			word_vecs.weight[word2index[word]] = tensor
			ic = ic + 1
		end
	end
	print(string.format("%d out of %d words initialized.", ic, #index2word))
	print(string.format("Done in %.2f seconds.", sys.clock() - start))
	return word_vecs
end

-- Function to build tweet vocabulary from the corpus
function utils.buildTweetVocab(config)
	print('Building tweet maps...')
	local start = sys.clock()
	config.tweet2index = {}
	config.index2tweet = {}
	config.word2tweets = {}

	local tweet_count = 0
	local fptr = io.open(config.data, 'r')
	while true do
		local line = fptr:read()
		if line == nil then
			break
		end
		local chat_id, chat_size = unpack(utils.splitByChar(line, '\t'))
		for i = 1, chat_size do
			local record = fptr:read()
			local tweet_id, user_id, date, tweet_text = unpack(utils.splitByChar(record, '\t'))
			local tweet_tokens = (tweet_text == nil) and {} or utils.splitByChar(tweet_text, ' ')
			-- Build tweet map
			config.index2tweet[#config.index2tweet + 1] = tweet_id
			config.tweet2index[tweet_id] = #config.index2tweet
			tweet_count = tweet_count + 1
			-- Build word - tweet map
			for _, word in ipairs(tweet_tokens) do
				local w_id = config.word2index[word]
				if w_id ~= nil then
					if config.word2tweets[w_id] == nil then
						config.word2tweets[w_id] = {}
					end
					config.word2tweets[w_id][config.tweet2index[tweet_id]] = 1
				end
			end
		end
	end
	io.close(fptr)
	print(string.format("%d tweets processed in %.2f minutes.", tweet_count, ((sys.clock() - start) / 60)))
end

-- Function to return array of numbers originally in string
function utils.computeArray(string)
	local res = {}
	local items = utils.splitByChar(string, ',')
	for _, item in ipairs(items) do
		table.insert(res, tonumber(item))
	end
	return res
end

-- Function to return sum of array of numbers
function utils.computeSum(num_table)
	local res = 0
	for _, item in ipairs(num_table) do
		res = res + tonumber(item)
	end
	return res
end

return utils