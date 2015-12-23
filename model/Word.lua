--[[

Proposed Word Embedding Model
-----------------------------

Semantic Compositionality of word, topic and character representations

]]--

local Word = {}

function Word.word(num_words, word_dim, num_chars, char_dim, length, feature_maps, kernels, isCudnn, highway_input_size, num_layers, bias, f)
	-- num_words = number of unique words in the corpus
	-- word_dim = size of the word embeddings
	-- num_chars = number of unique characters in the corpus
	-- char_dim = size of the character embeddings
    
    -- (CHAR-CONVOLUTION PARAMS)
    -- length = length of sentences/words (zero padded to be of same length)
    -- feature_maps = table of feature maps (for each kernel width)
    -- kernels = table of kernel widths
    -- isCudnn = use cudnn (1=yes) (this should greatly speed up convolutions)

    -- (HIGHWAY PARAMS)
    -- highway_input_size = dimensionality of inputs
    -- num_layers = number of hidden layers (default = 1)
    -- bias = bias for transform gate (default = -2)
    -- f = non-linearity (default = ReLU)

	local input = {}
	table.insert(input, nn.Identity()()) -- word index
	table.insert(input, nn.Identity()()) -- set of character indices of word
	table.insert(input, nn.Identity()()) -- topic distribution of tweet in which word is appearing.

	local output = {}
	local word_embed = nn.View(-1)(nn.LookupTable(num_words, word_dim)(input[1]):annotate{name = 'word_lookup'})
	local char_embed = nn.LookupTable(num_chars, char_dim)(input[2]):annotate{name = 'char_lookup'}
	local char_conv_out = CHARCONV.conv(length, char_dim, feature_maps, kernels, isCudnn)(char_embed)
	local concat = nn.JoinTable(1){word_embed, char_conv_out, input[3]}
	local hw_out = HIGHWAY.mlp(highway_input_size, num_layers)(concat)
	table.insert(output, hw_out)

	return nn.gModule(input, output)
end

return Word