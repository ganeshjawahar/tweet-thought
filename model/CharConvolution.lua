--[[

1-D Convolution Neural Network with multiple filter widths
----------------------------------------------------------

This code is based on implementation in https://github.com/yoonkim/lstm-char-cnn.

--]]

local CharConvolution = {}

function CharConvolution.conv(length, char_dim, feature_maps, kernels, isCudnn)
    -- length = length of sentences/words (zero padded to be of same length)
    -- char_dim = character embedding_size
    -- feature_maps = table of feature maps (for each kernel width)
    -- kernels = table of kernel widths
    -- isCudnn = use cudnn (1=yes) (this should greatly speed up convolutions)

    local input = nn.Identity()() --input is batch_size x length x char_dim
    local output

    local layer1 = {}
    for i = 1, #kernels do    	
		local reduced_l = length - kernels[i] + 1 
		local pool_layer
		if isCudnn == 1 then
			-- Use CuDNN for temporal convolution.			
			-- Fake the spatial convolution.
			local conv = cudnn.SpatialConvolution(1, feature_maps[i], char_dim, kernels[i], 1, 1, 0)
			local conv_layer = conv(nn.View(1, -1, char_dim):setNumInputDims(2)(input))
			pool_layer = nn.Squeeze()(cudnn.SpatialMaxPooling(1, reduced_l, 1, 1, 0, 0)(nn.Tanh()(conv_layer)))
		else
			-- Temporal conv. much slower
			local conv = nn.TemporalConvolution(char_dim, feature_maps[i], kernels[i])
			local conv_layer = conv(input)
			pool_layer = nn.TemporalMaxPooling(reduced_l)(nn.Tanh()(conv_layer))
			pool_layer = nn.Squeeze()(pool_layer)			
		end
		table.insert(layer1, pool_layer)
    end
	if #kernels > 1 then
		output = nn.JoinTable(1)(layer1)
	else
		output = layer1[1]
	end
	return nn.gModule({input}, {output})
end

return CharConvolution