--[[

Extension of Semantic Compositionality used in Recursive Neural Tensor Network [1]
----------------------------------------------------------------------------------

References:
1. Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank, EMNLP, 2013.

--]]

local RNTN, parent = torch.class('nn.RNTN', 'nn.Module')

function RNTN:__init(input_size, output_size, rank, gpu)
	-- input_size = dimensionality of the input embeddings
	-- output_size = expected dimensionality of the output
	-- rank = rank for approximation of tensor weight matrix for each slice
	-- gpu = 1=use gpu; 0=use cpu;
	parent.__init(self)
	self.input_size = input_size
	self.output_size = output_size
	self.rank = rank
	self:initializeModel()
	self.gpu = gpu
end

-- create the semantic composition model
function RNTN:createModel()
	local input = {}
	table.insert(input, nn.Identity()())
	-- output_size diag inputs
	for i = 1, self.output_size do
		table.insert(input, nn.Identity()())
	end

	local output = {}
	for i = 1, self.output_size do
		local left = nn.MM(false, false){nn.Diagonal(self.input_size, self.gpu)(input[i + 1]), nn.View(self.input_size, 1)(input[1])}
		local right = nn.Linear(self.rank, self.input_size)(nn.Linear(self.input_size, self.rank)(input[1]))
		local sum = nn.CAddTable(){left, right}
		local res = nn.MM(false, false){nn.View(1, self.input_size)(input[1]), sum}
		table.insert(output, res)
	end

	if self.output_size > 1 then
		output = nn.JoinTable(2)(output)
	else
		output = output[1]
	end

	return nn.gModule(input, {output})
end

-- forward propagation
function RNTN:forward(input)
	self.diag_out = self.diag_lookup:forward(self.diag_input)
	return self.model:forward({input, unpack(self.diag_out)})
end

-- backward propagation
function RNTN:backward(input, grads)
	local diag_grads = self.model:backward({input, unpack(self.diag_out)}, grads)
	self.diag_lookup:backward(self.diag_input, self:getInputGrads(diag_grads))
	return diag_grads[1]
end

-- reset gradient buffers
function RNTN:zeroGradParameters()
	self.master_cell:zeroGradParameters()
end

-- return the parameters of the model
function RNTN:parameters()
	return self.master_cell:parameters()
end

function RNTN:initializeModel()
	-- Define the diagonal input model
	self.diag_lookup = nn.Sequential()
	self.diag_lookup:add(nn.LookupTable(self.output_size, self.input_size))
	self.diag_lookup:add(nn.SplitTable(1))
	-- Define the diagonal input tensor
	self.diag_input = torch.IntTensor(self.output_size)
	for i = 1, self.output_size do
		self.diag_input[i] = i
	end	
	-- Define the model
	self.model = self:createModel()
	if self.gpu == 1 then 
		self.model = self.model:cuda()
		self.diag_lookup = self.diag_lookup:cuda() 
		self.diag_input = self.diag_input:cuda()
	end
	self.master_cell = nn.Parallel():add(self.model):add(self.diag_lookup)
end

function RNTN:getInputGrads(grads)
	local resGrads = {}
	for i = 1, self.output_size do
		table.insert(resGrads, grads[i + 1])
	end
	return resGrads
end