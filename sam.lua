require 'nn'
require 'nngraph'
require 'model.Squeeze'

inp=5;  -- dimensionality of one sequence element
outp=10; -- number of derived features for one sequence element
kw=2;   -- kernel only operates on one sequence element per step
dw=1;   -- we step once and go on to the next sequence element

model=nn.Sequential()
model:add(nn.TemporalConvolution(inp,outp,kw,dw))
model:add(nn.Tanh())
--print(model:forward(torch.rand(26,5)))
reduced = 26 - kw + 1
model:add(nn.TemporalMaxPooling(reduced))
--print(model:forward(torch.rand(26,5)))


--[[
input=nn.Identity()()
local conv = nn.TemporalConvolution(inp,outp,kw,dw)
local conv_layer = conv(input)
reduced = 26 - kw + 1
pool_layer = nn.TemporalMaxPooling(reduced)(nn.Tanh()(conv_layer))
output = nn.Squeeze()(pool_layer)	
m=nn.gModule({input},{output})
print(m:forward(torch.rand(26,5)))
]]--

--[[
require 'cunn'
require 'cudnn'
input=nn.Identity()()
local conv = cudnn.SpatialConvolution(1, outp, inp, kw, 1, 1, 0)
local conv_layer = conv(nn.View(1, -1, inp):setNumInputDims(2)(input))
pool_layer = nn.Squeeze()(cudnn.SpatialMaxPooling(1, reduced, 1, 1, 0, 0)(nn.Tanh()(conv_layer)))
output = nn.Squeeze()(pool_layer)	
m=nn.gModule({input},{output})
m=m:cuda()
--print(m:forward(torch.rand(26,5):cuda()))

local input = nn.Identity()()
local output
kernels = {2,3}
feature_maps={100,20}
isCudnn=false
char_dim=5
length=26

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
output = layer1[2]
m=nn.gModule({input}, {nn.JoinTable(2)(layer1)})
--print(m:forward(torch.rand(1,26,5)))
--]]

require 'model.Diagonal'
input_size = 3
rank = 2

local input = {}
table.insert(input, nn.Identity()())
table.insert(input, nn.Identity()())
table.insert(input, nn.Identity()())

local output = {}
--table.insert(output, nn.Identity()(input[1]))
--table.insert(output, nn.Identity()(input[2]))
--table.insert(output, nn.Diagonal()(input[2]))
--[[
local left= nn.MM(false,false){nn.Diagonal()(input[2]),nn.View(input_size,1)(input[1])}
local right=nn.Linear(rank,input_size)(nn.Linear(input_size,rank)(input[1]))
local sum=nn.CAddTable(){left,right}
local res=nn.MM(false,false){nn.View(1,input_size)(input[1]),sum}
table.insert(output,res)

local left= nn.MM(false,false){nn.Diagonal()(input[3]),nn.View(input_size,1)(input[1])}
local right=nn.Linear(rank,input_size)(nn.Linear(input_size,rank)(input[1]))
local sum=nn.CAddTable(){left,right}
local res=nn.MM(false,false){nn.View(1,input_size)(input[1]),sum}
table.insert(output,res)
output = nn.JoinTable(1)(output)
for k = 1, 2 do
	local left= nn.MM(false,false){nn.Diagonal()(input[k+1]),nn.View(input_size,1)(input[1])}
	local right=nn.Linear(rank,input_size)(nn.Linear(input_size,rank)(input[1]))
	local sum=nn.CAddTable(){left,right}
	local res=nn.MM(false,false){nn.View(1,input_size)(input[1]),sum}
	table.insert(output,res)
end
output = nn.JoinTable(1)(output)

model=nn.gModule(input, {output})
res=model:forward({torch.Tensor(3),torch.Tensor(3),torch.Tensor(3)})
print(res)
]]--
--[[
output_size = 2
local input = {}
table.insert(input, nn.Identity()())
-- output_size diag inputs
for i = 1, output_size do
	table.insert(input, nn.Identity()())
end

local output = {}
for i = 1, output_size do
	local left= nn.MM(false,false){nn.Diagonal()(input[i+1]),nn.View(input_size,1)(input[1])}
	local right=nn.Linear(rank,input_size)(nn.Linear(input_size,rank)(input[1]))
	local sum=nn.CAddTable(){left,right}
	local res=nn.MM(false,false){nn.View(1,input_size)(input[1]),sum}
	table.insert(output,res)
end

if output_size > 1 then
	output = nn.JoinTable(1)(output)
else
	output = output[1]
end

m=nn.gModule(input, {output})

--local right_in_sum = nn.MM(false, false){nn.Diagonal()(input[1 + 1]), nn.View(input_size,1)(input[1])}
print(m:forward({torch.Tensor(3), torch.Tensor(3),torch.Tensor(3)}))
]]--

-- create the semantic composition model
input_size = 20
rank = 3
function create_model()
	local input = {}
	table.insert(input, nn.Identity()())
	-- output_size diag inputs
	for i = 1, 2 do
		table.insert(input, nn.Identity()())
	end

	local output = {}
	for i = 1, 2 do
		local left = nn.MM(false, false){nn.Diagonal()(input[i + 1]), nn.View(input_size, 1)(input[1])}
		local right = nn.Linear(rank, input_size)(nn.Linear(input_size, rank)(input[1]))
		local sum = nn.CAddTable(){left, right}
		local res = nn.MM(false, false){nn.View(1, input_size)(input[1]), sum}
		table.insert(output, res)
	end

	if 2 > 1 then
		output = nn.JoinTable(1)(output)
	else
		output = output[1]
	end

	return nn.gModule(input, {output})
end
m=create_model()
print(m:forward({torch.Tensor(20),torch.Tensor(20),torch.Tensor(20)}))
