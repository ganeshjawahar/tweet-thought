local Diagonal, parent = torch.class('nn.Diagonal', 'nn.Module')

function Diagonal:__init(size, gpu)
	parent.__init(self)
	self.output = torch.zeros(size, size)
	if torch.typename(input) == 'torch.CudaTensor' then self.output = self.output:cuda() end
	self.gradInput = torch.Tensor(size)
	if torch.typename(input) == 'torch.CudaTensor' then self.gradInput = self.gradInput:cuda() end
end

function Diagonal:updateOutput(input)
	self.output:zero()
	for i = 1, input:size(1) do
		self.output[i][i] = input[i]
	end
	return self.output
end

function Diagonal:updateGradInput(input, gradOutput)	
	for i = 1, input:size(1) do
		self.gradInput[i] = gradOutput[i][i]
	end
	return self.gradInput  
end