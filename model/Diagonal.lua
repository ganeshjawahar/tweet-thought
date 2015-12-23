local Diagonal, parent = torch.class('nn.Diagonal', 'nn.Module')

function Diagonal:updateOutput(input)
	self.output = input:diag()
	return self.output
end

function Diagonal:updateGradInput(input, gradOutput)
	self.gradInput = torch.diag(gradOutput)
	return self.gradInput  
end