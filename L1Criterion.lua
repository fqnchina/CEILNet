local L1Criterion, parent = torch.class('nn.L1Criterion', 'nn.Criterion')

function L1Criterion:__init()
   parent.__init(self)
end

function L1Criterion:updateOutput(input, target)
   self.output = torch.mean(torch.abs(torch.csub(input,target)))
   return self.output
end

function L1Criterion:updateGradInput(input, target)
   local num = input:size(1)*input:size(2)*input:size(3)*input:size(4)
   self.gradInput = torch.CudaTensor():resizeAs(input):fill(-1)
   self.gradInput[torch.ge(torch.csub(input,target),0)] = 1
   self.gradInput = self.gradInput/num
   return self.gradInput
end
