local ComputeXGrad, parent = torch.class('nn.ComputeXGrad', 'nn.Module')

function ComputeXGrad:__init()
   parent.__init(self)
end

function ComputeXGrad:updateOutput(input)
   local width = input:size(4)
--   print(input:size(1),input:size(2),input:size(3),input:size(4))
   self.output = torch.csub(input:narrow(4,2,width-1), input:narrow(4,1,width-1))
   return self.output
end

function ComputeXGrad:updateGradInput(input, gradOutput)
   local width = input:size(4)
   self.gradInput:resizeAs(input)
   local temp1 = self.gradInput:narrow(4,1,1)
   temp1:copy(-gradOutput:narrow(4,1,1))
   local temp2 = self.gradInput:narrow(4,width,1)
   temp2:copy(gradOutput:narrow(4,width-1,1))
   local temp3 = self.gradInput:narrow(4,2,width-2)
   temp3:copy(torch.csub(gradOutput:narrow(4,1,width-2),gradOutput:narrow(4,2,width-2)))
   
   return self.gradInput
end