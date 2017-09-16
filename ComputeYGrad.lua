local ComputeYGrad, parent = torch.class('nn.ComputeYGrad', 'nn.Module')

function ComputeYGrad:__init()
   parent.__init(self)
end

function ComputeYGrad:updateOutput(input)
   local height = input:size(3)
   self.output = torch.csub(input:narrow(3,2,height-1),input:narrow(3,1,height-1))
   return self.output
end

function ComputeYGrad:updateGradInput(input, gradOutput)
   local height = input:size(3)
   self.gradInput:resizeAs(input)
   local temp1 = self.gradInput:narrow(3,1,1)
   temp1:copy(-gradOutput:narrow(3,1,1))
   local temp2 = self.gradInput:narrow(3,height,1)
   temp2:copy(gradOutput:narrow(3,height-1,1))
   local temp3 = self.gradInput:narrow(3,2,height-2)
   temp3:copy(torch.csub(gradOutput:narrow(3,1,height-2),gradOutput:narrow(3,2,height-2)))
   
   return self.gradInput
end