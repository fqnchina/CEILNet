local computeEdge, parent = torch.class('nn.computeEdge', 'nn.Module')

function computeEdge:__init(scale)
  parent.__init(self)
  self.scale = scale or 255
end

function computeEdge:updateOutput(input)
  self.scale = self.scale or 255
  local height,width = input:size(3),input:size(4)
  input = input / self.scale
  self.xGrad1 = torch.CudaTensor():resizeAs(input):zero()
  self.yGrad1 = torch.CudaTensor():resizeAs(input):zero()
  self.xGrad2 = torch.CudaTensor():resizeAs(input):zero()
  self.yGrad2 = torch.CudaTensor():resizeAs(input):zero()
  self.xGrad1[{{},{},{},{1,width-1}}] = input:narrow(4,2,width-1) - input:narrow(4,1,width-1)
  self.yGrad1[{{},{},{1,height-1},{}}] = input:narrow(3,2,height-1) - input:narrow(3,1,height-1)
  self.xGrad2[{{},{},{},{2,width}}] = input:narrow(4,2,width-1) - input:narrow(4,1,width-1)
  self.yGrad2[{{},{},{2,height},{}}] = input:narrow(3,2,height-1) - input:narrow(3,1,height-1)
  local xGrad = (torch.abs(self.xGrad1) + torch.abs(self.xGrad2))/2
  local yGrad = (torch.abs(self.yGrad1) + torch.abs(self.yGrad2))/2
  self.output = torch.sum(xGrad,2)+torch.sum(yGrad,2)

  return self.output
end

function computeEdge:updateGradInput(input, gradOutput)
  local bs,dim,height,width = input:size(1),input:size(2),input:size(3),input:size(4)
  gradOutput = torch.expand(gradOutput,bs,3,height,width)
  self.gradInput = torch.CudaTensor():resizeAs(input):zero()

  local neg = torch.ge(self.xGrad1,0):cuda()
  neg[torch.eq(neg,0)] = -1
  local gradx1 = torch.cmul(gradOutput,neg)/2
  gradx1 = gradx1[{{},{},{},{1,width-1}}]

  local temp1 = self.gradInput:narrow(4,1,1)
  temp1:add(-gradx1:narrow(4,1,1))
  local temp2 = self.gradInput:narrow(4,width,1)
  temp2:add(gradx1:narrow(4,width-1,1))
  local temp3 = self.gradInput:narrow(4,2,width-2)
  temp3:add(gradx1:narrow(4,1,width-2)-gradx1:narrow(4,2,width-2))

  local neg = torch.ge(self.xGrad2,0):cuda()
  neg[torch.eq(neg,0)] = -1
  local gradx2 = torch.cmul(gradOutput,neg)/2
  gradx2 = gradx2[{{},{},{},{2,width}}]

  local temp1 = self.gradInput:narrow(4,1,1)
  temp1:add(-gradx2:narrow(4,1,1))
  local temp2 = self.gradInput:narrow(4,width,1)
  temp2:add(gradx2:narrow(4,width-1,1))
  local temp3 = self.gradInput:narrow(4,2,width-2)
  temp3:add(gradx2:narrow(4,1,width-2)-gradx2:narrow(4,2,width-2))

  local neg = torch.ge(self.yGrad1,0):cuda()
  neg[torch.eq(neg,0)] = -1
  local grady1 = torch.cmul(gradOutput,neg)/2
  grady1 = grady1[{{},{},{1,height-1},{}}]

  local temp1 = self.gradInput:narrow(3,1,1)
  temp1:add(-grady1:narrow(3,1,1))
  local temp2 = self.gradInput:narrow(3,height,1)
  temp2:add(grady1:narrow(3,height-1,1))
  local temp3 = self.gradInput:narrow(3,2,height-2)
  temp3:add(grady1:narrow(3,1,height-2)-grady1:narrow(3,2,height-2))

  local neg = torch.ge(self.yGrad2,0):cuda()
  neg[torch.eq(neg,0)] = -1
  local grady2 = torch.cmul(gradOutput,neg)/2
  grady2 = grady2[{{},{},{2,height},{}}]
  
  local temp1 = self.gradInput:narrow(3,1,1)
  temp1:add(-grady2:narrow(3,1,1))
  local temp2 = self.gradInput:narrow(3,height,1)
  temp2:add(grady2:narrow(3,height-1,1))
  local temp3 = self.gradInput:narrow(3,2,height-2)
  temp3:add(grady2:narrow(3,1,height-2)-grady2:narrow(3,2,height-2))
  
  -- self.gradInput = self.gradOutput:clone()/2
  
  -- local label = torch.eq(gradOutput,0)
  -- self.gradInput[label] = 0
  self.gradInput = self.gradInput / self.scale

  return self.gradInput
end

function computeEdge:clearState()
  nn.utils.clear(self, 'xGrad1', 'xGrad2', 'yGrad1', 'yGrad2')
  return parent.clearState(self)
end