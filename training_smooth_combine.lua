require 'nn'
require 'optim'
require 'torch'
require 'cutorch'
require 'cunn'
require 'image'
require 'sys'
require 'cudnn'
cudnn.fastest = true
cudnn.benchmark = true

--GPU 5
model1 = torch.load('/mnt/codes/reflection/models/model_smooth_L0_ecnn_25.net')
model2 = torch.load('/mnt/codes/reflection/models/model_smooth_L0_icnn_25.net')

model_sub1 = nn.Sequential()
model_sub1:add(nn.SelectTable(1))
model_sub1:add(model1)

cont = nn.ConcatTable()
cont:add(nn.SelectTable(2))
cont:add(model_sub1)

model_sub2 = nn.Sequential()
model_sub2:add(nn.JoinTable(2))
model_sub2:add(nn.AddConstant(-115))
model_sub2:add(model2)

cont2 = nn.ConcatTable()
cont2:add(nn.SelectTable(2))
cont2:add(model_sub2)

model = nn.Sequential()
model:add(cont)
model:add(cont2)
model:add(nn.FlattenTable())

criterion = nn.ParallelCriterion():add(nn.MSECriterion(),0.4):add(nn.MSECriterion(),0.2):add(nn.L1Criterion(),0.4):add(nn.L1Criterion(),0.4)
model = model:cuda()
criterion = criterion:cuda()

model_edge = nn.computeEdge(100)



postfix = 'smooth_L0_combine'
max_iters = 40
batch_size = 1

model:training()
collectgarbage()

parameters, gradParameters = model:getParameters()

sgd_params = {
  learningRate = 1e-2,
  learningRateDecay = 1e-8,
  weightDecay = 0.0005,
  momentum = 0.9,
  dampening = 0,
  nesterov = true
}

adam_params = {
  learningRate = 1e-3,
  weightDecay = 0.0005,
  beta1 = 0.9,
  beta2 = 0.999
}

rmsprop_params = {
  learningRate = 1e-2,
  weightDecay = 0.0005,
  alpha = 0.9
}

-- Log results to files
savePath = '/mnt/codes/reflection/models/'

local file = '/mnt/codes/reflection/models/training_smooth_combine.lua'
local f = io.open(file, "rb")
local line = f:read("*all")
f:close()
print('*******************train file*******************')
print(line)
print('*******************train file*******************')


local file = '/mnt/codes/reflection/models/VOC2012_fullsize_L0_train.txt'
local trainSet = {}
local f = io.open(file, "rb")
while true do
  local line = f:read()
  if line == nil then break end
  table.insert(trainSet, line)
end
f:close()
local trainsetSize = #trainSet

local file = '/mnt/codes/reflection/models/VOC2012_fullsize_L0_test.txt'
local testSet = {}
local f = io.open(file, "rb")
while true do
  local line = f:read()
  if line == nil then break end
  table.insert(testSet, line)
end
f:close()
local testsetSize = #testSet

local iter = 0
local totalNum = 0
local epoch_judge = false
step = function(batch_size)
  local testCount = 1
  local current_loss = 0
  local current_testloss = 0
  local count = 0
  local testcount = 0
  batch_size = batch_size or 4
  local order = torch.randperm(trainsetSize)

  for t = 1,trainsetSize,batch_size do
    iter = iter + 1
    local size = math.min(t + batch_size, trainsetSize + 1) - t

    local feval = function(x_new)
      -- reset data
      if parameters ~= x_new then parameters:copy(x_new) end
      gradParameters:zero()

      local loss = 0
      for i = 1,size do
        local inputFile =  trainSet[order[t+i-1]]
        local labelFile = string.gsub(inputFile,'input','label')
        local tempInput = image.load(inputFile)
        local tempLabel = image.load(labelFile)
        local height = tempInput:size(2)
        local width = tempInput:size(3)
        local input = torch.CudaTensor(1, 3, height, width)
        local label = torch.CudaTensor(1, 3, height, width)
        local inputs = torch.CudaTensor(1, 4, height, width)

        input[1] = tempInput
        label[1] = tempLabel
        input = input * 255
        label = label * 255
        inputs[{{},{1,3},{},{}}] = input
        inputs[{{},{4},{},{}}] = model_edge:forward(input)
        inputs = inputs - 115
        inputs = {inputs,input}

        local xGrad = label:narrow(4,2,width-1) - label:narrow(4,1,width-1)
        local yGrad = label:narrow(3,2,height-1) - label:narrow(3,1,height-1)
        local labels = {model_edge:forward(label),label,xGrad,yGrad}
        
        local pred = model:forward(inputs)
        local tempLoss =  criterion:forward(pred, labels)
        loss = loss + tempLoss
        local grad = criterion:backward(pred, labels)

        model:backward(inputs, grad)
      end
      gradParameters:div(size)
      loss = loss/size

      return loss, gradParameters
    end
    
    if epoch_judge then
      adam_params.learningRate = adam_params.learningRate*0.1
      _, fs, adam_state_save = optim.adam_state(feval, parameters, adam_params, adam_params)
      epoch_judge = false
    else
      _, fs, adam_state_save = optim.adam_state(feval, parameters, adam_params)
    end

    count = count + 1
    current_loss = current_loss + fs[1]
    print(string.format('Iter: %d Current loss: %4f', iter, fs[1]))

    if iter % 20 == 0 then
      local loss = 0
      for i = 1,size do
        local inputFile = testSet[testCount]
        local labelFile = string.gsub(inputFile,'input','label')
        local tempInput = image.load(inputFile)
        local tempLabel = image.load(labelFile)
        local height = tempInput:size(2)
        local width = tempInput:size(3)
        local input = torch.CudaTensor(1, 3, height, width)
        local label = torch.CudaTensor(1, 3, height, width)
        local inputs = torch.CudaTensor(1, 4, height, width)

        input[1] = tempInput
        label[1] = tempLabel
        input = input * 255
        label = label * 255
        inputs[{{},{1,3},{},{}}] = input
        inputs[{{},{4},{},{}}] = model_edge:forward(input)
        inputs = inputs - 115
        inputs = {inputs,input}

        local xGrad = label:narrow(4,2,width-1) - label:narrow(4,1,width-1)
        local yGrad = label:narrow(3,2,height-1) - label:narrow(3,1,height-1)
        local labels = {model_edge:forward(label),label,xGrad,yGrad}
        
        local pred = model:forward(inputs)
        local tempLoss =  criterion:forward(pred, labels)
        loss = loss + tempLoss
        testCount = testCount + 1
      end
      loss = loss/size
      testcount = testcount + 1
      current_testloss = current_testloss + loss

      print(string.format('TestIter: %d Current loss: %4f', iter, loss))
    end
  end

  -- normalize loss
  return current_loss / count, current_testloss / testcount
end

netfiles = '/mnt/codes/reflection/models/'
timer = torch.Timer()
do
  for i = 1,max_iters do
    localTimer = torch.Timer()
    local loss,testloss = step(batch_size,i)
    if i == 20 then
      epoch_judge = true
    end
    print(string.format('Epoch: %d Current loss: %4f', i, loss))
    print(string.format('Epoch: %d Current test loss: %4f', i, testloss))

    local filename = string.format('%smodel_%s_%d.net',netfiles,postfix,i)
    model:clearState()
    torch.save(filename, model)
    local filename = string.format('%sstate_%s_%d.t7',netfiles,postfix,i)
    torch.save(filename, adam_state_save)
    print('Time elapsed (epoch): ' .. localTimer:time().real/(3600) .. ' hours')
  end
end
print('Time elapsed: ' .. timer:time().real/(3600*24) .. ' days')
