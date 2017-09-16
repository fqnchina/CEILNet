require 'nn'
require 'optim'
require 'torch'
require 'cutorch'
require 'cunn'
require 'image'
require 'sys'
require 'nngraph'
require 'cudnn'
cudnn.fastest = true
cudnn.benchmark = true

--GPU 2
model1 = torch.load('/mnt/codes/reflection/netfiles/model_reflection_e_cnn_40.net')
model2 = torch.load('/mnt/codes/reflection/netfiles/model_reflection_i_cnn_40.net')
local edge_var = 0.02

model_sub1 = nn.Sequential()
model_sub1:add(nn.SelectTable(1))
model_sub1:add(model1)
model_sub1:add(nn.MulConstant(edge_var))

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

model_edge = nn.computeEdge(edge_var)


postfix = 'reflection_combine'
max_iters = 5
batch_size = 2

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

local file = '/mnt/codes/reflection/models/training_reflection_combine.lua'
local f = io.open(file, "rb")
local line = f:read("*all")
f:close()
print('*******************train file*******************')
print(line)
print('*******************train file*******************')

local file = '/mnt/data/VOC2012_224_train_png.txt'
local trainSet = {}
local f = io.open(file, "rb")
while true do
  local line = f:read()
  if line == nil then break end
  table.insert(trainSet, line)
end
f:close()
local trainsetSize = #trainSet
if trainsetSize % 2 == 1 then
  trainsetSize = trainsetSize - 1
end
trainsetSize= 1000

local file = '/mnt/data/VOC2012_224_test_png.txt'
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
      for i = 1,size,2 do
        local inputFile1 =  trainSet[order[t+i-1]]
        local inputFile2 = trainSet[order[t+i]]
        local tempInput1 = image.load(inputFile1)
        local tempInput2 = image.load(inputFile2)
        local height = tempInput1:size(2)
        local width = tempInput1:size(3)
        local input1 = torch.CudaTensor(1, 3, height, width)
        local input = torch.CudaTensor(1, 3, height, width)
        local inputs = torch.CudaTensor(1, 4, height, width)

        local window = image.gaussian(11,torch.uniform(2,5)/11)
        window = window:div(torch.sum(window))
        local tempInput2 = image.convolve(tempInput2, window, 'same')

        local tempInput1 = tempInput1:cuda()
        local tempInput2 = tempInput2:cuda()
        tempInput = torch.add(tempInput1,tempInput2)
        if tempInput:max() > 1 then
          local label_ge1 = torch.gt(tempInput,1)
          tempInput2 = tempInput2 - torch.mean((tempInput-1)[label_ge1],1)[1]*1.3
          tempInput2 = torch.clamp(tempInput2,0,1)
          tempInput = torch.add(tempInput1,tempInput2)
          tempInput = torch.clamp(tempInput,0,1)
        end

        input1[1] = tempInput1
        input[1] = tempInput
        input1 = input1 * 255
        input = input * 255
        inputs[{{},{1,3},{},{}}] = input
        inputs[{{},{4},{},{}}] = model_edge:forward(input)
        inputs = inputs - 115
        local inputs = {inputs,input}
        local xGrad1 = input1:narrow(4,2,width-1) - input1:narrow(4,1,width-1)
        local yGrad1 = input1:narrow(3,2,height-1) - input1:narrow(3,1,height-1)
        local labels = {model_edge:forward(input1)*edge_var,input1,xGrad1,yGrad1}

        local pred = model:forward(inputs)
        local tempLoss =  criterion:forward(pred, labels)
        loss = loss + tempLoss
        local grad = criterion:backward(pred, labels)

        model:backward(inputs, grad)
      end
      gradParameters:div(size/2)
      loss = loss/(size/2)

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
      for i = 1,size,2 do
        local inputFile1 = testSet[testCount]
        local inputFile2 = testSet[testCount+1]
        local tempInput1 = image.load(inputFile1)
        local tempInput2 = image.load(inputFile2)
        local height = tempInput1:size(2)
        local width = tempInput1:size(3)
        local input1 = torch.CudaTensor(1, 3, height, width)
        local input = torch.CudaTensor(1, 3, height, width)
        local inputs = torch.CudaTensor(1, 4, height, width)

        local window = image.gaussian(11,torch.uniform(2,5)/11)
        window = window:div(torch.sum(window))
        local tempInput2 = image.convolve(tempInput2, window, 'same')

        local tempInput1 = tempInput1:cuda()
        local tempInput2 = tempInput2:cuda()
        tempInput = torch.add(tempInput1,tempInput2)
        if tempInput:max() > 1 then
          local label_ge1 = torch.gt(tempInput,1)
          tempInput2 = tempInput2 - torch.mean((tempInput-1)[label_ge1],1)[1]*1.3
          tempInput2 = torch.clamp(tempInput2,0,1)
          tempInput = torch.add(tempInput1,tempInput2)
          tempInput = torch.clamp(tempInput,0,1)
        end

        input1[1] = tempInput1
        input[1] = tempInput
        input1 = input1 * 255
        input = input * 255
        inputs[{{},{1,3},{},{}}] = input
        inputs[{{},{4},{},{}}] = model_edge:forward(input)
        inputs = inputs - 115
        local inputs = {inputs,input}
        local xGrad1 = input1:narrow(4,2,width-1) - input1:narrow(4,1,width-1)
        local yGrad1 = input1:narrow(3,2,height-1) - input1:narrow(3,1,height-1)
        local labels = {model_edge:forward(input1)*edge_var,input1,xGrad1,yGrad1}

        local pred = model:forward(inputs)
        local tempLoss =  criterion:forward(pred, labels)
        loss = loss + tempLoss
        testCount = testCount + 2
      end
      loss = loss/(size/2)
      testcount = testcount + 1
      current_testloss = current_testloss + loss

      print(string.format('TestIter: %d Current loss: %4f', iter, loss))
    end

    totalNum = totalNum + 1
    if totalNum % 500 == 0 then
      local filename = string.format('/mnt/codes/reflection/models/model_%s_multi100_%d.net',postfix,totalNum/100)
      model:clearState()
      torch.save(filename, model)
    end
  end

  -- normalize loss
  return current_loss / count, current_testloss / testcount
end

step(batch_size)