require 'torch'
require 'image'
require 'sys'
require 'cunn'
require 'cutorch'
require 'cudnn'
require 'nngraph'
cudnn.fastest = true
cudnn.benchmark = true

imgPath = '/mnt/codes/learning_to_optimize/testVOC'
savePath = '/mnt/codes/reflection/models/l0'

model = torch.load('/mnt/codes/reflection/models/CEILNet_smooth_L0.net')
model = model:cuda()
model:training()

model_computeEdge = nn.Sequential()
model_computeEdge:add(nn.computeEdge(100))

files = {}
for file in paths.files(imgPath) do
  if string.find(file,'-input.png') then
    table.insert(files, paths.concat(imgPath,file))
  end
end


for _,inputFile in ipairs(files) do

  local labelFile = string.gsub(inputFile,'input','label-L0smooth')
  local labelImg = image.load(labelFile)
  local inputImg = image.load(inputFile)
  local savLabel = string.gsub(labelFile,imgPath,savePath)
  local savColor = string.gsub(inputFile,imgPath,savePath)
  image.save(savLabel,labelImg)
  image.save(savColor,inputImg)

  local height = inputImg:size(2)
  local width = inputImg:size(3)

  local input = torch.CudaTensor(1, 3, height, width)
  input[1] = inputImg:cuda()
  input = input * 255
  local inputC = input:clone()

  local label = torch.CudaTensor(1, 3, height, width)
  label[1] = labelImg:cuda()
  label = label * 255

  local inputs = torch.CudaTensor(1, 4, height, width)
  inputs[{{},{1,3},{},{}}] = input
  inputs[{{},{4},{},{}}] = model_computeEdge:forward(input)
  inputs = inputs - 115
  inputs = {inputs,input}
  local predictions = model:forward(inputs)
  predictions2 = predictions[2]


  for m = 1,3 do
    local numerator = torch.dot(predictions2[1][m], inputC[1][m])
    local denominator = torch.dot(predictions2[1][m], predictions2[1][m])
    local alpha = numerator/denominator
    predictions2[1][m] = predictions2[1][m] * alpha
  end
  
  predictions2 = predictions2/255
  local sav = string.gsub(savColor,'%-input.png','-predict.png')
  image.save(sav,predictions2[1])

  ::done::
end