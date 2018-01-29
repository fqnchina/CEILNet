require 'torch'
require 'image'
require 'sys'
require 'cunn'
require 'cutorch'
require 'cudnn'

imgPath = '/mnt/codes/reflection/reflection_data_blurry_few_400'
savePath = '/mnt/codes/reflection/models/data'

model = torch.load('/mnt/codes/reflection/models/CEILNet_reflection.net')
model = model:cuda()
model:training()
model_edge = nn.computeEdge(0.02)

files = {}
for file in paths.files(imgPath) do
  if string.find(file,'-input.png') then
    table.insert(files, paths.concat(imgPath,file))
  end
end

for _,inputFile in ipairs(files) do
  local inputImg = image.load(inputFile)
  local height = inputImg:size(2)
  local width = inputImg:size(3)
  local input = torch.CudaTensor(1, 3, height, width)
  input[1] = inputImg:cuda()
  input = input * 255

  local inputs = torch.CudaTensor(1, 4, height, width)
  inputs[{{},{1,3},{},{}}] = input
  inputs[{{},{4},{},{}}] = model_edge:forward(input)
  inputs = inputs - 115
  local inputs = {inputs,input}
  local inputC = input:clone()

  local predictions = model:forward(inputs)
  local pred_b = predictions[2]

  for m = 1,3 do
    local numerator = torch.dot(pred_b[1][m], inputC[1][m])
    local denominator = torch.dot(pred_b[1][m], pred_b[1][m])
    local alpha = numerator/denominator
    pred_b[1][m] = pred_b[1][m] * alpha
  end
--  local pred_r = torch.csub(inputC, pred_b)
  
--  for m = 1,3 do
--    local numerator = torch.dot(pred_r[1][m], inputC[1][m])
--    local denominator = torch.dot(pred_r[1][m], pred_r[1][m])
--    local alpha = numerator/denominator
--    pred_r[1][m] = pred_r[1][m] * alpha
--  end

  local savColor = string.gsub(inputFile,imgPath,savePath)

  pred_b = pred_b/255
  local sav = string.gsub(savColor,'input.png','predict1.png')
  image.save(sav,pred_b[1])

 -- -- Our CNN dose not predict refletion layers. The following code computes approximate reflection layers by simply subtracting the predicted background layers from the input images. Note the result so-obtained may not reflect the image structure and appearance of the original reflection scene.
 -- pred_r = pred_r/255
 -- local sav = string.gsub(savColor,'input.png','predict2.png')
 -- image.save(sav,pred_r[1])

  ::done::
end
