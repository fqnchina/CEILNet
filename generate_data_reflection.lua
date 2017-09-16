require 'torch'
require 'image'
require 'sys'

imgPath = '/mnt/codes/reflection/COCO_picked_500'
savePath = '/mnt/codes/reflection/models/data/'

files = {}
for file in paths.files(imgPath) do
  if string.find(file,'-input.png') then
    table.insert(files, paths.concat(imgPath,file))
  end
end

for m = 1,#files,2 do
  inputFile1 = files[m+1]
  inputFile2 = files[m]

  local input1 = image.load(inputFile1)
  local input2 = image.load(inputFile2)

  local height1 = input1:size(2)
  local width1 = input1:size(3)
  local height2 = input2:size(2)
  local width2 = input2:size(3)
  if height1 < height2 then
    height = height1-50
  else
    height = height2-50
  end
  if width1 < width2 then
    width = width1-50
  else
    width = width2-50
  end
  if height < 0 or width < 0 then
    goto done
  end
  local xTrans = torch.random(25,input1:size(3)-width)
  local yTrans = torch.random(25,input1:size(2)-height)
  input1 = input1[{{},{yTrans,yTrans+height-1},{xTrans,xTrans+width-1}}]
  local xTrans = torch.random(25,input2:size(3)-width)
  local yTrans = torch.random(25,input2:size(2)-height)
  input2 = input2[{{},{yTrans,yTrans+height-1},{xTrans,xTrans+width-1}}]

  local savColor = string.gsub(inputFile1,imgPath,savePath)
  local window = image.gaussian(11,torch.uniform(2,5)/11)
  window = window:div(torch.sum(window))
  local input2 = image.convolve(input2, window, 'same')
  local inputs = torch.add(input1,input2)

  if inputs:max() > 1 then
    local label_ge1 = torch.gt(inputs,1)
    local mean = torch.mean((inputs-1)[label_ge1],1)[1]*1.3
    input2 = input2 - mean
    input2 = torch.clamp(input2,0,1)

    inputs = torch.add(input1,input2)
    inputs = torch.clamp(inputs,0,1)
    input2 = torch.csub(inputs,input1)
  end

  local savColor = string.gsub(inputFile1,imgPath,savePath)
  local savLabel1 = string.gsub(savColor,'input','label1')
  local savLabel2 = string.gsub(savColor,'input','label2')
  image.save(savColor,inputs)
  image.save(savLabel1,input1)
  image.save(savLabel2,input2)

  ::done::
end