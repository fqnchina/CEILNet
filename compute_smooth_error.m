totalPSNR = 0;
totalSSIM = 0;
totalMSE = 0;
count = 0;

s = dir('/mnt/codes/reflection/models/l0/*-input.png');
for n = 1:length(s)
    inputName = ['/mnt/codes/reflection/models/l0/' s(n,1).name(1:end-10) '-predict.png'];
    labelName = ['/mnt/codes/reflection/models/l0/' s(n,1).name(1:end-10) '-label-L0smooth.png'];
    disp(labelName);
    if ~exist(inputName, 'file')
        continue;
    end
    input = imread(inputName);
    label = imread(labelName);

    mse = immse(input, label);
    totalMSE = totalMSE + mse;

    [peaksnr, snr] = psnr(input, label);
    totalPSNR = totalPSNR + peaksnr;

    input = rgb2gray(input);
    label = rgb2gray(label);
    [ssimval, ssimmap] = ssim(input, label);
    totalSSIM = totalSSIM + ssimval;
    
    count = count + 1;
end
totalMSE = totalMSE/count;
totalPSNR = totalPSNR/count;
totalSSIM = totalSSIM/count;
disp(sprintf('mse: %f, psnr: %f, ssim: %f',totalMSE,totalPSNR,totalSSIM));

