% Load many 31 channel hyperspectral images
% Format should be [x, y, wavelengths, indexes]
% For information on finding hyperspectral data please see the readme
% For best results all data should be similarly normalized but not contrast
% normalized.
imgs_path = '..\..\..\Hyperspectral DATA\96x96x31\TrainingData\';
% load([imgs_path 'training_data.mat'], 'b');

% imgs_path = '..\..\..\Hyperspectral DATA\POSTER\';
%imgs_path = '..\..\..\Hyperspectral DATA\LEGO\';

load([imgs_path 'training_data.mat'], 'b');
%b的大小为96x96x31x5
% Define dictionary parameters [x, y, wavelengths, number of filters]
% For lightfields, etc. you may want e.g., [x, y, sx, sy, n.o.f]
 kernel_size = [11, 11, size(b,3), 20]; % kernel_size = [11, 11, 31, 10]; % 
% kernel_size = [11, 11, 31, 100];  减少了filter的个数为20
lambda_residual = 1.0;
lambda = 0.01; 
verbose = 'all';



% Filter from local contrast normalization 进行局部对比度归一化
k = fspecial('gaussian',[13 13],3*1.591); 
smooth_init = imfilter(b, k, 'same', 'conv', 'symmetric');%对任意类型的进行高斯滤波，

% Reconstruct & learn filters
fprintf('Doing sparse coding kernel learning for k = %d [%d x %d x %d] kernels.\n\n', kernel_size(4), kernel_size(1), kernel_size(2), kernel_size(3))

% Optimization options
verbose_admm = 'brief';
max_it = 1000;%40;%max_it = 3000;%60; %  
tol = 1e-4;% 1e-3;
init = [];

% Run optimization
tic();
[d, z, Dz, obj]  = admm_learn(b, kernel_size, lambda_residual, lambda, max_it, tol, verbose_admm, init, smooth_init);
%参数为：b输入的原图像、kernel_size为[11, 11, 31, 20]、lambda_residual:1.0
%、lambda:1.0、max_it:1000、tol:1e-4、verbose_admm:brief、init:[]、smooth_init:进行高斯滤波之后的图像。
tt = toc;

% Debug
fprintf('Done dictionary learning! --> Time %2.2f sec.\n\n', tt)

% Save dictionary
save('../Filters/2D-3D-Hyperspectral.mat', 'd', 'z', 'Dz', '-v7.3');



