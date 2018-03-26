%% SCRIPT: NLM_NAIVE_KERNEL
%
% GPU naive kernel for NLM filter
%
% DEPENDENCIES
%
%  nlmKernel.cu
%% 
  
  clear variables;
  clear all %#ok
  close all

  %% ARRAY PARAMETERS
  n = 512;
  m = 512;

  %% NLM FILTER PARAMETERS
  filtSigma = 0.02;

  %% CUDA PARAMETERS
  
  % maxThreadsPerBlock depend on the GPU
  maxThreadsPerBlock = 1024;

  %% (BEGIN)

  fprintf('...begin %s...\n',mfilename);  

 %%------------------PIPELINE------------------------%%
  %% PARAMETERS
  % input image
  strImgVar = 'house';
  
  
  % noise
  noiseParams = {'gaussian', ...
                 0,...
                 0.001};
  
  % filter sigma value
  filtSigma = 0.02;
  patchSigma = filtSigma;
  
  %% USEFUL FUNCTIONS

  % image normalizer
  normImg = @(I) (I - min(I(:))) ./ max(I(:) - min(I(:)));

  %% (BEGIN)

  fprintf('...begin %s...\n',mfilename);  

  %% INPUT DATA
  
  fprintf('...loading input data...\n')
  
  % I = mat2gray(imread('../data/image_128.jpg'));
  % I = mat2gray(imread('../data/image_256.png'));
  I = mat2gray(imread('../data/lena_512.png'));

  % ioImg = matfile('../data/house.mat');
  % I = ioImg.(strImgVar);
  
  imwrite(I, '../data/cuda_lena_1.png');
  
  %% PREPROCESS
  
  fprintf(' - normalizing image...\n')
  I = normImg( I );
  
  figure('Name','Original Image');
  imagesc(I); axis image;
  colormap gray;
  imwrite(I, '../data/cuda_lena_2_norm.png');
  
  %% NOISE
  
  fprintf(' - applying noise...\n')
  J = imnoise( I, noiseParams{:} );
  figure('Name','Noisy-Input Image');
  imagesc(J); axis image;
  colormap gray;
  imwrite(J, '../data/cuda_lena_3_noise.png')

  %%------------------PIPELINE END------------------------%%



  %% KERNEL
  
  k = parallel.gpu.CUDAKernel( '../cuda/nlm_naive/nlmNaiveKernel.ptx', ...
                              '../cuda/nlm_naive/nlmNaiveKernel.cu');

  threadsPerBlock = n;    %cols
 

  k.ThreadBlockSize = min(threadsPerBlock, maxThreadsPerBlock);
  k.GridSize        = ceil(m*n/k.ThreadBlockSize(1)); 

  %% DATA
  
  B = zeros([m n], 'gpuArray');
  fprintf('Naive NLM...\n');

  tic;
  B = gather( feval(k, n, J, B, filtSigma) );
  toc
  

  % fprintf('This implementation currently supports only NxN images. Sorry...\n');
  imwrite(B, '../data/cuda_lena_4_naive.png')

  
  %% (END)

  fprintf('Done %s...\n',mfilename);
