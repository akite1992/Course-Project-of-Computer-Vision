
addpath miniplacesCNN
addpath miniplacesCNN/matconvnet
run miniplacesCNN/matconvnet/matlab/vl_setupnn.m
opts.useBnorm = false ;
opts.train.batchSize = 128;
opts.train.numSubBatches = 1 ;
opts.train.continue = true ;
opts.numFetchThreads = 12 ;
opts.lite = false ;
opts.train.prefetch = true ;
opts.train.sync = false ;
opts.train.cudnn = true ;
opts.train.gpus = 1;
opts.train.learningRate =[0.01*ones(1,20),0.001*ones(1,20),0.0001*ones(1,10)];
opts.train.numEpochs = numel(opts.train.learningRate) ;
opts.train.expDir = fullfile('data','baseline7');
imdb=load('imdb.mat');
useGpu = numel(opts.train.gpus) > 0 ;
%imdb.images.data_mean=mean(imdb.images.data,4);
%save('imdb.mat','images');
net1 = sample_refNet_initial();
bopts.numThreads=3;
imageStatsPath = fullfile(opts.train.expDir, 'imageStats.mat') ;
if exist(imageStatsPath)
  load(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
else
  [averageImage, rgbMean, rgbCovariance] = getImageStats(imdb, bopts) ;
  save(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
end
net1.normalization.averageImage = rgbMean ;
[v,d] = eig(rgbCovariance) ;
bopts.transformation = 'stretch' ;
bopts.averageImage = rgbMean ;
bopts.rgbVariance = 0.1*sqrt(d)*v';
bopts.numAugments=2;
useGpu = numel(opts.train.gpus) > 0 ;
fn = getBatchDagNNWrapper(bopts,useGpu) ;
opts.train = rmfield(opts.train, {'sync', 'cudnn'}) ;
net = dagnn.DagNN.fromSimpleNN(net1, 'canonicalNames', true) ;
net.addLayer('error', dagnn.Loss('loss', 'classerror'), ...
                 {'prediction','label'}, 'top1error') ;
info= cnn_train_dag(net, imdb, fn, ...
   opts.train, ...
   'val', find(imdb.images.set == 2));
save(fullfile('data','net.mat'),'net','info');