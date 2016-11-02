%  =========================================================
%  A typical architecture of CNN is organized as follows:
%
%   - add convlayer to model
%   - add relu to model
%   - add poolLayer to model
%
%  and then repeat the above mini-structure as many times as you like.
%  
%  Before the final classification operation
%
%   - add fcLayer to model
%
%  and then add softmaxLossLayer
%
%
%  ATTENTION!
%    
%    fcLayer should implement matrix reshape operation to align the bottom input
%    as a 2D matrix with dimension (H * W * C) x N
%  ==========================================================

clear;
model = Network();
model = add(model, convLayer('conv1',3,1,8,1,0.1,0.02,0.005,0.9));
model = add(model, Relu('relu1'));
model = add(model, poolLayer('pool1',2,0));
model = add(model, convLayer('conv2',3,8,4,1,0.1,0.01,0.005,0.9));
model = add(model, Relu('relu2'));
model = add(model, poolLayer('pool2',2,0));
model = add(model, fcLayer('fc3',196,10,0.1,0.001,0.05,0.9));
model = add(model, softmaxLossLayer('softmax'));

load('mnist.mat');
train_x = train_x(:,1:1000);
train_y = train_y(:,1:1000);
test_x = test_x(:,1:1000);
test_y = test_y(:,1:1000);

mean_value = mean(train_x, 2);
train_x = bsxfun(@minus, train_x, mean_value);
test_x = bsxfun(@minus, test_x, mean_value);

train_x = permute(reshape(train_x, 28, 28, 1, []), [2, 1, 3, 4]);
test_x = permute(reshape(test_x, 28, 28, 1, []), [2, 1, 3, 4]);

model = Fit_cnn(model, train_x, train_y, test_x, test_y,...
    50, 10, 1, 10, 10);