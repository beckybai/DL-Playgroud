function [down_delta, grad_W, grad_b] = nnconv_bp(input, delta, W, b, pad)
% Your codes here

%kernel_size, kernel_size, num_input, num_output
down_delta = zeros(size(input));
%size(down_delta)
%size(delta)
%这块跟PPT上的有点不一样，PPT上往后卷积用的是一个valid 卷积，但是代码中用的是same (仅仅对于3*3的filter适用。
for j = 1:1:size(input,4)  %different pictures.
    for k =1:1:size(W,3)   %different features
        for i=1:1:size(W,4) %one filter for different feature picture of one thing.
            down_delta(:,:,k,j) =  down_delta(:,:,k,j) + conv2(delta(:,:,i,j),W(:,:,k,i),'same');
        end
        down_delta(:,:,k,j) = down_delta(:,:,k,j)./size(W,4); % bu tong de filter dui tong yi zhang tu jinxing chuli
    end
end

%这块跟PPT上的有点不一样，PPT上往后卷积用的是一个valid 卷积，但是代码中用的是same (仅仅对于3*3的filter适用。
newinput = zeros (size(input,1)+2,size(input,2)+2,size(input,3),size(input,4));
csize = size(newinput,1);
newinput(2:csize-1,2:csize-1,:,:) = input;

%
%size(newinput)
%size(input)

%    size(down_delta)
%size(delta)
%size(W,4)
%size(W,3)

bsize = power(size(delta,1),2);

grad_W = zeros(size(W));
grad_b = zeros(size(b));

for i = 1:1:size(W,4)
    for j = 1:1:size(W,3)
        for k = 1:1:size(input,4)
            grad_W(:,:,j,i) = grad_W(:,:,j,i) + conv2(newinput(:,:,j,k),rot90(rot90(delta(:,:,i,k))),'valid');
        end
        grad_W(:,:,j,i) = grad_W(:,:,j,i)/size(input,4);
    end
    x = mean(delta(:,:,i,:),4);
    grad_b(i) = sum(x(:))/bsize;
end





end