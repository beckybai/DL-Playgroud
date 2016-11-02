function output = nnpool(input, kernel_size, pad)
    % Your codes here
    output = zeros(size(input,1)/kernel_size,size(input,2)/kernel_size,size(input,3),size(input,4));
    ocn = size(output,1);
    orn = size(output,2); 
    for i=1:1:size(input,3)
        for j = 1:1:size(input,4)
            %output (:,:,i,j) is a picture  
            %size(mean(im2col(input(:,:,i,j),[2 2],'distinct')))
            %size(output(:,:,i,j))
            output(:,:,i,j) = reshape(mean(im2col(input(:,:,i,j),[2 2],'distinct')),ocn,orn);
        end
    end
end
            