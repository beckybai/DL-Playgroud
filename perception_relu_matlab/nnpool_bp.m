function down_delta = nnpool_bp(input, delta, kernel_size, pad)
    % Your codes here
    %repmat(delta,1,2,1,1,2);
    %x = repmat(repmat(delta,1,1,1,1,2),1,2,1,1,1);
    %x = permute(x,[5,2,1,3,4]);
    %x = kron( input,ones(2,2,1,1));
    down_delta = zeros(size(input,1),size(input,2),size(input,3),size(input,4));
    for i = 1:1:size(input,3)
        for j = 1:1:size(input,4)
            %kron(delta(:,:,i,j),ones(2,2))
            %down_delta (:,:,i,j)
            down_delta (:,:,i,j) = kron(delta(:,:,i,j),ones(2,2));
            %down_delta (:,:,i,j)
        end
    end
    
    %down_delta = reshape(x,size(input,1),size(input,2),size(input,3),size(input,4))/power(kernel_size,2);
end