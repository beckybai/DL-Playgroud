function output = nnconv(input, kernel_size, num_output, W, b, pad)
    % Your codes here
    % input is a 4d matrix
    % kernel_size //num_output is a number
    % W :  kernel_size, kernel_size, num_input, num_output;
    % b :  num_output, 1,
    % pad 
    output = zeros(size(input,1),size(input,2),(num_output),size(input,4));
    
    for k = 1:1:num_output   %the filter's number && output
        for i = 1:1:size(input,4)
            for j = 1:1:size(input,3)%channal numbers input 3 == num_input
                rotw = rot90(rot90(W(:,:,j,k)));
                output(:,:,k,i) = output(:,:,k,i) + conv2(input(:,:,j,i),rotw,'same');
            end
            %output(:,:,k,i) = output(:,:,k,i)./(size(input,3)) +b(k); % maybe here exists some problems
            output(:,:,k,i) = output(:,:,k,i) + b(k);
        end
    end   
end