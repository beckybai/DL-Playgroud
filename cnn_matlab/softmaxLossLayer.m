classdef softmaxLossLayer
    
    properties
        name;
        input;
        input_shape;

        output;
        delta;
        loss;
        accuracy;
    end

    methods
        function layer = softmaxLossLayer(name)
            layer.name = name;
        end
        
        function layer = forward(layer, input)
            % Your codes here
            layer.input = input;
            colsum = repmat( sum(exp(layer.input),1) ,[size(input,1),1]);
            layer.output = exp(layer.input) ./ colsum;
            
        end

        function layer = backprop(layer, label)
            % Your codes here
            newdelta = zeros(size(10,size(label,2)));
            for i=1:1:size(label,2)
                newdelta (label(1,i),i) =1;
            end
            
            layer.delta =(newdelta - layer.output);
            
            [~, pred_indx] = max(layer.output);
            layer.accuracy = sum(pred_indx == label) / size(label, 2);
            
            layer.loss = - sum(sum(newdelta .* log(layer.output),1))./size(label,2);
            
            
        end
    end
end


