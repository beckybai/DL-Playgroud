classdef euclideanLossLayer
    
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
        function layer = euclideanLossLayer(name)
            layer.name = name;
        end
    
        function layer = forward(layer, input)
            % Your codes here
            layer.input = input;
            layer.output = input;
        end
        
         function layer = backprop(layer, label)
            % Your codes here
            %target newdelta
            newdelta = zeros(size(10,size(label,2)));
            for i=1:1:size(label,2)
                newdelta (label(1,i),i) =1;
            end
            
            layer.loss = 0.5/size(label,2)* sum(diag((newdelta-layer.output)'*(newdelta-layer.output))) ;
            
            [~, pred_indx] = max(layer.output);
            layer.accuracy = sum(pred_indx == label) / size(label, 2);
            
            layer.delta = newdelta- layer.output;
         end
    end
end