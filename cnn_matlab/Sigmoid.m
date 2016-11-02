classdef Sigmoid
    properties
        name;
        input;
        output;
        delta;
        para_a;
    end
    
    methods
        function layer = Sigmoid(name)
            layer.name = name;
        end
        
        function layer = forward(layer, input)
            % Your codes here
            %parameter here is adjustable
            layer.para_a=1;
            layer.input=input;
            layer.output=1./(1+exp(-layer.para_a.*input));

        end
        
        function layer = backprop(layer, delta)
            % Your codes here
                layer.delta = layer.para_a *layer.output .*(1-layer.output) .* delta;
            end 
                
        end
end
