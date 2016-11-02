classdef fcLayer

    properties
        name;
        num_input;
        num_output;

        learning_rate;
        weight_decay;
        momentum;

        input;
        input_shape;
        inputforward;
        output;
        W;
        b;
       
        grad_W;
        grad_b;
        diff_W; % last update for W
        diff_b; % last update for b
        delta;
    end

    methods
        function layer = fcLayer(name, num_input, num_output, init_std, learning_rate, weight_decay, momentum)
            layer.name = name;
            layer.num_input = num_input;
            layer.num_output = num_output;
            layer.W = single(random('norm', 0, init_std, num_output, num_input));
            layer.b = zeros(num_output, 1, 'single');
            layer.diff_W = zeros(size(layer.W), 'single');
            layer.diff_b = zeros(size(layer.b), 'single');

            layer.learning_rate = learning_rate;
            layer.weight_decay = weight_decay;
            layer.momentum = momentum;
        end

        function layer = forward(layer, input)
            % Your codes here
            %y=wx+b
            layer.input = input;
            layer.inputforward = reshape(input,layer.num_input,[]);% 49*4= 196
            layer.output = layer.W * layer.inputforward;
            
          % can change for better performance.  
          for i=1:1:size(layer.inputforward,2)
                layer.output(:,i)=layer.output(:,i)+ layer.b;
          end
   
        end

        function layer = backprop(layer, delta)
            % Your codes here
            % delta is a matrix : (eg: 10*100)
            layer.grad_W = (delta* layer.inputforward')/size(delta,2);
            layer.grad_b = mean(delta')';%:get the idea from my classmate HaoWang
            
            %this is a little different from the text book.
            layer.diff_W = layer.momentum * layer.diff_W - layer.learning_rate * (layer.grad_W + layer.weight_decay * layer.grad_W);
            layer.diff_b = layer.momentum * layer.diff_b - layer.learning_rate * (layer.grad_b + layer.weight_decay * layer.grad_b);

            %use the weight factor to prepare the delta
            layer.W = layer.W + layer.diff_W;
            layer.b = layer.b + layer.diff_b;
            
            layer.delta = reshape( layer.W'*delta,size(layer.input)); 
            
        end
    end
end
