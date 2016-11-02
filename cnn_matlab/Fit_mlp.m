function net = Fit_mlp(net, train_data, train_label, test_data, test_label, ...
                       batchsize, epoch, display_freq, test_freq, snapshot_freq)
    
                   fprintf('sigmoid-layer1 %f- layer2 %f',net.layer_list{1}.learning_rate,net.layer_list{3}.learning_rate);
                   plotloss=[];%for plot
                   plotaccuracy=[];%for plot
    now = clock;
    fprintf('[%02d:%02d:%05.2f] Start training\n', now(4), now(5), now(6));
    num_input = size(train_data, 2);
    num_test_input = size(test_data, 2);
    iters = ceil(num_input / batchsize);
    for k = 1:epoch
        loss = [];
        accuracy = [];
        for i = 1:iters
            net = Forward(net, train_data(:,(i-1)*batchsize+1:min(i*batchsize, num_input)));
            net = Backpropagation(net, train_label(:,(i-1)*batchsize+1:min(i*batchsize, num_input)));
            loss = [loss, net.layer_list{net.num_layer}.loss];
            accuracy = [accuracy, net.layer_list{net.num_layer}.accuracy];
        end 

        if mod(k, display_freq) == 0
            mean_loss = mean(loss);
            mean_accuracy = mean(accuracy);
            now = clock;
            fprintf('[%02d:%02d:%05.2f] epoch %d, training loss %.5f, accuracy %.4f\n',...
                now(4), now(5), now(6), k, mean_loss, mean_accuracy);
            plotloss=[plotloss,mean_loss];
            plotaccuracy=[plotaccuracy,mean_accuracy];
            
        end
        
        a=[1:20];
        if mod(k, test_freq) == 0
            %plot(a,plotloss,'m');
            %text(5,plotloss(5),strcat('\leftarrow',num2str(net.layer_list{1}.learning_rate),'-',num2str(net.layer_list{3}.learning_rate),'-',num2str(net.layer_list{5}.learning_rate)));
            %hold on;
            
            %plot(a,plotaccuracy,'b');
            %legend('loss','accuracy');
            %text(15,plotloss(15),strcat('\leftarrow',num2str(net.layer_list{1}.learning_rate),'-',num2str(net.layer_list{3}.learning_rate),'-',num2str(net.layer_list{5}.learning_rate)));
            %title(strcat('1-',num2str(net.layer_list{1}.learning_rate),'-2-',num2str(net.layer_list{3}.learning_rate)));           
            
            test_iters = ceil(num_test_input / batchsize);
            test_loss = [];
            test_accuracy = [];
            for j = 1:test_iters
                [lss, ~, acc] = Predict(net, test_data(:,(j-1)*batchsize+1:min(j*batchsize, num_test_input)),...
                    test_label(:,(j-1)*batchsize+1:min(j*batchsize, num_test_input)));
                test_loss = [test_loss, lss];
                test_accuracy = [test_accuracy, acc];
            end
            test_mean_loss = mean(test_loss);
            test_mean_accuracy = mean(test_accuracy);
            now = clock;
            fprintf('[%02d:%02d:%05.2f]   test results: loss %.5f, accuracy %.4f\n',...
                now(4), now(5), now(6), test_mean_loss, test_mean_accuracy);
        end

        if mod(k, snapshot_freq) == 0
            save(['net_iter_' num2str(k) '.mat'], 'net');
            now = clock;
            fprintf('[%02d:%02d:%05.2f] Snapshotting to net_iter_%d.mat\n',...
                now(4), now(5), now(6), k);
        end
    end
end
      