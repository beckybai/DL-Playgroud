function [z_value,lambda_new] = min_q(ys,v,R)
%     lambdas_size = size(lambdas);
%     lambdas_n = lambdas_size(1);
%     for i=1:lambdas_n
%     end
    lambda_size = size(ys);
    lambda_n = lambda_size(2)/24;
    
    R_zero_sign = (R==0);
    
    cvx_begin quiet
        variable lambda(100,24) nonnegative
        variable z
        minimize z
        for i = 1:lambda_n
            z >= sum(sum(lambda.*ys(:,(i-1)*24+1:(i*24))));
        end
%         lambda >= 0;
        sum(lambda') == v';
        sum(diag(lambda'* R_zero_sign)) == 0;
    cvx_end
    lambda_new = lambda;
    z_value = z;
end