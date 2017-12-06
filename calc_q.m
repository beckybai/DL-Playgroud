function x = calc_q(r_t,cap_t,lambda_t)
%     c = sum(lambda'*;
    n = 100; % unknown variables
    options = optimoptions('intlinprog');
    options = optimoptions(options,'Display','off');
    lb = zeros(n,1);
    ub = ones(n,1);
    x = intlinprog(-lambda_t,1:n,r_t',cap_t,[],[],lb,ub,options);
end