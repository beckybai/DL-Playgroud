%%%
%get q
T = size(cap);
T = T(1);       % T = 24 
CUS = size(R);
CUS = CUS(1);   % Cus = 100
y = zeros(CUS,T);
ys = [];

init_flag = true;
% lambda_new = lambda0;
lambda_new = rand(100,24);

while 1
    % calculate the optimal y
    lambda = lambda_new;
    for t=1:T
        y(:,t) = calc_q(R(:,t),cap(t),lambda(:,t));
    end
    ys = [ys y];
    fprintf("q1: %f\t",sum(sum(lambda.*y)));
    q1 = sum(sum(lambda.*y));
    
    [z,lambda_new] = min_q(ys,v,R);
    q2 = z;
    fprintf("q2: %f\t",q2);
    
    fprintf("q1-q2: %f\n",q1-q2);
%     fprintf("-----\n");

    if(init_flag)
        init_flag = false;
    else
        if(abs(q1-q2) < 1e-8)
            break
        end
    % minimize q 
    end
end

%%%