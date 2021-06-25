dataset = table2array(readtable("./useful_files/lab2_1_data.csv"));
stop_condition_weights= 1e-6;
current_time=0;
theta_v=-1;
max_iteration=6000;
current_iteration=0;
learning_rate = 0.0001;
[rows,columns]= size(dataset);
w = -1 + 2.*rand(2,1); % random bw -1 and 1
array_weights = [];
normws = [];
convergence = [];
temp_dataset = dataset;
    while(true) %epochs
    dataset = dataset(:,randperm(columns)); %shuffling dataset
    current_iteration=current_iteration+1; %current_epoch
    w_temp=w;
    for i=1:columns %get new w vector
        v=w'*dataset(:,i);
        delta =v*dataset(:,i)*(v-theta_v);
        w = w+learning_rate*delta;
        theta_v=learning_rate*(v^2-theta_v)+theta_v;
    end
    array_weights = [array_weights;w];
    normws=[normws;norm(w)];
    
    %STOP CONDITION
    if(norm(w-w_temp) < stop_condition_weights || current_iteration > max_iteration)
        break;
    end
    
    
    end
w1 = array_weights(1:2:end);
w2=array_weights(2:2:end);

%PLOT FIGURE


figure
hold on;
Q = dataset * dataset';
[eigvecs, eigvals] = eig(Q);
eigvals = diag(eigvals);
[max_v, max_i] = max(eigvals);
plotv(eigvecs(:,max_i));
plot(dataset(1,:),dataset(2,:),'.')
plotv(w/norm(w))
legend('max eigenvector','point in dataset','weight')
savefig('./bonus1/principal.fig');


figure

plot(w1)
xlabel('time');
title('w(1) over time');
savefig('./bonus1/w1.fig');
figure
plot(w2)
xlabel('time');
title('w(2) component over time');
savefig('./bonus1/w2.fig');

% norm(w) over time
figure
plot(1:current_iteration, normws);
xlabel('time');
title('norm(w) over time');
savefig('./bonus1/norm(w).fig');
