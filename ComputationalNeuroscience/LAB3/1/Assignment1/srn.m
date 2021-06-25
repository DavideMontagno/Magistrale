parpool(8)
dataset = load('../useful_files/NARMA10timeseries.mat');

X = dataset.NARMA10timeseries.input;
y = dataset.NARMA10timeseries.target;

X_tr = X(1:4000);
y_tr = y(1:4000);

X_val = X(4001:5000);
y_val = y(4001:5000);

X_ts = X(5001:end);
y_ts = y(5001:end);
tr_fun='traingdm'
%% PARAM GRID
  nhs = [50 100 ];                 % hidden units
  etas = [0.1 0.01 0.001 0.0001];     % learning rate
  alphas = [0.1 0.5 0.9];         % momentum
  lambdas = [0.1 1e-3 1e-4];        % regularization
  max_epochs = 1000;


trfun = 'traingdm';

%% GRID SEARCH
nh_best = 0;
eta_best = 0;
alpha_best = 0;
lambda_best = 0;

error_tr_best = Inf;
error_val_best = Inf;

fprintf('- begin grid search\n');

for nh = nhs
for eta = etas
for alpha = alphas
for lambda = lambdas
    fprintf('\n-- params: nh: %d,\teta: %f,\talpha: %f,\tlambda: %f\n',...
        nh, eta, alpha, lambda);

    % setting network parameters
    srn_net = layrecnet(1, nh, 'traingdm');
    srn_net.trainParam.epochs = max_epochs;
    srn_net.trainParam.lr = eta;
    srn_net.trainParam.mc = alpha;
    srn_net.performParam.regularization = lambda;
    srn_net.divideFcn = 'dividetrain';
    
    % prepare timeseries for TR and VAL
    [delayedInput_tr, initialInput_tr, initialStates_tr, delayedTarget_tr] = preparets(srn_net, X_tr, y_tr);
    
    [delayedInput_val, initialInput_val, initialStates_val, delayedTarget_val] = preparets(srn_net, X_val, y_val);

    % train on TR
    [srn_net, tr] ...
        = train(srn_net, delayedInput_tr, delayedTarget_tr, initialInput_tr, 'UseParallel', 'yes');
    
    % computing immse on TR and VAL
    y_tr_pred = srn_net(delayedInput_tr, initialInput_tr);
    error_tr = immse(cell2mat(delayedTarget_tr), cell2mat(y_tr_pred));
    
    y_val_pred = srn_net(delayedInput_val, initialInput_val);
    error_val = immse(cell2mat(delayedTarget_val), cell2mat(y_val_pred));
    
    fprintf('-- TR error: %f,\t - VAL error: %f\n', error_tr, error_val);
    
    % check to find a new best
    if error_val < error_val_best
        fprintf('-- FOUND NEW BEST!\n');
        error_val_best = error_val;
        error_tr_best = error_tr;
        nh_best = nh;
        eta_best = eta;
        alpha_best = alpha;
        lambda_best = lambda;
    end
end
end
end
end

fprintf('- end grid search\n')
fprintf('- best params: nh: %d,\teta: %f,\talpha: %f,\tlambda: %f\n',...
        nh_best, eta_best, alpha_best, lambda_best);
fprintf('- best TR error: %f,\t - best VAL error: %f\n', error_tr_best, error_val_best);

%% TRAIN WITH FULL DATASET (TR+VAL)
% building best model
fprintf('- retraining model with full dataset\n');
srn_net = layrecnet(1, nh_best, 'traingdm');
srn_net.divideFcn = 'dividetrain';
srn_net.trainParam.lr = eta_best;
srn_net.trainParam.mc = alpha_best;
srn_net.trainParam.epochs = max_epochs;
srn_net.performParam.regularization = lambda_best;

[delayedInput_tr, initialInput_tr, initialStates_tr, delayedTarget_tr] = preparets(srn_net, [X_tr X_val], [y_tr y_val]);

[delayedInput_ts, initialInput_ts, initialStates_ts, delayedTarget_ts] =  preparets(srn_net, X_ts, y_ts);

% train on TR+VAL
[srn_net, tr_record] = train(srn_net, delayedInput_tr, delayedTarget_tr, initialInput_tr);


y_tr_pred = srn_net(delayedInput_tr, initialInput_tr);
error_tr_final_srn = immse(cell2mat(delayedTarget_tr), cell2mat(y_tr_pred));

y_ts_pred = srn_net(delayedInput_ts, initialInput_ts);
error_ts_final_srn = immse(cell2mat(delayedTarget_ts), cell2mat(y_ts_pred));

fprintf('- final TR error: %f,\t - final TS error: %f\n', error_tr_final_srn, error_ts_final_srn);

% saving results
save('summary_srn.mat')
%% PLOT
% learning curve
plot(tr_record.perf);
title('learning curve');
xlabel('epochs');
ylabel('error');
savefig('srn_lcurve_srn');

% target vs output
figure
subplot(2, 1, 1);
hold on
plot(1:size(y_tr_pred, 2), cell2mat(y_tr_pred));
plot(1:size(y_tr_pred, 2), cell2mat(delayedTarget_tr));
title('target vs output (TR+VAL)');
legend('output', 'target');

subplot(2, 1, 2);
hold on
plot(1:size(y_ts_pred, 2), cell2mat(y_ts_pred))
plot(1:size(y_ts_pred, 2), cell2mat(delayedTarget_ts))
title('target vs output (TS)');
legend('output', 'target');

savefig('srn_output_target');

