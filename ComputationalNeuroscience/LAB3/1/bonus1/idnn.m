clc
clear
%% LOAD DATASET

load laser_dataset;
dataset = cell2mat(laserTargets);

%load MGtimeseries;
%dataset = cell2mat(MGtimeseries);

dataset = mapminmax(dataset); % scaling to [-1 1]

X = dataset(1:end-1);
y = dataset(2:end);

X_tr = X(1:4000);
y_tr = y(1:4000);

X_val = X(4001:5000);
y_val = y(4001:5000);

X_ts = X(5001:end);
y_ts = y(5001:end);

%% PARAM GRID
  nhs = [5 10 15 20];                 % hidden units
  wss = [10 ];                  % window size
  etas = [0.1 0.01];     % learning rate
  alphas = [0.5 0.9];         % momentum
  lambdas = [0.1 1e-3 1e-4];        % regularization
  max_epochs = 3000;


trfun = 'traingdm';

%% GRID SEARCH
nh_best = 0;
ws_best = 0;
eta_best = 0;
alpha_best = 0;
lambda_best = 0;

error_tr_best = Inf;
error_val_best = Inf;

fprintf('- begin grid search\n');
for nh = nhs
for ws = wss
for eta = etas
for alpha = alphas
for lambda = lambdas
    fprintf('\n-- params: nh: %d,\tws: %d,\teta: %f,\talpha: %f,\tlambda: %f\n',...
        nh, ws, eta, alpha, lambda);

    % setting network parameters
    idnn_net = timedelaynet(0:ws, nh, trfun);
    idnn_net.trainParam.epochs = max_epochs;
    idnn_net.trainParam.lr = eta;
    idnn_net.trainParam.mc = alpha;
    idnn_net.performParam.regularization = lambda;
    idnn_net.divideFcn = 'dividetrain';
    
    % prepare timeseries for TR and VAL
    [delayedInput_tr, initialInput_tr, initialStates_tr, delayedTarget_tr] ...
        = preparets(idnn_net, num2cell(X_tr), num2cell(y_tr));
    
    [delayedInput_val, initialInput_val, initialStates_val, delayedTarget_val] ...
        = preparets(idnn_net, num2cell(X_val), num2cell(y_val));

    % train on TR
    [idnn_net, tr] ...
        = train(idnn_net, delayedInput_tr, delayedTarget_tr, initialInput_tr, 'UseParallel', 'yes');
    
    % computing immse on TR and VAL
    y_tr_pred = idnn_net(delayedInput_tr, initialInput_tr);
    error_tr = immse(cell2mat(delayedTarget_tr), cell2mat(y_tr_pred));
    
    y_val_pred = idnn_net(delayedInput_val, initialInput_val);
    error_val = immse(cell2mat(delayedTarget_val), cell2mat(y_val_pred));
    
    fprintf('-- TR error: %f,\t - VAL error: %f\n', error_tr, error_val);
    
    % check to find a new best
    if error_val < error_val_best
        fprintf('-- FOUND NEW BEST!\n');
        error_val_best = error_val;
        error_tr_best = error_tr;
        nh_best = nh;
        ws_best = ws;
        eta_best = eta;
        alpha_best = alpha;
        lambda_best = lambda;
    end
end
end
end
end
end

fprintf('- end grid search\n')
fprintf('- best params: nh: %d,\tws: %d,\teta: %f,\talpha: %f,\tlambda: %f\n',...
        nh_best, ws_best, eta_best, alpha_best, lambda_best);
fprintf('- best TR error: %f,\t - best VAL error: %f\n', error_tr_best, error_val_best);

%% TRAIN WITH FULL DATASET (TR+VAL)
fprintf('- retraining model with full dataset\n');
idnn_net = timedelaynet(0:ws_best, nh_best, trfun);
idnn_net.divideFcn = 'dividetrain';
idnn_net.trainParam.lr = eta_best;
idnn_net.trainParam.mc = alpha_best;
idnn_net.trainParam.epochs = max_epochs;
idnn_net.performParam.regularization = lambda_best;

% prepare timeseries for TR+VAL and TS
[delayedInput_tr, initialInput_tr, initialStates_tr, delayedTarget_tr] = ...
    preparets(idnn_net, num2cell([X_tr X_val]), num2cell([y_tr y_val]));

[delayedInput_ts, initialInput_ts initialStates_ts, delayedTarget_ts] = ...
    preparets(idnn_net, num2cell(X_ts), num2cell(y_ts));

% train on TR+VAL
[idnn_net, idnn_tr_record] = ...
    train(idnn_net, delayedInput_tr, delayedTarget_tr, initialInput_tr);


y_tr_pred = idnn_net(delayedInput_tr, initialInput_tr);
idnn_error_tr_final = immse(cell2mat(delayedTarget_tr), cell2mat(y_tr_pred));

y_ts_pred = idnn_net(delayedInput_ts, initialInput_ts);
idnn_error_ts_final = immse(cell2mat(delayedTarget_ts), cell2mat(y_ts_pred));

fprintf('- final TR error: %f,\t - final TS error: %f\n', idnn_error_tr_final, idnn_error_ts_final);

% saving results
save('summary_idnn.mat')

%% PLOT
% learning curve
plotperf(idnn_tr_record);
title('learning curve');
xlabel('epochs');
ylabel('error');

savefig('idnn_lcurve');

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

savefig('idnn_output_target');