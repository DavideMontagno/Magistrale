%%%%%%%%%%%%%%% Echo State Network %%%%%%%%%%%%%%%
clear variables;

load('./useful_files/NARMA10timeseries.mat');  % import data
input = cell2mat(NARMA10timeseries.input);
target = cell2mat(NARMA10timeseries.target);

steps = 5000;
val_steps = 4000;

% design set
X_design = input(1:5000);
Y_design = target(1:5000);
% test set
X_test = input(5001:6001);
Y_test = target(5001:6001);
% training set
X_train = X_design(1:4000);
Y_train = Y_design(1:4000);
% validation set
X_val = X_design(4001:end);
Y_val = Y_design(4001:end);

% parameters for grid search
input_scalings = [0.5 1 2];
Nrs = [5 10 25 50 100 200];  % reservoir dimension (number of recurrent units)
rho_values = [0.1 0.5 0.9 1.2 2];  % spectral radius
lambdas = [0.0001 0.001 0.01 0.1];  % readout regularization for ridge regression
guesses = 10;  % network guesses for each reservoir hyper-parametrization


  
Ers_tr = [];
Ers_val = [];
for omega_in=input_scalings
   for Nr=Nrs
       for rho=rho_values
           for l=lambdas
            Nu = size(X_train,1);
            trainingSteps = size(X_train,2);
            validationSteps = size(X_val,2);
            E_trs = [];
            E_vals = [];

            fprintf('Input scaling: %.2f - Reservoir dimension: %d Spectral radius: %.2f - Lambda: %.4f\n', omega_in, Nr, rho, l);

            for n = 1:guesses        
                % initialize the input-to-reservoir matrix
                U = 2*rand(Nr,Nu+1)-1;
                U = omega_in * U;
                % initialize the inter-reservoir weight matrices
                W = 2*rand(Nr,Nr) - 1;
                W = rho * (W / max(abs(eig(W))));
                state = zeros(Nr,1);
                H = [];

                % run the reservoir on the input stream
                for t = 1:trainingSteps
                    state = tanh(U * [X_train(t);1] + W * state);
                    H(:,end+1) = state;
                end
                % discard the washout
                H = H(:,Nr+1:end);
                % add the bias
                H = [H;ones(1,size(H,2))];
                % update the target matrix dimension
                D = Y_train(:,Nr+1:end);
                % train the readout
                V = D*H'*inv(H*H'+ l * eye(Nr+1));

                % compute the output and error (loss) for the training samples
                Y_train_pred = V * H;
                err_tr = immse(D,Y_train_pred);
                E_trs(end+1) = err_tr;

                state = zeros(Nr,1);
                H_val = [];
                % run the reservoir on the validation stream
                for t = 1:validationSteps
                    state = tanh(U * [X_val(t);1] + W * state);
                    H_val(:,end+1) = state;
                end
                % add the bias
                H_val = [H_val;ones(1,size(H_val,2))];
                % compute the output and error (loss) for the validation samples
                Y_val_pred = V * H_val;
                err_val = immse(Y_val,Y_val_pred);
                E_vals(end+1) = err_val;

            end
            error_tr = mean(E_trs);
            Ers_tr(end+1) = error_tr;
            fprintf('Error on training set: %.5f\n', error_tr);
            error_val = mean(E_vals);
            Ers_val(end+1) = error_val;
            fprintf('Error on validation set: %.5f\n\n', error_val);
           end
       end
    end
end

%Model selection
[value, idx] = min(Ers_val);
[I, NR, R, L] = ndgrid(input_scalings,Nrs,rho_values,lambdas);
grid = [I(:) NR(:) R(:) L(:)];
omega_in = grid(idx,1);
Nr = grid(idx,2);
rho = grid(idx,3);
l = grid(idx,4);
best_validation = value;
best_training = Ers_tr(idx);
%end model selection

fprintf('\nBest hyper-params:\nInput scaling: %f - Reservoir dimension: %d - Spectral radius: %f - Lambda: %f\nError Training (MSE): %f\nError Validation (MSE): %f\n', omega_in, Nr, rho, l, best_training, best_validation);

% model assessment
Nu = size(X_design,1);
designSteps = size(X_design,2);
testSteps = size(X_test,2);
% initialize the input-to-reservoir matrix
U = 2*rand(Nr,Nu+1)-1;
U = omega_in * U;
% initialize the inter-reservoir weight matrices
W = 2*rand(Nr,Nr) - 1;
W = rho * (W / max(abs(eig(W))));
state = zeros(Nr,1);
H = [];
E_trs = [];
E_tests = [];
for n = 1:guesses        
        % initialize the input-to-reservoir matrix
        U = 2*rand(Nr,Nu+1)-1;
        U = omega_in * U;
        % initialize the inter-reservoir weight matrices
        W = 2*rand(Nr,Nr) - 1;
        W = rho * (W / max(abs(eig(W))));
        state = zeros(Nr,1);
        H = [];
        
        % run the reservoir on the input stream
        for t = 1:designSteps
            state = tanh(U * [X_design(t);1] + W * state);
            H(:,end+1) = state;
        end
        % discard the washout
        H = H(:,Nr+1:end);
        % add the bias
        H = [H;ones(1,size(H,2))];
        % update the target matrix dimension
        D = Y_design(:,Nr+1:end);
        % train the readout
        V = D*H'*inv(H*H'+ l * eye(Nr+1));
        
        % compute the output and error (loss) for the design samples
        Y_design_pred = V * H;
        err_tr = immse(D,Y_design_pred);
        E_trs(end+1) = err_tr;
        
        state = zeros(Nr,1);
        H_test = [];
        % run the reservoir on the test stream
        for t = 1:testSteps
            state = tanh(U * [X_test(t);1] + W * state);
            H_test(:,end+1) = state;
        end
        % add the bias
        H_test = [H_test;ones(1,size(H_test,2))];
        % compute the output and error (loss) for the test samples
        Y_test_pred = V * H_test;
        error_test = immse(Y_test,Y_test_pred);
        E_tests(end+1) = error_test;     
 end
error_design = mean(E_trs);
fprintf('Error on design set: %.5f\n', error_design);
error_test = mean(E_tests);
fprintf('Error on test set: %.5f\n\n', error_test);

figure
subplot(2, 1, 1);
hold on
plot(1:size(Y_design(:,Nr+1:end),2),Y_design(:,Nr+1:end)); 
plot(1:size(Y_design_pred,2),Y_design_pred); 
title('target vs output (TR+VAL)');
legend('output', 'target');

subplot(2, 1, 2);
hold on
plot(1:size(Y_test, 2), Y_test)
plot(1:size(Y_test_pred, 2), Y_test_pred)
title('target vs output (TS)');
legend('target', 'output');

savefig('./mandatory_output/final_plot');

save('./mandatory_output/summary.mat','omega_in','Nr','Nu','rho','l','U','V','W','best_training','best_validation','error_test','error_design')

