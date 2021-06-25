load laser_dataset;
whole_dataset = cell2mat(laserTargets);
input_dataset = whole_dataset(:,1:end-1);
target_dataset = whole_dataset(:,2:end);
parallel.gpu.enableCUDAForwardCompatibility(1)
% TR set
X_tr = input_dataset(:, 1:1000,:);
y_tr = target_dataset(:,1:1000,:);
X_tr = gpuArray(X_tr);
y_tr = gpuArray(y_tr);
% VAL set
X_val = input_dataset(:,5001:5500,:);
y_val = target_dataset(:,5001:5500,:);
X_val = gpuArray(X_val);
y_val = gpuArray(y_val);
% TS set
X_ts = input_dataset(:,3001:3500,:);
y_ts = target_dataset(:,3001:3500,:);
X_ts = gpuArray(X_ts);
y_ts = gpuArray(y_ts);
% final params
best_Ne = 0;
best_Ni = 0;
best_error_val = Inf;
best_error_tr = Inf;
last_error_ts= 0;
% param grid
Nes = [500 550 600 650 700];
Nis = [300 350 400 450];
str = strcat(pwd,"/figures/");
% MODEL SELECTION
fprintf('*** MODEL SELECTION ***\n')
for Ne = Nes
    for Ni = Nis
        fprintf('- Params: Ne: %d, Ni:%d\n', Ne, Ni);
        % TRAINING
        states_tr = liquid_state_machine(Ne, Ni, X_tr);
        Wout = y_tr * pinv(states_tr);
        y_tr_pred = Wout * states_tr;
        error_tr = mean(abs(y_tr_pred - y_tr));

        % VALIDATION
        states_val = liquid_state_machine(Ne, Ni, X_val);
        Wout = y_val * pinv(states_val);
        y_val_pred = Wout * states_val;
        error_val = mean(abs(y_val_pred - y_val));
        
         % TEST
        states_ts = liquid_state_machine(Ne, Ni, X_ts);
        Wout = y_ts * pinv(states_ts);
        y_ts_pred = Wout * states_ts;
        error_ts = mean(abs(y_ts_pred - y_ts));
        
        f=figure('visible','on');
        subplot(2,1,1);
        plot(1:size(y_tr_pred, 2), y_tr_pred,1:size(y_tr, 2), y_tr)
        title("Error Training: " +  num2str( error_tr) +" Ne: "+ num2str( Ne)+" Ni: "+ num2str( Ni));
        xlabel('time');
        ylabel('value');
        
        
        subplot(2,1,2);
        plot(1:size(y_ts_pred, 2), y_ts_pred,1:size(y_tr, 2), y_tr)
        title("Error Test: " +  num2str( error_tr) +" Ne: "+ num2str( Ne)+" Ni: "+ num2str( Ni));
        xlabel('time');
        ylabel('value');
        
        path = strcat(str,"Ne_"+ num2str( Ne)+"_Ni_"+ num2str( Ni)+"___"+num2str(error_tr)+".png");
        %exportgraphics(f,path);
        close(f);
        % SELECTION
        %fprintf('- TR error: %f, VAL error: %f\n', error_tr, error_val);
        if error_val < best_error_val
            best_error_val = error_val;
            best_error_tr = error_tr;
            best_Ne = Ne;
            best_Ni = Ni;
            states_tr = liquid_state_machine(Ne, Ni, X_ts);
            Wout = y_ts * pinv(states_tr);
            y_tr_pred = Wout * states_tr;
            last_error_ts = mean(abs(y_tr_pred - y_ts));
            fprintf('- Best TR error: %f, VAL error: %f, TS error: %f\n', error_tr, error_val, last_error_ts);
            
        end
    end
end
fprintf('******\n')
fprintf('- Best Params: Ne: %d, Ni:%d\n', best_Ne, best_Ni);
fprintf('- Best TR error: %f, VAL error: %f\n', best_error_tr, best_error_val);

fprintf('- Error on test: %f\n',last_error_ts);
% SAVE 
% retrain on full training set

f=figure('visible','on');
subplot(2,1,1);
states_tot = liquid_state_machine(best_Ne, best_Ni, X_tr);
Wout = y_tr * pinv(states_tot);
y_tot_pred = Wout * states_tot;
plot(1:size(y_tot_pred, 2), y_tot_pred,1:size(y_tr, 2), y_tr)
title('BEST: output vs target (Training Set)');
xlabel('time');
ylabel('value');
subplot(2,1,2);
states_tot = liquid_state_machine(best_Ne, best_Ni, X_ts);
Wout = y_ts * pinv(states_tot);
y_tot_pred = Wout * states_tot;
plot(1:size(y_tot_pred, 2), y_tot_pred,1:size(y_ts, 2), y_ts)
title('BEST: output vs target (Test Set)');
xlabel('time');
ylabel('value');
path = strcat(str,"BEST_Ne_"+ num2str( best_Ne)+"_Ni_"+ num2str( best_Ni)+"___"+num2str(error_tr)+".png");
exportgraphics(f,path);
           
fprintf("done!\n");
  
  save('summary.mat')
