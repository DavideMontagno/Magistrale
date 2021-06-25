%% GENERATE DATASET
digits = load('lab2_2_data.mat');

% input patterns (memories)
U(1,:) = digits.p0;
U(2,:) = digits.p1;
U(3,:) = digits.p2;

%distorted patterns
Ud(1,:) = distort_image(U(1,:), 0.05);
Ud(2,:) = distort_image(U(1,:), 0.1);
Ud(3,:) = distort_image(U(1,:), 0.25);

Ud(4,:) = distort_image(U(2,:), 0.05);
Ud(5,:) = distort_image(U(2,:), 0.1);
Ud(6,:) = distort_image(U(2,:), 0.25);

Ud(7,:) = distort_image(U(3,:), 0.05);
Ud(8,:) = distort_image(U(3,:), 0.1);
Ud(9,:) = distort_image(U(3,:), 0.25);


%% INIT NETWORK
N = size(U, 2); % # of neurons
c = 1 / N;
M = c * (U' * U) ; % weight matrix

I = ones(N,1)*0.5; % bias

% remove self-recurrent connections
for i = 1:size(M, 1)
    M(i, i) = 0;
end

%% RETRIEVE DISTORTED PATTERNS
% for each test pattern

for i=1:size(Ud,1)
    fprintf('- Retrieving pattern %d...\n', i);
    % extract pattern i
    k = 1;
    eps = 1;
    t = 1;
    u = Ud(i,:);
    
    % init state, activations and energy
    xs = (M*u')';
    vs = u;
    
    % compute initial energy for u  
    energy = (-1/2) * u * (M * u') - u * I; 
    es = [energy];
    
    % compute initial overlaps with 3 memories 
    overlap(1) = c * (U(1,:) * u');
    overlap(2) = c * (U(2,:) * u');
    overlap(3) = c * (U(3,:) * u');
    
    os = [overlap];
   
    % retrive pattern i
    energy_old = energy;
    while true
        energy_old = energy;
        % for each neuron (asynchronous update)
        for j = randperm(N)
            t = t + 1;
            
            % init current state (to previous one)
            x = xs(t-1, :);
            v = vs(t-1, :);
            
            % update neuron j (state and activation)
            x(j) = M(j,:) * vs(t-1, :)' + I(j);
            if x(j) <= 0 
                v(j) = -1;
            else
                v(j) = 1;
            end
            
            %update overlaps with 3 memories
            overlap(1) = c * (U(1,:) * v');
            overlap(2) = c * (U(2,:) * v');
            overlap(3) = c * (U(3,:) * v');
            
            % update energy of network
            energy = (-1/2) * v * (M * v') - v * I;
            
            %store new state, activations, overlaps and energy
            xs = [xs; x];
            vs = [vs; v];
            os = [os; overlap];
            es(end+1) = energy;
        end
        
        fprintf('- epoch: %d, network energy: %f, energy gain: %f\n', k, energy, abs(energy - energy_old));
        k = k +1;
        if abs(energy - energy_old) < eps
            break
        end
    end
    
    ps = [0.05 0.1 0.25];
    p = ps(mod(i-1, 3) + 1);
    % compute which memory the pattern refers to
    % distorted 0s.
    if i <= 3
        j = 1;
    elseif i >= 7 % distorted 2s
        j = 3;
    else % distorted 1s
        j = 2;
    end
    
      %% PLOT
    %plot overlap with the 3 memories over time
    figure
    plot(1:t, os(:,1), 1:t, os(:,2), 1:t, os(:,3));
    legend('overlap with "0"','overlap with "1"','overlap with "2"');
    title(sprintf('pattern %d - real: "%d", noise:%0.2f%)', i, j-1, p));
     savefig(['./images_mandatory/overlaps' num2str(j-1) '-' num2str(p) '.fig']);
   
    % plot reconstructed image
    original  = U(j, :);
    distorted = Ud(i, :);
    retrievied = vs(end, :);
    final_overlap = c*(original*retrievied');
    count = 0;
    for k=1:size(retrievied,2)
        if(retrievied(k)==original(k))
            count=count+0;
        else
            count=count+1;
        end
    end
    if(final_overlap(end)>0)
        fprintf('pattern %d (noise %.2f) - overlap with "%d". Overlap: %f - Discrepancy: %f', mod(i-1, 3) + 1, p, j-1, final_overlap(end), count);
    else
        fprintf('pattern %d (noise %.2f) - overlap with "%d". Overlap: %f - Discrepancy: %f', mod(i-1, 3) + 1, p, j-1, final_overlap(end)*(-1), count);

    end
        figure
    subplot(1, 3, 1)
    imagesc(reshape(original, 32, 32));
    title('original');
    subplot(1, 3, 2)
    imagesc(reshape(distorted, 32, 32));
    title('distorted');
    subplot(1, 3, 3)
    imagesc(reshape(retrievied, 32, 32));
    title('retrievied');
    if(final_overlap(end)>0)
        sgt = sgtitle(sprintf('pattern %d (noise %.2f) - overlap with "%d". Overlap: %0.2f - Discrepancy: %f', mod(i-1, 3) + 1, p, j-1, final_overlap(end), count));
    else
      sgt = sgtitle(sprintf('pattern %d (noise %.2f) - overlap with "%d". Overlap: %0.2f - Discrepancy: %f', mod(i-1, 3) + 1, p, j-1, final_overlap(end)*(-1), count));   
    end
    sgt.FontSize = 7;
    savefig(['./images_mandatory/images_' num2str(j-1) '-' num2str(p) '.fig']);
    % plot energy function over time
    figure
    plot(1:t, es);
    title(sprintf('pattern %d: energy function over time.',  mod(i-1, size(Ud,1)) + 1));
    ylabel('energy');
    xlabel('epoch');
    savefig(['./images_mandatory/energyfunction_' num2str(mod(i-1, size(Ud,1)) + 1) '.fig']);
end