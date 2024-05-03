clear
close all
clc

%% basic configurations

M = 256;

c = 3e8; % unit: m/s
f_c = 100e9; % unit: Hz

wavelength = c / f_c; % unit: m

bandwidth_ratio = 0.1;

B = f_c * bandwidth_ratio;

D = (M-1)*(wavelength/2); % unit: m

%% near-field condition

Fraunhofer_distance = (2 * D^2) / wavelength;
fprintf('The Fraunhofer distance is %.3f meters.\n', Fraunhofer_distance)
%% boundary between the near field and the far field.

%% this is about the new architecture!!!!!!

num_of_TTD = 16;
num_of_PS_per_TTD = M / num_of_TTD;

num_of_bits = 3;
phase_collection = linspace(-pi, pi, 2^num_of_bits+1);
%%linspace to generate a collection of evenly spaced phase values ranging from -pi to pi
phase_collection = phase_collection(1:end-1);

%%

num_of_sub_carrier = 4096;

f_list = linspace(f_c-B/2, f_c+B/2, num_of_sub_carrier); %f ∈ fc − B/2C ,fc +B/2C
% f_list = f_list * (10^9); converted from Hz to GHz 

%% ground true geometry

% -------- antenna elements are on x axis and centered on the origin point
% ant_positions = linspace(0, D, M) - D / 2;
% equally spaced antenna positions along the x-axis, ranging from 0 to D. These positions are centered around the origin point (0)

% ant_pos_x = (-D/2) + D * rand(1, M); %%effectively positioning the antenna elements randomly along the x-axis.
%random positions for the antenna elements along the x-axis. It first
%generates M random values within the range (-D/2, D/2), sorts them in ascending order.
% ant_pos_x = sort(ant_pos_x);  
% ant_pos_x = ant_pos_x.';
% ant_pos_y = zeros(M, 1);
% ant_positions = cat(2, ant_pos_x, ant_pos_y); % unit: m

tmp = load('C:/Users/AAYUSHI SINGH/Downloads/NFWB_BF-main/NFWB_BF-main/td_searching/ULA_pos.mat');
ant_positions = tmp.ant_positions; % unit: m
% ------------------------------------------------------------------------

% -------- user position: (x, y)
focal_point = [2, 2]; % unit: m

%% ground true distance information

center_dist = calc_dist(focal_point, ant_positions); %distances between the focal point (the location of the user) and each antenna element in the array.

%% near-field beamforming: independent of everything!!!!

wideband_channel = zeros(M, length(f_list)); % (ant, sub)

for ii = 1:length(f_list)
    
    %% wavelength per subcarrier
    wavelength_f = c / f_list(ii);
    
    %% wideband channel generation: sphere wavefront
    wideband_channel(:, ii) = (1/sqrt(M)) * exp(-1j*(2*pi/wavelength_f)*center_dist);
    
end


%% EGC BF vector design

num_of_user = size(focal_point, 1);% Each row in focal_point represents a user
w_EGC = zeros(M, num_of_user);%each column corresponds to the beamforming vector for a specific user

for cc = 1:num_of_user
    
    f_point = focal_point(cc, :);
    center_dist = calc_dist(f_point, ant_positions);
    
    center_channl = (1/sqrt(M)) * exp(-1j*(2*pi/wavelength)*center_dist);
    disp(center_channl);
    w_EGC(:, cc) = (1/sqrt(M)) * exp(1j * angle(center_channl));
    disp(w_EGC);
    % w = (1/sqrt(M)) * exp(1j * 2 * pi * rand(M, 1));
    
end

%%

load('ULA_PS_only1.mat')
% 
% w_EGC = (1/sqrt(M)) * exp(1j * beams.');

learned_phases = beams;

% learned_phases = [-0.78540,1.57080,-0.78540,-0.78540,1.57080,2.35619,0.78540,0.78540,2.35619,-2.35619,-0.78540,0.00000,-2.35619,-1.57080,0.00000,0.00000,-2.35619,-1.57080,-1.57080,-1.57080,-0.78540,0.00000,2.35619,2.35619,2.35619,-0.78540,2.35619,-2.35619,-1.57080,-0.78540,0.78540,2.35619,3.14159,-2.35619,-2.35619,0.00000,-2.35619,-1.57080,-1.57080,-0.78540,1.57080,-1.57080,0.78540,0.00000,3.14159,-2.35619,1.57080,1.57080,-2.35619,3.14159,-0.78540,1.57080,1.57080,1.57080,0.00000,1.57080,-0.78540,-0.78540,1.57080,3.14159,0.00000,2.35619,2.35619,-1.57080,0.00000,0.78540,2.35619,-2.35619,2.35619,-2.35619,-1.57080,1.57080,3.14159,-1.57080,1.57080,1.57080,-0.78540,0.78540,3.14159,3.14159,2.35619,2.35619,-2.35619,0.00000,1.57080,1.57080,-0.78540,-1.57080,-1.57080,0.00000,-2.35619,-1.57080,0.00000,-2.35619,0.00000,0.78540,1.57080,1.57080,-2.35619,-1.57080,-0.78540,2.35619,-2.35619,-0.78540,0.00000,0.78540,-1.57080,-0.78540,0.00000,0.78540,-2.35619,-0.78540,2.35619,-2.35619,-2.35619,-1.57080,0.78540,0.78540,-2.35619,0.00000,0.78540,2.35619,3.14159,-2.35619,0.00000,-1.57080,0.78540,0.78540,-0.78540,-2.35619,1.57080,3.14159,-0.78540,0.78540,1.57080,0.00000,0.78540,1.57080,2.35619,-0.78540,1.57080,-2.35619,-1.57080,-0.78540,2.35619,2.35619,3.14159,-1.57080,0.00000,2.35619,3.14159,-0.78540,0.78540,1.57080,3.14159,-1.57080,-0.78540,-0.78540,-2.35619,-0.78540,-0.78540,0.78540,1.57080,1.57080,-0.78540,0.00000,0.78540,0.78540,3.14159,-2.35619,-1.57080,1.57080,2.35619,3.14159,-0.78540,0.00000,1.57080,-2.35619,-2.35619,-0.78540,0.00000,0.00000,0.00000,0.00000,-2.35619,-2.35619,0.00000,2.35619,-0.78540,0.00000,-2.35619,-0.78540,0.78540,1.57080,2.35619,0.00000,0.78540,0.78540,2.35619,2.35619,2.35619,3.14159,-2.35619,-0.78540,-0.78540,1.57080,0.00000,0.00000,0.78540,-2.35619,-1.57080,-1.57080,0.00000,2.35619,-1.57080,0.00000,-2.35619,-2.35619,-2.35619,0.78540,2.35619,-2.35619,-2.35619,-0.78540,-0.78540,0.78540,0.78540,0.78540,-2.35619,0.78540,-1.57080,-0.78540,-0.78540,1.57080,-2.35619,-0.78540,1.57080,3.14159,-0.78540,0.00000,2.35619,-0.78540,3.14159,-1.57080,1.57080,-0.78540,0.00000,0.00000,0.78540,2.35619,2.35619,-0.78540,0.00000,0.78540,2.35619,-1.57080];
w_EGC = (1/sqrt(M)) * exp(1j * learned_phases.');

plot_pattern(w_EGC, 9)

%% true-time-delay unit design

% delta_tau = center_dist / c; % s
% % delta_tau = linspace(delta_tau(1+3), delta_tau(end-3), num_of_TTD);
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % delta_tau = delta_tau(1:num_of_PS_per_TTD:end)-1.18973567e-7;
% % delta_tau = delta_tau(2:num_of_PS_per_TTD:end);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% sample_idx = zeros(num_of_TTD, 1);
% for ii = 1:num_of_TTD
%     
%     sample_idx(ii) = randi(num_of_PS_per_TTD) + (ii-1)*num_of_PS_per_TTD;
%     
% end
% delta_tau = delta_tau(sample_idx)-1.18973567e-7;
% 
% % delta_tau = abs(randn(num_of_TTD, 1)) * 1e-9;
% % delta_tau = cat(1, rand*1e-9, rand*1e-9, zeros(num_of_TTD-2, 1));



%% distance estimation based TD design

% ant_pos_x_est = (-D/2) + D * rand(1, M);
% ant_pos_x_est = sort(ant_pos_x_est);
% ant_pos_x_est = ant_pos_x_est.';
% ant_pos_y_est = zeros(M, 1);
% ant_positions_est = cat(2, ant_pos_x_est, ant_pos_y_est); % unit: m

% ant_positions_est = linspace(0, D, M) - D / 2;
% ant_positions_est = cat(2, ant_positions_est.', zeros(M, 1));
% focal_point_est = [2.2, 1.8];
% focal_point_est = [2, 2];

% dist_range = [2.6, 3.1];

% sample_func = inline('sqrt(a.^2+b.^2-2*a.*b.*cos(theta))', 'a', 'b', 'theta');

% center_dist_est = calc_dist(focal_point_est, ant_positions_est);



%% TD unit design algorithm goes here

% delta_tau = center_dist_est / c; % s

% delta_tau = linspace(0, D*(0.5), M)/c;

% This is to generate the delay search space
linear_space = search_space_gen(1); %search space for the algorithm to explore during optimization.

% for test
% linear_space = [2, -0.2632, -1];
% The above values produce the best performance and they are within
% search_space_gen(1).

for i = 1:size(linear_space, 1) + 1
    
    if i == size(linear_space, 1) + 1
        a_x = best_a_x; %ax : parameter controlling the shape of the estimated space. 
        a_y = best_a_y; %parameter influencing the shape of the estimated space. 
        % b = best_b;For each set of parameters (a_x, a_y, b), a time delay estimation is computed.
    else
        a_x = linear_space(i, 1);
        a_y = linear_space(i, 2);
        b = linear_space(i, 3);
    end
    % Two sets of time delay estimations
    %equation 18
    space_est_1 = (a_y/a_x)*linspace(0, a_x, floor(M*(a_x/2)));
    space_est_2 = (b-a_y)/(2-a_x)*(linspace(a_x, 2, M-floor(M*(a_x/2)))-a_x)+a_y;
    
    space_est = cat(2, space_est_1, space_est_2);
    
    % delta_tau = linspace(delta_tau(1+3), delta_tau(end-3), num_of_TTD);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % delta_tau = delta_tau(1:num_of_PS_per_TTD:end)-1.18973567e-7;
    % delta_tau = delta_tau(2:num_of_PS_per_TTD:end);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
%     figure(1)
%     
%     plot(linspace(0,2,256),center_dist-center_dist(1));hold on;plot(linspace(0,2,256), space_est)
%     legend('True', 'Approx')
%     drawnow;
%     pause(0.5);
%     cla;
    
    sample_idx = zeros(num_of_TTD, 1);%column vector of zeros with a length of num_of_TTD
    % sample_idx, which likely represents specific indices of interest in the time delay estimation array.
    for ii = 1:num_of_TTD
        
        %     sample_idx(ii) = randi(num_of_PS_per_TTD) + (ii-1)*num_of_PS_per_TTD;
        %     sample_idx(ii) = 2 + (ii-1)*num_of_PS_per_TTD;
        sample_idx(ii) = 8 + (ii-1)*num_of_PS_per_TTD; %graph minimum 8 and max 16
        
    end
    space_est_sampled = space_est(sample_idx);
    % delta_tau = cat(1, delta_tau(1:6), zeros(10,1));
    
    
    delta_tau = space_est_sampled / c;
    % fτ = fd/c .
    
    
    
    
    
    
    
    %% PS configuration
    
    ph_EGC_for_ttd = zeros(M, 1);
    
    for ww = 1:length(delta_tau)
        sub_arr_index = (ww-1)*num_of_PS_per_TTD+1:ww*num_of_PS_per_TTD;
        %calculates the indices corresponding to the phase shifters associated with the current time delay unit.

        ph_EGC_for_ttd(sub_arr_index, :) = angle(w_EGC(sub_arr_index, :)) + 2*pi*f_c*delta_tau(ww); %([θ ]m +2πfcτn)|
        % calculates the phase angle of the beamforming weights (w_EGC) associated with the current time delay unit.
        %2*pi*f_c*delta_tau(ww) calculates the additional phase shift
        %introduced by the time delay for the current time delay unit.
    end
    
    ph_EGC_for_ttd = circular_shift(ph_EGC_for_ttd, [-pi, pi]);
    
    dist_matrix = abs(ph_EGC_for_ttd - phase_collection); % [θ ]m =θ  - ([θ ]m +2πfcτn) %eqn 14
    [~, I] = min(dist_matrix, [], 2);% argmin|θ  - ([θ ]m +2πfcτn)|
    ph_EGC_for_ttd = phase_collection(I).';
    % ph_EGC_local = ph_EGC_local;
    
    w_PS = (1/sqrt(M)) * exp(1j * ph_EGC_for_ttd); %[wPS]m = 1/ √Mejθm
    
    %% hybrid BF vector design
    
    w_TTD_only = zeros(num_of_TTD, length(f_list));
    w_TTD_ph = zeros(num_of_TTD, length(f_list));
    for ii = 1:length(f_list)
        f = f_list(ii);
        
        for jj = 1:num_of_TTD
            
            %% THIS IS THE CORE. %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % cophasing : to adjust the phase of a signal
            %         cophasing = - 2 * pi * (f-f_c) * delta_tau(jj); %
            %         minus sign is very important!!!! minus sign likely determines the direction of phase shift
            %calculates the phase shift based on the difference between the signal frequency f and a centeral frequency f_c, multiplied by a time delay delta_tau(jj)
            cophasing = - 2 * pi * f * delta_tau(jj); % minus sign is very important!!!!
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            w_TTD_ph(jj, ii) = cophasing;
            
            %         sub_array_index = ((jj-1)*num_of_PS_per_TTD+1):(jj*num_of_PS_per_TTD);
            w_TTD_only(jj, ii) = exp(1j * ( cophasing ));
            %         w_TTD(sub_array_index, ii) = (1/sqrt(M)) * exp(1j * cophasing);
            
        end
        
    end
    
    w_TTD = w_PS .* kron(w_TTD_only, ones(num_of_PS_per_TTD, 1)); %wTD,k = e−j2πfkτ⊗1P
    
    
    %% channel at center frequency
    
    % channel_fc = (1/sqrt(M)) * exp(-1j*(2*pi/wavelength)*center_dist);
    
    % save('H_fc.mat', 'channel_fc')
    
    %% PS-only beamforming gain calculation
    
    bf_gain_vec_EGC_linear = abs(w_EGC'*wideband_channel).^2;
    bf_gain_vec_EGC_db = 10*log10(bf_gain_vec_EGC_linear);
    
    %% TTD-PS-hybrid beamforming gain calculation
    
    gain_list_TTD_linear = zeros(1, length(f_list));
    gain_list_TTD_db = zeros(1, length(f_list));
    
    % phase_pattern = zeros(num_of_TTD, length(f_list));
    
    for ii = 1:length(f_list)
        
        f = f_list(ii);
        gain_list_TTD_linear(ii) = abs(w_TTD(:, ii)'*wideband_channel(:, ii))^2;
        gain_list_TTD_db(ii) = 10*log10(abs(w_TTD(:, ii)'*wideband_channel(:, ii))^2);
        
    end
    
    %%
    
    if i == 1
        perf_best = mean(gain_list_TTD_db);
        best_a_x = a_x;
        best_a_y = a_y;
        best_b = b;
    else
        perf = mean(gain_list_TTD_db);
        if perf > perf_best
            best_a_x = a_x;
            best_a_y = a_y;
            best_b = b;
            perf_best = perf;
        end
    end
    
    
    %%
    
    sig_ps_layer = wideband_channel .* conj(repmat(w_PS, 1, num_of_sub_carrier));
    sig_ps_layer = reshape(sig_ps_layer, num_of_PS_per_TTD, []);
    sig_ps_layer = sum(sig_ps_layer, 1);
    sig_ps_layer = reshape(sig_ps_layer, num_of_TTD, []);
    
    sig_ps_layer_ph = angle(sig_ps_layer);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%% Interface is here %%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % sig_ph,ttd : phase of the signals after PS and TTD processing, respectively.
    sig_ttd_layer = sig_ps_layer .* conj(w_TTD_only);
    
    test = sig_ps_layer_ph - w_TTD_ph;
    test = circular_shift(test, [-pi, pi]);
    
    sig_ttd_layer_ph = angle(sig_ttd_layer);
    
    result = 10*log10(abs(sum(sig_ttd_layer, 1)).^2);
    
    % figure(1)
    %
    % plot(f_list/(1e9), result)
    
    %%
    
    if i == 1
        figure(2)

        p0 = plot(f_list/(1e9), bf_gain_vec_EGC_db, '-');
        p0.LineWidth = 1.0;
        hold on
        p = plot(f_list/(1e9), gain_list_TTD_db, '-');
        p.LineWidth = 1.0;
        
        ylim([-60, 0])

        xlabel('Frequency (GHz)')
        ylabel('Beamforming gain (dB)')
        legend({'EGC', 'Hybrid'})

        grid on
        box on
        
        drawnow;
        pause(0.001);
%         cla;
        delete(p);
        fprintf("%d / %d \n", i, size(linear_space, 1))
    elseif i == size(linear_space, 1) + 1
        p = plot(f_list/(1e9), gain_list_TTD_db, '-k');
        p.LineWidth = 1.0;
        ylim([-45, 0])
        legend({'Phase-shifter only architecture', ...
            ['Proposed method with unknown channel (N=',num2str(num_of_TTD),')']})
        fprintf("Search results: best ax: %.5f, best ay: %.5f, best b: %.5f. \n", best_a_x, best_a_y, best_b)
    else
        p = plot(f_list/(1e9), gain_list_TTD_db);
        legend({'EGC', 'Hybrid'})
        drawnow;
        pause(0.001);
%         cla;
        delete(p);
        fprintf("%d / %d \n", i, size(linear_space, 1))
    end
    
end

