function [saturation_data] = CO2_Simul(perm)

run("startup.m")

%% Load modules
mrstModule add co2lab ad-core ad-props ad-blackoil mrst-gui;

%% Grid and rock
cartDim = [96, 96, 1];
L       = [480, 480, 1];
G = cartGrid(cartDim, L);
G = computeGeometry(G);

log_data = perm;
log_data = transpose(log_data);
log_data = flip(log_data,2);
log_data_perm = log_data*2-1;
data_perm = power(10,log_data_perm);
data_poro = power(10,log_data);

data_perm = reshape(data_perm,cartDim(1,1)*cartDim(1,2),1);
data_poro = reshape(data_poro,cartDim(1,1)*cartDim(1,2),1);

rock.perm = darcy*data_perm*0.1;
rock.poro = 0.03*data_poro;


%% Initial state
gravity on;
rhow = 1000; % density of brine 
initState.pressure = repmat(10* mega * Pascal, G.cells.num, 1);
initState.s = repmat([1, 0], G.cells.num, 1); % initial saturations
initState.sGmax = initState.s(:,2); % initial max. gas saturation (hysteresis)

%% Fluid model
co2     = CO2props(); % load sampled tables of co2 fluid properties
p_ref   = 15 * mega * Pascal; % choose reference pressure
t_ref   = 70 + 273.15; % choose reference temperature, in Kelvin
rhoc    = co2.rho(p_ref, t_ref); % co2 density at ref. press/temp
cf_co2  = co2.rhoDP(p_ref, t_ref) / rhoc; % co2 compressibility
cf_wat  = 0; % brine compressibility (zero)
cf_rock = 4.35e-5 / barsa; % rock compressibility
muw     = 8e-4 * Pascal * second; % brine viscosity
muco2   = co2.mu(p_ref, t_ref) * Pascal * second; % co2 viscosity

mrstModule add ad-props; % The module where initSimpleADIFluid is found

% Use function 'initSimpleADIFluid' to make a simple fluid object
fluid = initSimpleADIFluid('phases', 'WG'           , ...
                           'mu'  , [muw, muco2]     , ...
                           'rho' , [rhow, rhoc]     , ...
                           'pRef', p_ref            , ...
                           'c'   , [cf_wat, cf_co2] , ...
                           'cR'  , cf_rock          , ...
                           'n'   , [2 2]);

% Change relperm curves
srw = 0.27;
src = 0.20;
fluid.krW = @(s) fluid.krW(max((s-srw)./(1-srw), 0));
fluid.krG = @(s) fluid.krG(max((s-src)./(1-src), 0));

% Add capillary pressure curve
pe = 5 * kilo * Pascal;
pcWG = @(sw) pe * sw.^(-1/2);
fluid.pcWG = @(sg) pcWG(max((1-sg-srw)./(1-srw), 1e-5)); %@@


%% Wells
nx = G.cartDims(1);
ny = G.cartDims(2);

% Injection rate
inj_rate = 2e-5;

% Start with empty set of wells
W = [];

% Add a well to the set
W = addWell(W, G, rock, 48*ny+48, ...
            'type', 'rate', ...  % inject at constant rate
            'val', inj_rate, ... % volumetric injection rate
            'comp_i', [0 1]);    % inject CO2, not water


%% Boundary conditions
bc = [];
bc = pside(bc, G, 'Left', 10* mega * Pascal, 'sat', [1, 0]);
bc = pside(bc, G, 'Right', 10* mega * Pascal, 'sat', [1, 0]);


%% Schedule
schedule.control = struct('W', W, 'bc', bc);

timesteps = rampupTimesteps(3000*day,60*day,2);
schedule.step.val = timesteps;
schedule.step.control = ones(numel(schedule.step.val), 1);   
                         

%% Model
model = TwoPhaseWaterGasModel(G, rock, fluid, 0, 0);

%% Simulate
[wellSol, states] = simulateScheduleAD(initState, model, schedule);


%% Save saturation data
h_n = 16;      % number of wells
h_spacing = 24;         % spacing between two wells
h_Start = 12;
h_End = 84;  

obs_num = length(schedule.step.val);
saturation_data=[];
first_step = 1;
for ns = first_step:obs_num
    for i = h_Start:h_spacing:h_End
        for j = h_Start:h_spacing:h_End
            if i == 48 && j == 48
                continue
            else
                s_co2 = states{ns}.s(:,2);
                saturation_data = [saturation_data;s_co2(i*ny+j)];     % extract saturation data
            end
        end
    end
end

end