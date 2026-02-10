
function pose_plane_landing(datapath,evalpath,targetname)

%% SETTINGS ---
fprintf('\n--- Plane Landing ---\n')

% camera parameters
f = 250;
w = 200; h = 200;

% pose space
bounds = [
  -50  50;
  -50 150;
   50 350;
    0  90; 
   -5   5; 
   -5   5; 
];
bounds(4:6,:) = bounds(4:6,:) .* pi ./ 180;
POSE = interval(bounds(:,1)-eps, bounds(:,2)+eps);
pose_printUncertainty(POSE)

% get target
fprintf('Loading target.. ')
% targetname = '30';
[V_TCF, B_TCF, V_prop] = pose_getTarget(targetname,true);
I_scale = interval([1;1]);
[lambdas, PZ_lambdas] = pose_getLambdasScaling(B_TCF, V_TCF, V_prop, I_scale);
disp('Done.')

%% OFFLINE PREPARATION ---
warning off
CANDIDATES = pose_gatherCandidates(POSE, B_TCF, V_prop, lambdas, PZ_lambdas, f, w, h);

%% save and load
filename = sprintf('%s/candidates-%s-%ix%i.mat',evalpath,targetname,w,h);

%% save
save(filename, 'CANDIDATES', '-v7.3')
warning on

%% load
% load(filename, 'CANDIDATES')

%% get certified pose estimates

% settings
posesType = 'random';
switch posesType
    case 'plane_landing'
        S_poses = [
        0 100 300 45 0 0;
        0 70 225 50 0 0;
        0 20 120 60 0 0
        0 0 80 70 0 0;  
        0 -15 60 80 0 0;
        ]';
    case 'random'
        numSamples = 100;
        S_poses = nan(6,numSamples);
        fprintf('Searching for sample poses.. \n')
        for i = 1:numSamples
            S_pose = POSE.sup + 1;
            while ~any(arrayfun(@(CAND) contains(CAND.PZ_pose,S_pose), CANDIDATES))
                S_pose = POSE.randPoint();
            end
            S_poses(:,i) = S_pose;
        end
        fprintf('Done.\n')

    otherwise
        throw(CORAerror('CORA:wrongValue',posesType))

end

S_poses = [S_poses(1:3,:);S_poses(4:6,:)*pi/180];

%% test samples
S_res = pose_testSamples(S_poses, [], V_prop, V_TCF, CANDIDATES, f, w, h);

%% display results

% filter valid samples
% (although we checked above that they are actually visible on the image
% according to our candidates, we still have to filter some 
% due to the outer approximation)
idx = S_res.volApproach > 0;
timeFilter = S_res.timeFilter(idx);
timeApproach = S_res.timeApproach(idx);
volFilter = S_res.volFilter(idx);
volApproach = S_res.volApproach(idx);
volAll = S_res.volAll;
numCands = S_res.numCands(idx);
numCritPixels = S_res.numCritPixels(idx);

% display
table = CORAtable('minimalistic',{'Filter [s]', 'Approach [s]', 'Filter [%]', 'Approach [%]', 'Candidates', '#Crit Pixels'},{'sum{%.2f & %.2f}','sum{%.2f & %.2f}','sum{%.2f & %.2f}','sum{%.2f & %.2f}','sum{%.1f & %.1f}','sum{%.1f & %.1f}'});
table.printHeader();
table.printContentRow({timeFilter,timeApproach,volFilter ./ volAll * 100, volApproach ./ volAll * 100, numCands,numCritPixels})
table.printFooter();

% assert(contains(P_total,S_alphas))

disp('Done.')


