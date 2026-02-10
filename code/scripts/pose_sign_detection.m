
function pose_sign_detection(datapath,evalpath)

%% real-world experiment: sign detection

fprintf('\n--- Slow Vehicle Sign ---\n')

% settings
h = 480; w = 640;
f = 0.008 / 15e-6; % focal length / pixel size 

% get target
fprintf('Loading target.. ')
targetname = 'slowvehicle';
[V_TCF, B_TCF, V_prop] = pose_getTarget(targetname,true);
% make all outer vertices valid
V_prop.alwaysValid = [true(6,1);false(3,1)];
I_scale = interval([1;1]);
[lambdas, PZ_lambdas] = pose_getLambdasScaling(B_TCF, V_TCF, V_prop, I_scale);
disp('Done.')

% load noisy images and respective poses
fprintf('Loading data set.. ')
load(sprintf('%s/data.mat',datapath), 'Is_test', 'S_poses_expanded')
% for cleaned images
% Is_test = convn(Is_test,[0 1 0; 1 0 1; 0 1 0],"same") >= 4;
disp('Done.')

POSE = interval([-0.5;-0.5;0.5;-0.087;-0.087;-0.26],[0.5;0.5;1.5;0.087;0.087;0.26]);
pose_printUncertainty(POSE);
assert(all(contains(POSE, S_poses_expanded)))
disp(' ')

%% OFFLINE PREPARATION ---

warning off
CANDIDATES = pose_gatherCandidates(POSE, B_TCF, V_prop, lambdas, PZ_lambdas, f, w, h);

%% save and load
filename = sprintf('%s/candidates-%s-%ix%i.mat',evalpath,targetname,w,h);

%% save
save(filename, 'CANDIDATES', '-v7.3')
warning on

%%
% load(filename, 'CANDIDATES')

%% test samples

maxNoiseLevel = 0.01;
S_res = pose_testSamples(S_poses_expanded, Is_test, V_prop, V_TCF, CANDIDATES, f, w, h, maxNoiseLevel);

%%

for i=1:size(S_poses_expanded); assertLoop(any(cellfun(@(P) contains(P, S_poses_expanded(:,i)), S_res.polytopes{i})),i); end
disp('All poses contained!')

idx = S_res.volApproach > 0;

% display
table = CORAtable('minimalistic',{'Filter [s]', 'Approach [s]', 'Filter [%]', 'Approach [%]', 'Candidates', '#Crit Pixels'},{'sum{%.2f & %.2f}','sum{%.2f & %.2f}','sum{%.2f & %.2f}','sum{%.2f & %.2f}','sum{%.1f & %.1f}','sum{%.1f & %.1f}'});
table.printHeader();
table.printContentRow({S_res.timeFilter(idx),S_res.timeApproach(idx),S_res.volFilter(idx) ./ S_res.volAll * 100, S_res.volApproach(idx) ./ S_res.volAll * 100, S_res.numCands(idx),S_res.numCritPixels(idx)})
table.printFooter();

