function S_res = pose_testSamples(S_poses, Is, V_prop, V_TCF, CANDIDATES, f, w, h, maxNoiseLevel)
% pose_encloseSupportFunc - encloses all convex regions
%
% Syntax:
%    convRegions = pose_testSamples(S_poses, Is, V_prop, V_TCF, CANDIDATES, f, w, h)
%
% Inputs:
%    S_poses - numeric, sample poses
%    Is - (optional) numeric, corresponding images
%    V_prop - struct, properties of target
%    CANDIDATES - struct, offline prepared candidates
%    f, w, h - numeric, intrinsic parameters
%    maxNoiseLevel - numeric, max number of noisy pixels, in %
%
% Outputs:
%    S_res - struct, evaluation results
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: pose_gatherCandidates

% ------------------------------ BEGIN CODE -------------------------------

% parse input
if nargin < 9
    maxNoiseLevel = 0;
end
maxNoisyPixels = maxNoiseLevel*(h*w);

% setup
timeFilter = nan(1,size(S_poses,2));
timeApproach = nan(1,size(S_poses,2));
volFilter = nan(1,size(S_poses,2));
volApproach = nan(1,size(S_poses,2));
numCands = nan(1,size(S_poses,2));
numCritPixels = nan(1,size(S_poses,2));
polytopes = cell(1,size(S_poses,2));
doPlot = false;

% gather images 
CANDIDATE_IMAGES = cat(3,CANDIDATES.isContained);

% compute volume of all candidates (not POSE, as there are skipped hypercubes!)
volAll = sum(arrayfun(@(CAND) volume(interval(CAND.PZ_pose)), CANDIDATES));

for i=1:size(S_poses,2)
    fprintf('### SAMPLE %i/%i ###\n',i,size(S_poses,2))

    S_pose = S_poses(:,i);
    
    % validity test
    % CAND = CANDIDATES(arrayfun(@(cand) contains(interval(cand.PZ_pose),S_pose), CANDIDATES));
    %assert(~isempty(CAND))
    
    % compute image
    [~,~,S_l_PCF] = pose_forward(V_prop, V_TCF, S_pose, f, w, h);
    PZ_S_L_PCF = polyZonotope(reshape(S_l_PCF(:,~V_prop.jumps),[],1));
    convRegions = pose_encloseSupportFunc(V_prop, PZ_S_L_PCF);
    if isempty(Is)
        [I,I_pixels] = pose_getContainedPixels(w,h,convRegions);
    else
        I = Is(:,:,i);
    end

    % visualize 
    if doPlot
        figure(f1)
        subplot(1,size(S_poses,2),i); hold on;
        pose_plotSetup('I',true,w,h)
        pose_plot(V_prop,'I_pgon',I_pixels,1:2)
        drawnow
    end

    % filter candidates
    

    timerVal = tic;
    fprintf('Filtering from %i candidates... ', numel(CANDIDATES))
    % check if image is in candidate
    idx = sum(~(~I | (I == CANDIDATE_IMAGES)),1:2) <= maxNoisyPixels;
    FILTERED_CANDIDATES = CANDIDATES(idx);
    fprintf('to %i.. ', numel(FILTERED_CANDIDATES));
    
    % check if each vertex is contained
    FILTERED_CANDIDATES_V = arrayfun(@(cand) cand.V_PCF.isContained, FILTERED_CANDIDATES, 'UniformOutput', false);
    FILTERED_CANDIDATES_V = cat(4,FILTERED_CANDIDATES_V{:});
    idx = all(any(I & FILTERED_CANDIDATES_V,1:2),3);
    FILTERED_CANDIDATES = FILTERED_CANDIDATES(idx);
    fprintf('to %i candidates!\n', numel(FILTERED_CANDIDATES));
    timeFilter(i) = toc(timerVal);
    fprintf('Time to filter candidates: %.2f\n',timeFilter(i))
    numCands(i) = numel(FILTERED_CANDIDATES);
    
    % CAND_true = FILTERED_CANDIDATES(arrayfun(@(cand) contains(interval(cand.PZ_pose),S_pose), FILTERED_CANDIDATES));
    % assert(~isempty(CAND_true),'Sampled point is not within filtered candidate!')
    
    % visualize
    if doPlot
        figure(f2); hold on; box on; grid on;
        xlim([POSE.inf(1) POSE.sup(1)])
        ylim([POSE.inf(2) POSE.sup(2)])
        zlim([POSE.inf(3) POSE.sup(3)])
        plotPoints(S_pose,1:3,'o','Color',CORAcolor('CORA:yellow'))
    end

    % run through all candidates
    table = CORAtable("minimalistic",{' #','Time','#processed','[%]'},{'rownr','time-detailed','i','.2f'});
    table.printHeader();
    Ps = cell(1,numel(FILTERED_CANDIDATES));
    numCritPixels(i) = 0;
    for c=1:(numel(FILTERED_CANDIDATES)+1)
        
        % print status update (after first, second, fifth, then every tenth)
        if c == 1 || c == 2 || c == 5 || mod(c, 10) == 0 || c > numel(FILTERED_CANDIDATES)
            table.printContentRow({[],[],c-1,(c-1)/numel(FILTERED_CANDIDATES)*100})
        end
        if c > numel(FILTERED_CANDIDATES)
            break
        end
        
        CAND = FILTERED_CANDIDATES(c);
        
        % gather critical pixels
        [critPixels,sfPixels] = pose_gatherCriticalPixels(I,CAND,maxNoisyPixels);

        % compute constraints
        [P_total,Ps_c] = pose_enclosePointsViaConstraints(critPixels,CAND);
        if representsa(P_total,'emptySet')
            continue
        end

        idx = ~isnan(sfPixels(1,:));
        if sum(idx) > 0
            numCritPixels(i) = numCritPixels(i) + sum(cellfun(@(critPixels) size(critPixels,2), critPixels(idx))) / sum(idx) / numel(FILTERED_CANDIDATES);
        end    

        % scale P_total to true pose
        I_pose = interval(CAND.PZ_pose);
        scale = rad(I_pose);
        shift = center(I_pose);
        P_trans = diag(scale) * P_total + shift;
            
        % visualize
        if doPlot
            pose_plot(V_prop,'C',P_trans,1:3,'Color',CORAcolor(i))
            drawnow
        end
    
        % save polytope
        Ps{c} = P_trans;
    
        % figure;
        % pose_plot(V_prop,'CRIT_PIXELS',critPixels,1:2)
        % pose_plot(V_prop,'CRIT_PIXELS',sfPixels,1:2)
        % 
        % pose_plotSetup('H_POS',true,w,h)
        % pose_plot(V_prop,'C',P_total,1:3)
        % % pose_plot(V_prop,'CRIT_PIXELS',S_alphas,1:3)
        % 
        % pose_plotSetup('H_ANGLES',true,w,h)
        % pose_plot(V_prop,'C',P_total,4:6)
        % % pose_plot(V_prop,'CRIT_PIXELS',S_alphas,4:6)
        % drawnow
    end
    table.printFooter();

    % compute metrics
    timeApproach(i) = toc(timerVal);
    fprintf('Time of approach: %.2f\n',timeApproach(i))
    fprintf('Computing metrics.. ')
    try
        volFilter(i) = sum(arrayfun(@(CAND) volume(interval(CAND.PZ_pose)), FILTERED_CANDIDATES));
        % filter empty sets
        Ps = Ps(cellfun(@(P_trans) isa(P_trans,'polytope'),Ps));
        polytopes{i} = Ps;
        % volume computation sometimes fails ...
        volApproach(i) = sum(cellfun(@(P_trans) volume(P_trans), Ps));
    catch
        % in case it fails, volApproach will have nan values -> filter later
    end
    fprintf('Done.\n\n')
end

S_res = struct;
S_res.timeFilter = timeFilter;
S_res.timeApproach = timeApproach;
S_res.volFilter = volFilter;
S_res.volApproach = volApproach;
S_res.volAll = volAll;
S_res.numCands = numCands;
S_res.numCritPixels = numCritPixels;
S_res.polytopes = polytopes;



% Auxiliary functions -----------------------------------------------------


% ------------------------------ END OF CODE ------------------------------
