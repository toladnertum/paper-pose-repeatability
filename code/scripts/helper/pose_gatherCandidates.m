function CANDIDATES = pose_gatherCandidates(POSE, B_TCF, V_prop, lambdas, PZ_lambdas, f, w, h)
% pose_gatherCandidates - gathers and prepares all candidates offline
%
% Syntax:
%    CANDIDATES = pose_gatherCandidates(POSE, B_TCF, V_prop, lambdas, PZ_lambdas, f, w, h)
%
% Inputs:
%    POSE - numeric, entire pose space
%    B_TCF - numeric, boundary/reference in TCF
%    V_prop - struct, properties of target
%    lambdas, PZ_lambdas - numeric, polyZonotope, scaling parameters
%    f, w, h - numeric, intrinsic parameters
%
% Outputs:
%    CANDIDATES - struct
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: pose_testSamples

% ------------------------------ BEGIN CODE -------------------------------


disp('Offline preparations.. ---')

% settings und setup
maxIteration = 50;
minIteration = 4;
splitType = 'sensitivity+len';

changeIterAt = maxIteration;
maxErrRatio = 0.3;

% show statistics table
table = CORAtable("minimalistic",{' #','Time','#queued','#finished', '#skipped'},{'rownr','time-detailed','i','i','i'});
table.printHeader();

% recursively partition pose space
queue = {POSE};
finishedQueue = cell(maxIteration,1);
numSkipped = 0;
for i = 1:(maxIteration+1)
    table.printContentRow({[],[],numel(queue),sum(cellfun(@(finishedQueue_i) numel(finishedQueue_i), finishedQueue)),numSkipped})
    
    if isempty(queue) || i > maxIteration
        break
    end
    
    % init queues
    nextQueue = cell(1,2^3*numel(queue));
    finishedQueue_i = cell(1,numel(queue));

    for c=1:numel(queue)
        % read out current pose
        pose_c = queue{c};
        PZ_pose = polyZonotope(pose_c(1:6));

        try
            % compute forward pass
            [~,~,PZ_l_PCF] = pose_forward(V_prop, B_TCF, PZ_pose, f, w ,h, PZ_lambdas);

            % quick check if visible on image
            V_I = reshape(interval(PZ_l_PCF),2,[]);

            if all(any((V_I.sup < 0) | (V_I.inf > [w;h]), 1), 2)
                numSkipped = numSkipped + 1;
                continue
            end

            % determine error ratio
            errRatio = pose_computeErrorRatio(PZ_l_PCF);
            
        catch ME
            % split set if error occurred due to target being too close
            if isempty(ME.identifier) && contains(ME.message,'Target too close to camera')
                errRatio = inf;
            else
                rethrow(ME)
            end
        end
        
        % check if error is small enough
        if i >= minIteration && errRatio <= maxErrRatio
            % pose_c done
            candidate_c = struct;
            candidate_c.PZ_pose = pose_c;
            candidate_c.PZ_l_PCF = PZ_l_PCF;

            finishedQueue_i{c} = candidate_c;
        else
            % split
            switch splitType
                case 'round-robin'
                    splits = 1+mod(i-1,6);
                case 'round-robin2'
                    splits = mod([1 3] + mod(i-1,3));
                case 'position'
                    splits = 1:3;
                case 'angles'
                    splits = 4:6;
                case 'mainly-position'
                    splits = 1:3;
                    if i <= changeIterAt
                        splits = [splits 4];
                    end
                case {'sensitivity','sensitivity+len'}
                    % use (approximative) sensitivity
                    sensitivity = nan(6,1);
                    for s=1:6
                        step = 1e-8;
                        [~,~,l_PCF_m] = pose_forward(V_prop, B_TCF, center(PZ_pose)-step*unitvector(s,6), f, w, h, lambdas);
                        [~,~,l_PCF_p] = pose_forward(V_prop, B_TCF, center(PZ_pose)+step*unitvector(s,6), f, w, h, lambdas);
                        sensitivity(s) = mean(abs(l_PCF_p(:,~V_prop.jumps)-l_PCF_m(:,~V_prop.jumps)),"all");
                    end
                    heuristic = sensitivity;
                    if strcmp(splitType,'sensitivity+len')
                        heuristic = heuristic .* pose_c.rad;
                    end
                    [~,splits] = sort(heuristic,1,"descend");
                    if i < changeIterAt
                        splits = splits(1:3)';
                    else
                        splits = splits(1:2)';
                    end

                otherwise
                    throw(CORAerror('CORA:wrongValue',splitType));
            end

            splitQueue = {pose_c};
            for n=splits
                splitQueue = cellfun(@(I) split(I,n), splitQueue,'UniformOutput',false);
                splitQueue = [splitQueue{:}];
            end
            nextQueue((c-1)*2^nnz(splits)+(1:2^nnz(splits))) = splitQueue;
        end
    end

    % prepare for next iteration
    queue = nextQueue(cellfun(@(entry) ~isnumeric(entry), nextQueue));
    finishedQueue{i} = [finishedQueue_i{:}];

end
finishedQueue = [finishedQueue{:}];
table.printFooter();
assert(isempty(queue),'Warning: Queue is not empty!')


%% finish offline preparation
disp('Finalizing offline preparation.. ')
CANDIDATES = cell(1,numel(finishedQueue));
table = CORAtable("minimalistic",{' #','Time','#processed','[%]'},{'rownr','time-detailed','i','.2f'});
table.printHeader();
for i=1:numel(finishedQueue)
    cand_c = finishedQueue(i);
    CANDIDATES{i} = pose_prepareCandOffline(V_prop,cand_c.PZ_pose, cand_c.PZ_l_PCF, w, h);

    % print status update (after first, second, fifth, then every tenth)
    if i == 1 || i == 2 || i == 5 || mod(i, 10) == 0 || i == numel(finishedQueue)
        table.printContentRow({[],[],i,i/numel(finishedQueue)*100})
    end
end
table.printFooter();
CANDIDATES = cell2mat(CANDIDATES);
disp('Done.')

end

% ------------------------------ END OF CODE ------------------------------
