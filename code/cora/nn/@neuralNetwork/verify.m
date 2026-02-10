function [res, x_, y_] = verify(nn, x, r, A, b, safeSet, varargin)
% verify - automated verification for specification on neural networks.
%
% Syntax:
%    [res, z] = nn.verify(x, r, A, b, options)
%
% Inputs:
%    nn - object of class neuralNetwork
%    x, r - center and radius of the initial set (can already be a batch)
%    A, b - specification, prove A*y <= b
%    safeSet - bool, safe-set or unsafe-set [optional: if > 1 then the
%       number of union (safe set) constraints.]
%    options - struct, evaluation options
%    timeout - positive integer, timeout in seconds
%    verbose - bool, print verbose output
%    plotDims - 2x2 plot dimensions; empty for no plotting; 
%       plotDims(1,:) for input and plotDims(2,:) for output; sets 
%       are stored in res.Xs, res.uXs
%    plotSplittingTree - bool, if split tree should be create for
%       visualization
%
% Outputs:
%    res - result: true if specification is satisfied, false if not, empty if unknown
%    x_ - counterexample in terms of an initial point violating the specs
%    y_ - output for x_
%
% References:
%    [1] VNN-COMP'24
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: -

% Authors:       Lukas Koller
% Written:       23-November-2021
% Last update:   14-June-2024 (LK, rewritten with efficient splitting)
%                20-January-2025 (LK, constraint zonotope splitting)
% Last revision: ---

% ------------------------------ BEGIN CODE -------------------------------

% Check number of input arguments.
narginchk(6,11);

% Validate parameters.
[options, timeout, verbose, plotDims, plotSplittingTree] = ...
    setDefaultValues({struct, 100, false, [], false}, varargin);
plotting = ~isempty(plotDims);

% Validate parameters.
inputArgsCheck({ ...
    {nn, 'att','neuralNetwork'}; ...
    {x, 'att',{'numeric','gpuArray'}}; ...
    {r, 'att',{'numeric','gpuArray'}}; ...
    {A, 'att',{'numeric','gpuArray'}}; ...
    {b, 'att',{'numeric','gpuArray'}}; ...
    {options,'att','struct'}; ...
    {timeout,'att','numeric','scalar'}; ...
    {verbose,'att','logical'}; ...
    {plotting,'att','logical'}; ...
})
options = nnHelper.validateNNoptions(options,true);

nSplits = options.nn.num_splits; % Number of input splits per dimension.
nDims = options.nn.num_dimensions; % Number of input dimension to splits.
nNeur = options.nn.num_neuron_splits; % Number of neurons to split.
nReLU = options.nn.num_relu_constraints; % Number of ReLU constraints.

% Extract parameters.
bSz = options.nn.train.mini_batch_size;

% Obtain number of input dimensions.
n0 = size(x,1);
% Limit the number of dimensions to split.
nDims = min(nDims,n0);
% Check the maximum number of input generators.
numInitGens = min(options.nn.train.num_init_gens,n0);
% Obtain the maximum number of approximation errors in an activation layer.
nk_max = max(cellfun(@(li) ...
    isa(li,'nnActivationLayer')*prod(li.getOutputSize(li.inputSize)), ...
    nn.layers) ...
);
% We always have to use the approximation during set propagation to ensure
% soundness.
options.nn.use_approx_error = true;
% Ensure the interval-center flag is set, if there are less generators than
% input dimensions.
options.nn.interval_center = ...
    (options.nn.train.num_approx_err < nk_max) | (numInitGens < n0);

% Specify indices of layers for propagation.
idxLayer = 1:length(nn.layers);
% Enumerate the layers of the neural networks.
[layers,~,~,~] = nn.enumerateLayers();

% To speed up computations and reduce GPU memory, we only use single 
% precision.
inputDataClass = single(1);
% Check if a GPU is used during verification.
if options.nn.train.use_gpu
    % Training data is also moved to GPU.
    inputDataClass = gpuArray(inputDataClass);
end
% Move weights of the neural network to GPU.
nn.castWeights(inputDataClass);
% Clear all stored variables in all layers.
layers = aux_clearLayerFields(layers);

% Move the specification ot the GPU.
A = cast(A,'like',inputDataClass);
b = cast(b,'like',inputDataClass);

% In each layer, store ids of active generators and identity matrices 
% for fast adding of approximation errors.
q = nn.prepareForZonoBatchEval(x,options,idxLayer);

% Initialize queue.
xs = x;
rs = r;
nrXs = zeros([0 size(x,2)]);
% Compute number of union constraints (all intersection constrains -> 
% union of one single constraint).
if safeSet > 1
    numUnionConst = safeSet; 
elseif safeSet == 1
    numUnionConst = size(A,1);
else
    % There are no union constraints.
    numUnionConst = 1;
end
% Reset to be a bool.
safeSet = logical(safeSet);
% Initialize result.
res.str = 'UNKNOWN';
x_ = [];
y_ = [];

% Initialize iteration stats.
numVerified = 0;
% Initialize iteration counter.
iter = 1;

% Specify the batch variables so we can clear them after every iteration.
batchVars = {'xi','ri','nrXi','S_','S','sens','cxi','Gxi',...
    'yic','yid','Gyi','ld_yi','ld_Gyi'};

if plotting
    % Initialize cell arrays to store intermediate results for plotting.
    res.Xs = {};
    res.Ys = {};
    res.xs_ = {};
    res.ys_ = {};
    
    % Compute samples.
    sampX = randPoint(interval(x - r,x + r),1000);
    sampY = gather(double(nn.evaluate(sampX)));

    % Create a new figure.
    fig = figure('Theme','light');
    % Initialize plot.
    [fig,hx0,hspec] = aux_initPlot(fig,plotDims, ...
        sampX,sampY,x,r,A,b,safeSet);
    drawnow;
end
if plotSplittingTree
    % Store the splitting tree; initialize only with the root node.
    res.tree = 0;
    res.tlabels = {'root'};
    res.qIdx2tIdx = 1;
end

if verbose
    % Setup table.
    table = CORAtable('double', ...
        {'Iteration','#Queue','#Verified Branches','avg. Radius', ...
            '~Unknown Vol. [%]'}, ...
        {'d','d','d','.3e','.4f'});
    table.printHeader();
end

% Specify the heuristics.
% Input-Generator Options: 
%  {'most-sensitive-input-radius',
%   'zono-norm-gradient'}.
inputGenHeuristic = options.nn.input_generator_heuristic;

% Input-Split Options: 
%  {'most-sensitive-input-radius',
%   'zono-norm-gradient'}.
inputSplitHeuristic = options.nn.input_split_heuristic;

% Neuron-Split Options: 
%  {'least-unstable', 
%   'most-sensitive-approx-error',
%   'most-sensitive-input-radius',
%   'zono-norm-gradient',
%   'most-sensitive-input-aligned'}.
neuronSplitHeuristic = options.nn.neuron_split_heuristic;

% Specify the heurstic for selecting the ReLU-neurons to constrain.
% Options:
%  {'most-sensitive-input-radius',
%   'most-sensitive-approx-error',
%   'zono-norm-gradient'}.
reluConstrHeuristic = options.nn.relu_constraint_heuristic;

% The inputs are needed for the neuron splitting, relu tightening 
% constraints, or layerwise refinement.
storeInputs = nReLU > 0 || nNeur > 0 ...
     || strcmp(options.nn.refinement_method,'zonotack-layerwise') ...
     || strcmp(inputGenHeuristic,'zono-norm-gradient') ...
     || strcmp(inputSplitHeuristic,'zono-norm-gradient');
% The sensitivity is used for selecting input generators, neuron
% -splitting, and FGSM attacks.
computeAndStoreSensitivity = ...
    any(strcmp(inputGenHeuristic,{'most-sensitive-input-radius'})) ...
|| any(strcmp(inputSplitHeuristic,{'most-sensitive-input-radius'})) ...
|| any(strcmp(neuronSplitHeuristic,{ ...
    'most-sensitive-approx-error','most-sensitive-input-radius'})) ...
|| any(strcmp(reluConstrHeuristic,{ ...
    'most-sensitive-approx-error','most-sensitive-input-radius'})) ...
|| strcmp(options.nn.falsification_method,'fgsm') ...
|| (strcmp(options.nn.approx_error_order,'sensitivity*length') ...
                && (options.nn.train.num_approx_err < nk_max));

timerVal = tic;

% Main splitting loop.
while size(xs,2) > 0

    try % Catch any error and adapt parameters accordingly. 

        % Check if we reach the maximum number of iterations.
        if iter > options.nn.max_verif_iter
            break;
        end

        time = toc(timerVal);
        if time > timeout
            % Time is up.
            res.time = time;
            break;
        end
    
        if verbose
            % Print the iteration stats.
            aux_printIterationStats(table,iter,xs,r,rs,numVerified);
        end
    
        % Pop next batch from the queue.
        [xi,ri,nrXi,xs,rs,nrXs,qIdx] = aux_pop(xs,rs,nrXs,bSz,options);

        if plotSplittingTree
            % Obtain the indics of the parent nodes in the tree.
            pIdx = res.qIdx2tIdx(qIdx);
        end

        % Obtain the current batch size.
        [~,cbSz] = size(xi);
        % Move the batch to the GPU.
        xi = cast(xi,'like',inputDataClass);
        ri = cast(ri,'like',inputDataClass);
        nrXi = cast(nrXi,'like',inputDataClass);

        % Initialize flag indicating if the input to layers have to be
        % cleared.
        clearLayerInputs = false;
    
        if computeAndStoreSensitivity
            % Compute the sensitivity (store sensitivity for neuron-splitting).
            [S,~] = nn.calcSensitivity(xi,options,true);
            % We have to clear the layer inputs.
            clearLayerInputs = true;
            % The sensitivity should not be lower than 1e-3, otherwise it 
            % is too low to be effective for the (neuron-) splitting 
            % heuristic.
            sens = reshape(max(max(abs(S),1e-6),[],1),[n0 cbSz]); 

            % TODO: investigate a more efficient implementation of the 
            % sensitivity computation using backpropagation.
        else
            % We do not need to compute the sensitivity.
            S = [];
            sens = [];
        end
    
        % 1. Verification -------------------------------------------------
        % 1.1. Use batch-evaluation of zonotopes.

        if (numInitGens < n0) && strcmp(inputGenHeuristic,'zono-norm-gradient')
            % Compute the gradient of the interval norm.
                   
            % Store inputs for the gradient computation. 
            options.nn.train.backprop = true;
            % Compute an interval enclosure of the output set.
            Yival = nn.evaluate_(interval(xi-ri,xi+ri),options,idxLayer);
            % Store inputs for the gradient computation. 
            options.nn.train.backprop = false;
            % Compute gradient of the zonotope norm.
            ivalGrad = aux_updateGradients(nn,options,idxLayer,Yival, ...
                A,b,false);
            % We have to clear the layer inputs.
            clearLayerInputs = true;
        else
            % We do not need to compute the interval gradient.
            ivalGrad = [];
        end

        if clearLayerInputs
            % Clear the stored inputs.
            layers = aux_clearLayerFields(layers);
        end
    
        % Construct input zonotope. 
        [cxi,Gxi,inputDimIds] = aux_constructInputZonotope(options, ...
            inputGenHeuristic,xi,ri,sens,ivalGrad,numInitGens);

        if ~isempty(nrXi)
            % Obtain the number of previous split constraints.
            [p,~] = size(nrXi); 
            % For now we only support storing ReLU splits at 0. 
            % TODO: remember arbitrary splits. 
            % Therefore, create dummy offsets.
            dummyd = zeros([p cbSz],'like',xi);
            % Store computed bounds in the layers for tighter 
            % approximations.
            for i=1:length(layers)
                % Obtain the i-th layer.
                layeri = layers{i};
                if isa(layeri,'nnActivationLayer')
                    % Obtain the indices of the neurons of the current 
                    % layer.
                    neuronIds = layeri.neuronIds;
                    % Create dummy centers.
                    dummyc = zeros([length(neuronIds) cbSz],'like',xi);
                    % Compute bounds from previous splits. 
                    [li,ui] = aux_obtainBoundsFromSplits(neuronIds,cbSz, ...
                        nrXi,dummyd,dummyc);
                    % Store the computed bounds in the layers.
                    layeri.backprop.store.l = li;
                    layeri.backprop.store.u = ui;
                end
            end
        end

        % Store inputs for each layer by enabling backpropagation. 
        options.nn.train.backprop = storeInputs;
        % Compute output enclosure.
        [yi,Gyi] = nn.evaluateZonotopeBatch_(cxi,Gxi,options,idxLayer);
        % Disable backpropagation.
        options.nn.train.backprop = false;
    
        % Obtain number of output dimensions.
        [nK,~] = size(yi);
    
        % 2.2. Compute logit difference.
        [ld_yi,ld_Gyi,ld_Gyi_err,yic,yid,Gyi] = ...
            aux_computeLogitDifference(yi,Gyi,A,options);
        % Compute the radius of the logit difference.
        ld_ri = sum(abs(ld_Gyi),2) + ld_Gyi_err;
        % 2.3. Check specification.
        if safeSet
            % safe iff all(A*y <= b) <--> unsafe iff any(A*y > b)
            % Thus, unknown if any(A*y > b).
            unknown = any(ld_yi + ld_ri(:,:) > b,1);
        else
            % unsafe iff all(A*y <= b) <--> safe iff any(A*y > b)
            % Thus, unknown if all(A*y <= b).
            unknown = all(ld_yi - ld_ri(:,:) <= b,1);
        end

        % Update counter for verified patches.
        numVerified = numVerified + sum(~unknown,'all');
    
        if plotting
            % % Reset the figure.
            % clf(fig);
            % [fig,hx0,hspec] = aux_initPlot(fig,plotDims,x,r,A,b,safeSet);
            % Store input sets.
            if options.nn.interval_center
                xid = 1/2*(cxi(:,2,:) - cxi(:,1,:));
                res.Xs{end+1} = struct( ...
                    'c',gather(xi),'G',gather(cat(2,Gxi,xid.*eye(n0))),...
                    'verified',gather(~unknown) ...
                );
                % Store the output set.
                res.Ys{end+1} = struct( ...
                    'c',gather(yic),'G',gather(cat(2,Gyi,yid.*eye(nK))) ...
                );
            else
                res.Xs{end+1} = struct('c',gather(cxi),'G',gather(Gxi),...
                    'verified',gather(~unknown));
                % Store the output set.
                res.Ys{end+1} = struct('c',gather(yic),'G',gather(Gyi));
            end
            % Plot current input sets and propagated output sets.
            [fig,hxi,hx,hxv,hy,hyv] = aux_plotInputAndOutputSets(fig, ...
                plotDims,x,r,res);
            drawnow;
        end 
        if plotSplittingTree
            % Mark verified nodes in the splitting tree.
            [res.tlabels,pIdx] = aux_markVerifiedInTheSplittingTree( ...
                res.tlabels,pIdx,~unknown,'ZP');
        end
    
        if all(~unknown)
            % Verified all subsets of the current batch. We can skip to 
            % next iteration.
            iter = iter + 1;
            continue;
        elseif any(~unknown)
            % Only keep un-verified patches.
            % Queue entries....
            xi(:,~unknown) = [];
            ri(:,~unknown) = [];
            nrXi(:,~unknown) = [];
            % Sensitivity...
            if ~isempty(S)
                S(:,:,~unknown) = [];
            end
            if ~isempty(sens)
                sens(:,~unknown) = [];
            end
            % Input sets...
            inputDimIds(:,~unknown) = [];
            if options.nn.interval_center
                cxi(:,:,~unknown) = [];
                yi(:,:,~unknown) = [];
            else
                cxi(:,~unknown) = [];
                yi(:,~unknown) = [];
            end
            Gxi(:,:,~unknown) = [];

            % Output sets...
            yic(:,~unknown) = [];
            Gyi(:,:,~unknown) = [];
            yid(:,:,~unknown) = [];
            % Logit difference...
            ld_yi(:,~unknown) = [];
            ld_Gyi(:,:,~unknown) = [];
    
            if storeInputs || computeAndStoreSensitivity
                % % Specify field that are not batched.
                % excludedFields = {...
                %     'genIds','neuronIds','approxErrGenIds',...
                %     'scale','offset'};

                % Update the sets and sensitivity matrices; remove the
                % verified batch entries.
                for i=1:length(layers)
                    % Obtain the i-th layer.
                    layeri = layers{i};

                    % % Obtain all fields.
                    % fieldsi = fieldnames(layeri.backprop.store);
                    % % Exclude fields that are not batched.
                    % fieldsi = fieldsi(arrayfun( ...
                    %     @(f) ~any(strcmp(f,excludedFields)),fieldsi));
                    % % Remove the verified batch entries.
                    % for j=1:length(fieldsi)
                    %     fprintf('Field name: %s\n',fieldsi{j});
                    %     % Obtain the j-th field.
                    %     fieldij = layeri.backprop.store.(fieldsi{j});
                    %     % We only want to index the last dimension.
                    %     subs = repmat({':'},1,ndims(fieldij));  
                    %     subs{end} = ~unknown;
                    %     % Remove the verified batch entries.
                    %     fieldij(subs{:}) = []; 
                    %     % Update the field.fieldsj
                    %     layeri.backprop.store.(fieldsi{j}) = fieldij;
                    % end

                    % Update the sensitivity.
                    if computeAndStoreSensitivity
                        layeri.sensitivity(:,:,~unknown) = [];
                    end
                end
            end

            if storeInputs
                % Recompute the hidden sets and approximations. TODO: reuse
                % the computations; tricky to adapt, because of the 
                % indices stored in nnActivationLayer.
                options.nn.train.backprop = true;
                nn.evaluateZonotopeBatch_(cxi,Gxi,options,idxLayer);
                options.nn.train.backprop = false;
            end
        end
        % Update the current batch size.
        [~,cbSz] = size(xi);
    
        % 2. Falsification ----------------------------------------------------
    
        % 2.1. Compute adversarial examples.
        switch options.nn.falsification_method
            case 'fgsm'
                % Obtain number of constraints.
                [p,~] = size(A);
                % Try to falsification with a FGSM attack.
                if safeSet
                    grad = pagemtimes(-A,S);
                    % We combine all constraints for a stronger attack.
                    p = 1;
                else
                    grad = pagemtimes(A,S);
                end
                % If there are multiple output constraints we try to falsify
                % each one individually.
                sgrad = reshape(permute(sign(grad),[2 3 1]),[n0 cbSz*p]);
        
                % Compute adversarial attacks based on the sensitivity.
                xi_ = repelem(xi,1,p) + repelem(ri,1,p).*sgrad;

                % Clear unused variables.
                clear('grad','sgrad');
            case 'center'
                % Use the center for falsification.
                xi_ = xi;
            case 'zonotack'
                % Obtain number of constraints.
                [p,~] = size(A);
                % Compute the vertex that minimizes the distance to each 
                % halfspace.
                beta_ = -permute(sign(ld_Gyi(:,1:numInitGens,:)),[2 4 1 3]);
                if safeSet
                    % We have to reverse the signs for safe sets.
                    beta_(:,:,1:numUnionConst,:) = -beta_(:,:,1:numUnionConst,:);
                end
                % Put multiple candidates into the batch.
                beta = reshape(beta_,[numInitGens 1 p*cbSz]);
    
                % Compute attack.
                delta = pagemtimes(repelem(Gxi(:,1:numInitGens,:),1,1,p),beta);
                % Compute candidates for falsification.
                xi_ = repelem(xi,1,p) + delta(:,:);

                % Clear unused variables.
                clear('beta_','beta','delta');
            otherwise
                % Invalid option.
                throw(CORAerror('CORA:wrongFieldValue', ...
                    'options.nn.falsification_method', ...
                        {'fgsm','center','zonotack'}));
        end
    
        % 2.2. Check the specification for adversarial examples.
        
        % Check the adversarial examples.
        [~,critVal,falsified,x_,y_] = ...
            aux_checkPoints(nn,options,idxLayer,A,b,safeSet,xi_);
    
        if any(falsified)
            % Found a counterexample.
            res.str = 'COUNTEREXAMPLE';
            break;
        end
    
        % Check if the batch was extended with multiple candidates.
        if size(critVal,2) > cbSz
            critVal_ = reshape(critVal,1,cbSz,[]);
            % Find the worst candidate: If any candicate has a negative 
            % criticallity value, we have a counterexample.
            critVal = min(critVal_,[],3);
        end
    
        % 3. Refine input sets. -------------------------------------------
    
        if strcmp(inputSplitHeuristic,'zono-norm-gradient') ...
           || (nNeur > 0 && any(strcmp(neuronSplitHeuristic, ...
                {'zono-norm-gradient','least-unstable-gradient'}))) ...
           || (nReLU > 0 && strcmp(reluConstrHeuristic,'zono-norm-gradient'))
            % Specify the gradient of the approximation errors should be
            % stored.
            storeGradients = any(strcmp( ...
                {neuronSplitHeuristic,reluConstrHeuristic}, ...
                {'zono-norm-gradient','least-unstable-gradient'}));
            % Store the gradients of the approximation errors.
            grad = aux_updateGradients(nn,options,idxLayer,Gyi,A,b, ...
                storeGradients);

            % Compute indices for the gradient of the interval 
            % norm w.r.t. the different generators.
            dimGenIdx = reshape(sub2ind(size(grad), ...
                inputDimIds, ...
                repmat((1:numInitGens)',1,cbSz), ...
                repelem(1:cbSz,numInitGens,1)),[numInitGens cbSz]);
            % Compute gradient of the norm.
            grad = reshape(grad(dimGenIdx),[numInitGens cbSz]);
        else
            % There is no stored gradient.
            grad = 0;
        end

        switch options.nn.refinement_method
            case 'naive'
                if any(strcmp(inputSplitHeuristic, ...
                    {'zono-norm-gradient','least-unstable-gradient'}))
                    % Pad the gradient in case not all input dimensions
                    % have a generator.
                    grad_ = zeros([n0 cbSz],'like',grad);
                    % Compute indices.
                    inputDimIdx = reshape(sub2ind(size(xi), ...
                        inputDimIds,repelem(1:cbSz,numInitGens,1)), ...
                            [numInitGens cbSz]);
                    % Set the gradient for the correct dimensions.
                    grad_(inputDimIdx) = grad;
                    grad = grad_;
                end
                % The sets are not refined; split the input dimensions.
                xis = xi;
                ris = ri;
                % Store the indices of the split dimensions.
                dimIds = NaN([0 cbSz],'like',xis);
                for i=1:nDims
                    % Compute the heuristic.
                    his = aux_computeHeuristic(inputSplitHeuristic, ...
                        0, ... the input has layer index 0
                        xis - ris, ... lower bound
                        xis + ris, ... upper bound
                        ris, ... approximation error
                        sens, ... sensitivity
                        grad, ... zonotope norm gradient
                        [],[],[],false,1);
                    % Split the input sets along one dimensions.
                    [xis,ris,dimId] = aux_split(xis,ris,his,nSplits);
                    % Append the split dimension.
                    dimIds = [repmat(dimIds,1,nSplits); dimId];
                    % Replicate sensitivity and criticallity value.
                    sens = repmat(sens,1,nSplits);
                    grad = repmat(grad,1,nSplits);
                    critVal = repmat(critVal,1,nSplits);
                end
                % There is no neuron splitting.
                nrXis = zeros([0 size(xis,2)],'like',Gyi);
            case {'zonotack','zonotack-layerwise'}
    
                % Initialize number of splitted sets.
                newSplits = 1;
    
                % Construct neuron-split constraints.
                if nNeur > 0 && nSplits > 1
                    % Create split constraints for neurons within the network.
        
                    % Construct the constraints.
                    [An,bn,newNrXi,hn] = aux_neuronConstraints(nn,options,[], ...
                        neuronSplitHeuristic,nSplits,nNeur,numInitGens,nrXi);
    
                    % % Compute output-split constraints.
                    % [As,bs] = aux_outputDimSplitConstraints(yi,Gyi,nSplits,nNeur);
                    % newNrXi = NaN(size(As,[1 3]),'like',Gyi);
    
                    % Compute number of new splits.
                    newSplits = nSplits^nNeur*newSplits;
                else
                    % There are no general-split constraints.
                    An = zeros([0 size(Gyi,2) cbSz],'like',Gyi);
                    bn = zeros([0 1 cbSz],'like',Gyi);
                    newNrXi = -ones([0 cbSz],'like',Gyi);
                    hn = -ones([0 cbSz],'like',Gyi);
                end

                % Identify dummy splits; we insert dummy splits to maintain
                % batch size in the case where no meaningful split can be 
                % found.
                isDummySplit = all(isinf(newNrXi),1) & any(isinf(newNrXi),1);
    
                % Construct input split constraints.
                if nDims > 0 && nSplits > 1
                    % When not all input dimensions get an assigned generator
                    % we have to restrict and reorder the dimensions.
                    % Therefore, we compute indices.
                    permIdx = reshape(sub2ind(size(xi), ...
                        inputDimIds,repelem(1:cbSz,numInitGens,1)), ...
                            [numInitGens cbSz]);
                    % Permute the input and radius.
                    xi_ = xi(permIdx);
                    ri_ = ri(permIdx);
                    % Permute the sensitivity.
                    if ~isempty(sens)
                        sens_ = sens(permIdx);
                    else
                        % There is no sensitivity.
                        sens_ = [];
                    end

                    % Compute the heuristic.
                    hi = aux_computeHeuristic(inputSplitHeuristic, ...
                        0, ... the input has layer index 0
                        xi_ - ri_, ... lower bound
                        xi_ + ri_, ... upper bound
                        ri_, ... approximation error
                        sens_, ... sensitivity
                        grad, ... zonotope norm gradient
                        [],[],[],false,1);
    
                    % Compute input-split constraints.
                    [Ai,bi,dimIds,hi] = ...
                        aux_dimSplitConstraints(hi(:,:),nSplits,nDims);
    
                    % Update number of new splits.
                    newSplits = nSplits^nDims*newSplits;
                else
                    % There are no input-split constraints.
                    Ai = zeros([0 size(Gyi,2) cbSz],'like',Gyi);
                    bi = zeros([0 1 cbSz],'like',Gyi);
                    dimIds = NaN([0 cbSz],'like',Gyi);
                    hi = -ones([0 cbSz],'like',Gyi);
                end

                % Pad offsets if there are different number of offsets in general
                % split and input split constraints.
                if size(bn,2) ~= size(bi,2)
                    bn = [bn NaN([nNeur max(size(bi,2)-size(bn,2),0) cbSz],'like',bn)];
                    bi = [bi NaN([nDims max(size(bn,2)-size(bi,2),0) cbSz],'like',bi)];
                end 
                % Append zeros for generators.
                An_ = [An zeros([size(An,1) q-size(An,2) cbSz],'like',An)];
                Ai_ = [Ai zeros([size(Ai,1) q-size(Ai,2) cbSz],'like',Ai)];
                % Concatenate input and neuron splits.
                As = [An_; Ai_];
                bs = [bn; bi];
                % Pad the neuron split indices with NaN for the input dimensions.
                newNrXi = [newNrXi; NaN(size(Ai,[1 3]),'like',newNrXi)];

                if options.nn.input_xor_neuron_splitting ...
                        && nNeur > 0 && nDims > 0
                    % We only allow input xor neuron splitting. Therefore,
                    % we select for each batch entry the favorable split.

                    % We select the minimum of split dimensions.
                    nDims_ = min(nNeur,nDims);

                    % Only pick either neuron or input split.
                    [~,splitIds] = sort([hn; hi],1,'descend');
    
                    % Obtain the indices for the relevant constraints.
                    sIdx = sub2ind(size(As,[1 3]), ...
                        splitIds(1:nDims_,:),repmat(1:cbSz,nDims_,1));
                    
                    % Extract the corresponding constraint; transpose the
                    % constraints and offsets for easier indexing.
                    As_ = permute(As,[2 1 3]);
                    As_ = reshape(As_(:,sIdx),[q nDims_ cbSz]);
                    bs_ = permute(bs,[2 1 3]);
                    bs_ = reshape(bs_(:,sIdx),[nSplits-1 nDims_ cbSz]);
                    % Re-transpose the constraints and offsets.
                    As = permute(As_,[2 1 3]);
                    bs = permute(bs_,[2 1 3]);
                    % Extract the corresponding neuron indices.
                    newNrXi = newNrXi(sIdx);

                    % Update number of new splits.
                    newSplits = nSplits^nDims_;
                end

                % Refine the input set based on the output specification.
                [li,ui,nrXis] = aux_refineInputSet(nn,options,storeInputs, ...
                    cxi,Gxi,yi,Gyi,A,b,numUnionConst,safeSet, ...
                        As,bs,newNrXi,nrXi,reluConstrHeuristic);
    
                % We enclose all unsafe outputs; therefore, a set is 
                % verified if it is empty. Identify empty sets.
                isVerified = any(isnan(li),1) | any(isnan(ui),1);
                % Remove the verified sets...
                li(:,isVerified) = [];
                ui(:,isVerified) = [];
                nrXis(:,isVerified) = [];
    
                % Compute center and radius of refined sets.
                xis = 1/2*(ui + li);
                ris = 1/2*(ui - li);

                % Identify which sets were refined to just being a point.
                isPoint = all(ris == 0,1);
                % Check the specification for the points.
                if any(~isPoint)
                    [~,critVal,falsified,x_,y_] = aux_checkPoints(nn,options, ...
                        idxLayer,A,b,safeSet,xis);
                    if any(falsified)
                        % Found a counterexample.
                        res.str = 'COUNTEREXAMPLE';
                        break;
                    else
                        % Remove the sets...
                        xis(:,isPoint) = [];
                        ris(:,isPoint) = [];
                        nrXis(:,isPoint) = [];
                        critVal(:,isPoint) = [];
                    end
                else
                    critVal = zeros([0 size(ris,2)],'like',ris);
                end

                % Check contained of the refined sets. TODO: we can not
                % just remove sets because they are contained; we have to
                % respect the neuron splitting and the stored bounds.
                % [isContained,nrXis] = aux_isContained(xis,ris,nrXis);
                % % Remove contained splits.
                % xis(:,isContained) = [];
                % ris(:,isContained) = [];
                % nrXis(:,isContained) = []; % TODO: We have to merge splits, otherwise we can have contradictions.
                % critVal(:,isContained) = [];
        
                % All removed subproblems are verified, i.e., empty set or
                % point sets.
                numVerified = numVerified + sum(isVerified) ...
                    + sum(isPoint); % + sum(isContained);
                % We have to subtract the number of dummy splits.
                numVerified = numVerified - sum(isDummySplit);
    
                if plotting && isfield(res,'uYs') && isfield(res,'uXs')
                    % Add a slack variable to convert between equality and 
                    % inequality constraints.
                    uYi = struct('c',yic,'G',Gyi,'r',yid, ...
                        'A',ld_Gyi,'b',b - ld_yi - ld_Gyi_err);
                    % Store constraint zonotope.
                    res.uYs{end+1} = aux_2ConZonoWithEqConst(uYi,0);
    
                    % Store input constraint zonotope.
                    res.uXs{end+1} = aux_2ConZonoWithEqConst(uXi,0);
                end
            otherwise
                % Invalid option.
                throw(CORAerror('CORA:wrongFieldValue', ...
                    'options.nn.refinement_method', ...
                        {'naive','zonotack','zonotack-layerwise'}));
        end
    
        % Order remaining sets by their criticality.
        [~,idx] = sort(critVal.*1./max(ris,[],1),'ascend');
        % Order sets.
        xis = xis(:,idx);
        ris = ris(:,idx);
        nrXis = nrXis(:,idx);

        % Add new splits to the queue.
        [xs,rs,nrXs] = aux_push(xis,ris,nrXis,xs,rs,nrXs,options);
    
        if plotting
            % Delete previously contained sets.
            if exist('hx_','var')
                cellfun(@(hxi_) delete(hxi_), hx_);
            end
            % Compute the bounds of the new sets.
            lis = xis - ris;
            uis = xis + ris;
            % Plot the unsafe output sets and the new input sets.
            [fig,huy,huy_,hux,hx,hx_] = ...
                aux_plotUnsafeOutputAndNewInputSets(fig,plotDims,res, ...
                    lis,uis,nSplits^nDims);
            drawnow;
        end
        if plotSplittingTree
            % Update the splitting tree.
            [res.tree,res.tlabels,tIdx] = ...
                aux_addNewSplitsToTheSplittingTree( ...
                    res.tree,res.tlabels,pIdx,dimIds,newNrXi,newSplits);

            if exist('isVerified','var') && exist('isPoint','var')
                % Compute the verified nodes.
                verifiedNodes = isVerified;
                verifiedNodes(~isVerified) = isPoint;
                % Update verified nodes in the splitting tree.
                [res.tlabels,tIdx] = aux_markVerifiedInTheSplittingTree( ...
                    res.tlabels,tIdx,verifiedNodes,'IR');
            end

            % Update the map of queue indices to tree indices.
            [res.qIdx2tIdx,~,~] = aux_push(tIdx,[],[],res.qIdx2tIdx,[],[],options);
        end
    
        % To save memory, we clear all variables that are no longer used.
        clear(batchVars{:});
        % Clear all batch variables in all layers.
        layers = aux_clearLayerFields(layers);
        
        % Increment iteration counter.
        iter = iter + 1;

    catch e
        if ismember(e.identifier,{'parallel:gpu:array:pmaxsize', ...
            'parallel:gpu:array:OOM','MATLAB:array:SizeLimitExceeded',...
            'MATLAB:nomem'}) % It is a memory error.
            if nReLU > 0
                % Reduce the number of ReLU constraints.
                if isinf(options.nn.num_relu_constraints)
                    % Reduce to 100 constraints in total.
                    nReLU = 100;
                else
                    % Reduce by factor 10.
                    nReLU = floor(1/100*nReLU);
                end
                fprintf('--- OOM error: reduce ReLU constraints %d...\n',nReLU);
                % Update the options.
                options.nn.num_relu_constraints = nReLU;
            elseif bSz > 1
                % Reduce the batch size.
                bSz = ceil(1/2*bSz);
                fprintf('--- OOM error: half batchSize %d...\n',bSz);
            elseif options.nn.train.num_approx_err > 0
                % Reduce the number of approximation errors.
                if isinf(options.nn.train.num_approx_err)
                    % Reduce the number of approximation errors to 100.
                    options.nn.train.num_approx_err = 1000;
                else
                    % Reduce by factor 10.
                    options.nn.train.num_approx_err = ...
                        floor(1/100*options.nn.train.num_approx_err);
                end
                fprintf('--- OOM error: reduce number of approximation errors %d...\n', ...
                    options.nn.train.num_approx_err);
                % We have to enable the interval center.
                options.nn.interval_center = true;
                % Recompute the indices for the approximation errors.
                q = nn.prepareForZonoBatchEval(x,options,idxLayer);
            elseif numInitGens > 1
                % Reduce the number of input generators by factor 2.
                numInitGens = ceil(1/2*numInitGens);
                fprintf('--- OOM error: reduce number of input generators %d...\n', ...
                    numInitGens);
                % Update the options.
                options.nn.train.num_init_gens = numInitGens;
                % We have to enable the interval center.
                options.nn.interval_center = true;
                % Recompute the indices for the approximation errors.
                q = nn.prepareForZonoBatchEval(x,options,idxLayer);
            else
                % We cannot adapt any parameters.
                fprintf('--- OOM error: no parameters to adapt!\n');
                rethrow(e);
            end
            % Append the current batch items.
            [xs,rs,nrXs] = aux_push(xi,ri,nrXi,xs,rs,nrXs,options);

            % Clear the batch variables.
            clear(batchVars{:});
            % Clear all batch variables in all layers.
            layers = aux_clearLayerFields(layers);
            % Clear the allocated generator matrix.
            % clear('batchG');
            % Re-allocate generators for initial perturbation set.
            % batchG = zeros([n0 q bSz],'like',inputDataClass);
        else
            fprintf('unexpected Error --- neuralNetwork/verify...\n');
            rethrow(e);
        end
    end
end

if size(xs,2) == 0 && ~strcmp(res.str,'COUNTEREXAMPLE')
    % Verified all patches.
    res.str = 'VERIFIED';
    x_ = [];
    y_ = [];
end

% Store time.
res.time = toc(timerVal);
% Store number of verified patches.
res.numVerified = gather(numVerified);

if verbose
    % Print the final stats.
    aux_printIterationStats(table,iter,xs,r,rs,numVerified)
    % Print table footer.
    table.printFooter();
    % Print the result.
    fprintf('--- Result: %s (time: %.3f [s])\n',res.str,res.time);
end

if plotSplittingTree
    % Plot the splitting tree.
    aux_plotSplittingTree(res.tree,res.tlabels);
end

end


% Auxiliary functions -----------------------------------------------------

function aux_printIterationStats(table,iter,xs,r,rs,numVerified)
    % Compute the length of the queue.
    queueLen = size(xs,2);
    % Compute the other stats.
    if ~isempty(rs)
        % Compute the average radius.
        avgRad = mean(rs,'all');
        % Compute the ratio of the unknown input volume (assume intervals).
        unknVol = sum(prod(2*rs,1),'all')/sum(prod(2*r),'all')*100;
        % Compute the ratio of f-radii to approximate the volume of unknown
        % input space.
        % unknVol = sum(sqrt(sum(rs.^2,1)),'all')...
        %         /sum(sqrt(sum(r.^2,1)),'all')*100;
    else
        % The queue is empty; there are no stats.
        avgRad = 0;
        unknVol = 0;
    end
    % Print new table row.
    table.printContentRow({iter,queueLen,numVerified,avgRad,unknVol});
end

function [xi,ri,nrXi,xs,rs,nrXs,qIdx] = aux_pop(xs,rs,nrXs,bSz,options)
    % Obtain the number of elements in the queue.
    nQueue = size(xs,2);
    % Construct indices to pop.
    switch options.nn.verify_dequeue_type
        case 'front'
            % Take the first entries.
            qIdx = 1:min(bSz,nQueue);
        case 'half-half'
            % Half from the front and half from the back.
            qIdx = 1:min(bSz,nQueue);
            offsetIdx = ceil(length(qIdx)/2 + 1):length(qIdx);
            qIdx(offsetIdx) = qIdx(offsetIdx) + nQueue - length(qIdx);
        otherwise
            % Invalid option.
            throw(CORAerror('CORA:wrongFieldValue', ...
                'options.nn.verify_enqueue_type',{'append','prepend'}));
    end
    % Pop centers.
    xi = xs(:,qIdx);
    xs(:,qIdx) = [];
    % Pop radii.
    ri = rs(:,qIdx);
    rs(:,qIdx) = [];
    % Pop indices for split neurons.
    nrXi = nrXs(:,qIdx);
    nrXs(:,qIdx) = [];
end

function [xs,rs,nrXs] = aux_push(xi,ri,nrXi,xs,rs,nrXs,options)
    % Try to remove irrelevant rows from the stored indices.
    nrXi = sort(nrXi,1,'descend'); % Move the NaN to the bottom.
    minNumNan = min(sum(isnan(nrXi),1)); % Identify the minimum number of NaN.
    nrXi = nrXi((minNumNan+1):end,:); % Trim the minimum number of NaN.

    % Retrieve variables from the GPU.
    xi = gather(xi);
    ri = gather(ri);

    % Pad the indices of the split neurons.
    if size(nrXi,1) > size(nrXs,1)
        nrXs = [nrXs; NaN([size(nrXi,1)-size(nrXs,1) size(nrXs,2)])];
    else
        nrXi = [nrXi; NaN([size(nrXs,1)-size(nrXi,1) size(nrXi,2)])];
    end
    % Retrieve the indices from the GPU.
    nrXi = gather(nrXi);
    % Add new splits to the queue.
    switch options.nn.verify_enqueue_type
        case 'append'
            % Append the new entries.
            xs = [xs xi];
            rs = [rs ri];
            nrXs = [nrXs nrXi];
        case 'prepend'
            % Prepend the new entries.
            xs = [xi xs];
            rs = [ri rs];
            nrXs = [nrXi nrXs];
        otherwise
            % Invalid option.
            throw(CORAerror('CORA:wrongFieldValue', ...
                'options.nn.verify_enqueue_type',{'append','prepend'}));
    end
end

function layers = aux_clearLayerFields(layers)
    % Remove all stored auxiliary computations in the layers.
    for i=1:length(layers)
        % Obtain the i-th layer.
        layeri = layers{i};
        % Clear all fields.
        layeri.backprop.store = ...
            rmfield(layeri.backprop.store,fields(layeri.backprop.store));
    end
end

function [critValPerConstr,critVal,falsified,x_,y_] = ...
    aux_checkPoints(nn,options,idxLayer,A,b,safeSet,xs)
    % Compute the output of the adversarial examples.
    ys = nn.evaluate_(xs,options,idxLayer);
    % Compute the logit difference.
    ld_ys = A*ys;
    % Check the specification and compute a value indicating how close we 
    % are to finding an adversarial example (< 0 mean the specification is 
    % violated).
    critValPerConstr = ld_ys - b;
    if safeSet
        % safe iff all(A*y <= b) <--> unsafe iff any(A*y > b)
        % Thus, unsafe if any(-A*y < -b).
        falsified = any(ld_ys > b,1);
        critValPerConstr = -critValPerConstr;
        critVal = min(critValPerConstr,[],1);
    else
        % unsafe iff all(A*y <= b) <--> safe iff any(A*y > b)
        % Thus, unsafe if all(A*y <= b).
        falsified = all(ld_ys <= b,1);
        critVal = max(critValPerConstr,[],1);
    end

    if any(falsified)
        % Found a counterexample.
        idNzEntry = find(falsified);
        id = idNzEntry(1);
        x_ = gather(xs(:,id));
        % Gathering weights from gpu. There is are precision error when 
        % using single gpuArray.
        nn.castWeights(single(1));
        y_ = nn.evaluate_(x_,options,idxLayer); % yi_(:,id);
    else
        % We have not found a counterexample.
        x_ = [];
        y_ = [];
    end
end

function [isContained,nrXis] = aux_isContained(xs,rs,nrXis)
    % Specify a tolerance.
    tol = 1e-6;

    % Obtain the number of dimensions and batch size.
    [~,bSz] = size(xs);

    % Compute the bounds of the sets.
    ls = permute(xs - rs,[1 2 4 3]);
    us = permute(xs + rs,[1 2 4 3]);

    % We sort the items to prevent a previous batch item to be contained
    % in a previous one.
    [~,idx] = sortrows([ls; -us]');
    ls = ls(:,idx);
    us = us(:,idx);
    % Also sort the neuron indices for the splits.
    nrXis_ = nrXis(:,idx);

    % Compute a pairwise containment matrix.
    pairwiseContainment = ...
        all(permute(ls-tol,[2 3 1]) <= permute(ls,[3 2 1]),3) ...  % lower bounds are larger
        & all(permute(us,[3 2 1]) <= permute(us+tol,[2 3 1]),3); % upper bounds are smaller
    % Specify which indices can be removed, to ensure that only the first
    % enclosing set is kept.
    canBeRemoved = (1:bSz)' < (1:bSz);

    % Identify sets that can be removed because they are contained in
    % another set.
    isContained_ = reshape(any(pairwiseContainment & canBeRemoved,1),[1 bSz]);

    % Identify the batch entries that are kept.
    areKept = find(~isContained_(1:bSz));
    for i=areKept
        % Identify which sets are contained within the i-th set. We
        % have to merge those.
        areEnclosed = pairwiseContainment(i,:) & canBeRemoved(i,:);

        % Identify contradiction neuron splits.
        areContra = any(nrXis_(:,i) ...
            == permute(-nrXis_(:,areEnclosed),[3 1 2]),2:3);

        % Invalidate these splits by adding 0.5; thereby, the 
        % splits are no longer applied but we remember that this 
        % split is not good.
        nrXis_(areContra,i) = nrXis_(areContra,i) + 0.5;
    end

    % Reorder.
    revIdx(idx) = 1:bSz;
    isContained = isContained_(revIdx);
    nrXis = nrXis_(:,revIdx);
end

function [cxi,Gxi,dimIdx] = aux_constructInputZonotope(options,heuristic, ...
    xi,ri,sens,grad,numInitGens)
    % Obtain the number of input dimensions and the batch size.
    [n0,bSz] = size(xi);

    % Initialize the generator matrix.
    Gxi = zeros([n0 numInitGens bSz],'like',xi);

    if numInitGens >= n0
        % We create a generator for each input dimension.
        dimIdx = repmat((1:n0)',1,bSz);
    else
        % Compute the heuristic.
        hi = aux_computeHeuristic(heuristic,0,xi-ri,xi+ri,ri,sens,grad, ...
            [],[],[],false,1);

        % Find the input pixels that affect the output the most.
        [~,dimIdx] = sort(hi,'descend');
        % Select the most important input dimensions and add a generator
        % for each of them.
        dimIdx = dimIdx(1:numInitGens,:);
    end
    % Compute indices for non-zero entries.
    gIdx = sub2ind(size(Gxi),dimIdx, ...
        repmat((1:numInitGens)',1,bSz),repelem(1:bSz,numInitGens,1));
    % Set non-zero generator entries.
    Gxi(gIdx) = ri(sub2ind(size(ri),dimIdx,repelem(1:bSz,numInitGens,1)));
    % Sum generators to compute remaining set.
    ri_ = (ri - reshape(sum(Gxi,2),[n0 bSz]));

    % Construct the center.
    if options.nn.interval_center
        % Put remaining set into the interval center.
        cxi = permute(cat(3,xi - ri_,xi + ri_),[1 3 2]);
    else
        % The center is just a vector.
        cxi = xi;
    end
end

function [ld_yi,ld_Gyi,ld_Gyi_err,yic,yid,Gyi] = ...
    aux_computeLogitDifference(yi,Gyi,A,options)
    % Obtain number of output dimensions and batch size.
    [nK,~,bSz] = size(Gyi);

    if options.nn.interval_center
        % Compute the center and the radius of the center-interval.
        yic = reshape(1/2*(yi(:,2,:) + yi(:,1,:)),[nK bSz]);
        % Compute approximation error.
        yid = 1/2*(yi(:,2,:) - yi(:,1,:));
    else
        % The center is just a vector.
        yic = yi;
        % There are no approximation errors stored in the center.
        yid = zeros([nK 1 bSz],'like',yi);
    end

    % Compute the logit difference of the input generators.
    ld_yi = A*yic;
    ld_Gyi = pagemtimes(A,Gyi);
    % Compute logit difference of the approximation errors.
    ld_Gyi_err = sum(abs(A.*permute(yid,[2 1 3])),2);
end

function [c,G] = aux_matchBatchSize(c,G,bSz,options)
    % Replicate a zonotope batch for splitting.
    if bSz ~= size(G,3) % iff newSplits > 1
        newSplits = bSz/size(G,3);
        if options.nn.interval_center
            c = repelem(c,1,1,newSplits);
        else
            c = repelem(c,1,newSplits);
        end
        G = repelem(G,1,1,newSplits);
    end
end

function [l,u,c,cr,G,cl_,cu_,G_] = aux_computeBoundsZonotope(c,G,options, ...
    A,b,numUnionConst,bc,br)
    % Compute the bounds of a batch of zonotopes. We return the bounds of
    % the refined set, the unrefined (only duplicated set), as well as the
    % refined set.

    if ~isempty(bc)
        % Obtain the batch size.
        [~,bSz] = size(bc);
        % Replicate sets to match the batch size.
        [c,G] = aux_matchBatchSize(c,G,bSz,options);

        % Scale and offset the input set.
        [c_,G_] = aux_scaleAndOffsetZonotope(c,G,bc,br);
    else
        % The input set is not scaled.
        c_ = c;
        G_ = G;
    end
    
    % Obtain number of hidden neurons.
    [nk,~,bSz] = size(G_);
    
    % Compute the radius of the zonotope.
    r_ = reshape(sum(abs(G_),2),[nk bSz]);
    if options.nn.interval_center
        % Compute center and center radius.
        cl_ = reshape(c_(:,1,:),[nk bSz]);
        cu_ = reshape(c_(:,2,:),[nk bSz]);
        % Compute the un-scaled center.
        cr = reshape(1/2*(c(:,2,:) - c(:,1,:)),[nk bSz]);
        c = reshape(1/2*(c(:,2,:) + c(:,1,:)),[nk bSz]);
    else
        % The radius is zero.
        cl_ = c_;
        cu_ = c_;
        cr = 0;
    end
    % Compute the bounds.
    l = cl_ - r_;
    u = cu_ + r_;

    if ~isempty(A) && ~isempty(b) && options.nn.conzonotope_bound_max_iter > 0
        % Approximate the bounds of the constrained zonotope.
        options.nn.conzonotope_bounding_method = 'dual-iter';
        cZs = struct('c',c,'G',G,'dr',cr,'A',A,'b',b);
        [l_,u_,~,~] = conZonotope.approximateBoundsWithGPU(cZs, ...
            numUnionConst,options);
        % Take the tighter bounds.
        l = max(l,l_);
        l(isnan(l) | isnan(l_)) = NaN; % Keep nan values.
        u = min(u,u_);
        l(isnan(l) | isnan(l_)) = NaN; % Keep nan values.
    end
end

function [l,u] = aux_obtainBoundsFromSplits(neuronIds,bSz,constNrIdx,d,c)
    % Extract the bounds from constraints. The argument d represents the
    % offset of a split constraint; together with the center c we can
    % recompute the split-bound.

    % Initialize the bounds.
    l = -inf([length(neuronIds) bSz],'like',d);
    u = inf([length(neuronIds) bSz],'like',d);

    % TODO: implement handling of multiple splits per dimension. Use
    % aggregation to get the best bounds for each dimension.
    if ~isempty(constNrIdx) && any(~isnan(constNrIdx),'all')
        % Check which constraints contain bounds for the current layer.
        isBnd = (permute(abs(constNrIdx),[3 2 1]) ...
            == repmat(neuronIds',1,bSz));
        if ~any(isBnd,'all')
            % There are no bounds from splits.
            return;
        end
        % Check which constraints contain lower or upper bounds.
        isLBnd = (permute(sign(constNrIdx),[3 2 1]) == -1);
        isUBnd = (permute(sign(constNrIdx),[3 2 1]) == 1);
        % Compute the indices into the bounds.
        ljIdx = any(isBnd & isLBnd,3);
        ujIdx = any(isBnd & isUBnd,3);

        if ~isempty(c) && ~isempty(d)
            % Compute indices into the constraints.
            cstrLjIdx = permute(any(isBnd & isLBnd,1),[3 2 1]);
            cstrUjIdx = permute(any(isBnd & isUBnd,1),[3 2 1]);
            % Extract the bounds based on the splits.
            bndLj = reshape(d(cstrLjIdx),[],1) - c(ljIdx);
            bndUj = reshape(d(cstrUjIdx),[],1) + c(ujIdx);
        else
            % The bounds are 0, because we split a ReLU at 0.
            bndLj = 0;
            bndUj = 0;
        end
        % Update bounds based on the splits.
        if any(ljIdx,'all')
            l(ljIdx) = bndLj;
        end
        if any(ujIdx,'all')
            u(ujIdx) = bndUj;
        end
    end
end

function [l,u,cl_,cu_,G_] = aux_computeBoundsOfInputSet(layer,options, ...
    A,b,numUnionConst,bc,br,constNrIdx,d,prevNrXs)
    % For a given layer, we compute the bounds w.r.t. the current
    % hypercube. We return the bounds and the scaled input set.

    % Obtain the batch size.
    bSz = max(size(constNrIdx,2),size(prevNrXs,2));

    % Obtain the number of input generators.
    qIds = layer.genIds;
    if isfield(layer.backprop.store,'inc') ...
            && isfield(layer.backprop.store,'inG')
        % Obtain the input set.
        c = layer.backprop.store.inc;    
        G = layer.backprop.store.inG(:,qIds,:);
        
        % (i) Compute bound based on the refinement.
        [l,u,c,~,~,cl_,cu_,G_] = aux_computeBoundsZonotope(c,G,options, ...
            A,b,numUnionConst,bc,br);
    else
        % There is no stored input set. Initialize the bounds.
        l = -inf;
        u = inf;
        % The outputs can not be set.
        cl_ = [];
        cu_ = [];
        G_ = [];
    end

    % (ii) Use bounds based on splitting.

    % Obtain the indices of the neurons of the current layer.
    neuronIds = layer.neuronIds;

    if any(~isnan(constNrIdx),'all')
        % Compute bounds based on current splits.
        [l_,u_] = aux_obtainBoundsFromSplits(neuronIds,bSz,constNrIdx,d,c);
        % Update the bounds
        l = max(l,l_);
        u = min(u,u_);
    end

    if ~isempty(prevNrXs) && options.nn.num_splits == 2 ...
            && strcmp(options.nn.split_position,'zero')
        % Apply bounds from previous splits (TODO: for now only ReLU splits 
        % at 0; future work, remember arbitrary splits).
        [l_,u_] = aux_obtainBoundsFromSplits(neuronIds,bSz,prevNrXs,[],[]);
        % Update the bounds
        l = max(l,l_);
        u = min(u,u_);
    end

    % Identify NaN (indicate empty sets; we cannot remove them due 
    % to the fixed batch size).
    isEmpty = isnan(l) | isnan(u);

    % Remove empty sets.
    l(isEmpty) = NaN;
    u(isEmpty) = NaN;
end

function imgM = aux_patchWeightMask(imgSize,ph,pw,l,u)
    % Generate a patch-wise weight mask for input splitting, where the 
    % center of a patch gets a higher score than the remaining pixels of 
    % the patch.

    % Create the weight mask for a patch; initialize with the low score.
    pM = l*ones([ph pw]);
    % Set the center pixel to the high score.
    pM(ceil(ph/2),ceil(pw/2)) = u;
    % Tile mask over input.
    imgM = kron(ones(ceil(imgSize(1:2)./[ph pw])),pM);
    % Trim excess dimensions.
    imgM = imgM(1:imgSize(1),1:imgSize(2));

    if length(imgSize) == 3
        % There is a color channel.
        imgM = repmat(imgM,1,1,imgSize(3));
    end
end

% Heuristics --------------------------------------------------------------

function grad = aux_updateGradients(nn,options,idxLayer,Yi,A,b,storeGradients)
    % Update the gradient of the f-radius store in the layers of the neural
    % network. The gradients are used to optimize the approximation slope
    % as well as for splitting heuristics.

    % Store the gradients of the approximation errors.
    options.nn.store_approx_error_gradients = storeGradients;

    if isa(Yi,'interval')
        % We compute the gradient of an interval.
        % Obtain the bounds.
        yli = Yi.inf;
        yui = Yi.sup;

        % Compute the gradient of the f-radius.
        frad = sqrt(sum((yui - yli).^2,1));
        gyli = -yli./(frad + 1e-6);
        gyui = -yui./(frad + 1e-6);
        % Backpropagate the gradients to identify which input dimensions
        % create the largest interval approximation error.
        % For now, we do not store the gradient, because we only use the
        % gradient for the construction of the input zonotope.
        [~,grad] = nn.backpropIntervalBatch(gyli,gyui,options,idxLayer,false);
    else
        % We compute the gradient of a zonotope.
        Gyi = Yi;
        % Obtain the number of output dimensions and batch size.
        [nK,~,bSz] = size(Gyi);

        % ld_Gyi = pagemtimes(A,Gyi); 
   
        % Compute the gradient of the f-radius.
        frad = sqrt(sum(Gyi.^2,[1 2]));
        gGyi = Gyi./(frad + 1e-6);
        % gGyi = pagemtimes(A',gGyi);

        % Compute a dummy center gradient.
        if options.nn.interval_center
            gcyi = zeros([nK 2 bSz],'like',Gyi);
        else
            gcyi = zeros([nK bSz],'like',Gyi);
        end

        % Compute gradient of the f-radius of the output set; the gradient 
        % is used to split the neuron in the network as well as input 
        % dimensions.
        [~,grad] = nn.backpropZonotopeBatch_(gcyi,gGyi,options,idxLayer,false);
    end
end

function h = aux_computeHeuristic(heuristic,layerIdx,l,u,dr,sens,grad, ...
    varargin)
    % Set default parameters.
    [sim,prevNrXs,neuronIds,onlyUnstable,layerDiscount,...
        imgSz,patchSz,patchScore] = ...
        setDefaultValues({[],[],[],true,1,[],[],[]}, varargin);

    % Compute the heuristic.
    switch heuristic
        case 'least-unstable'
            % Least unstable neuron (normalize the un-stability).
            minBnd = 1./min(-l,u);
            % Compute the heuristic.
            h = minBnd.*sens;
        case 'least-unstable-gradient'
            % Take the absolute value and add small epsilon to avoid 
            % numerical problems.
            grad = abs(grad) + 1e-3;
            % Least unstable neuron (normalize the un-stability).
            minBnd = 1./min(-l,u);
            % Compute the heuristic.
            h = minBnd.*grad;
        case 'approx-error'
            % Compute the heuristic.
            h = dr;
        case 'most-sensitive-approx-error'
            % Compute the heuristic.
            h = dr.*sens;
        case 'input-radius'
            % Compute the radius as the heuristic.
            h = 1/2*(u - l);
        case 'most-sensitive-input-radius'
            % Compute the radius.
            r = 1/2*(u - l);
            % Compute the heuristic.
            h = r.*sens;
        case 'zono-norm-gradient'
            % Take the absolute value and add small epsilon to avoid 
            % numerical problems.
            grad = abs(grad) + 1e-3;
            % Normalize the gradient across the batch.
            % grad = (grad - min(grad,[],'all'))...
            %     ./(max(grad,[],'all') - min(grad,[],'all') + 1e-6);
            % Compute the radius.
            r = dr; % 1/2*(u - l);
            % Normalize the radius.
            % r = (r - min(r,[],'all'))...
            %     ./(max(r,[],'all') - min(r,[],'all') + 1e-6);
            % Compute the heuristic.
            h = grad.*r;
        otherwise
            % Invalid option.
            throw(CORAerror('CORA:wrongFieldValue','heuristic', ...
                {'least-unstable','most-sensitive-approx-error', ...
                'most-sensitive-input-radius','zono-norm-gradient'}));
    end

    if onlyUnstable
        % Flag unstable neurons.
        unstable = (l < 0 & 0 < u);
        % Only consider unstable neurons. 
        h(~unstable) = -inf;
    end

    if layerDiscount ~= 1
        % Prefer earlier layers.
        h = h.*(layerDiscount^layerIdx);
    end

    if ~isempty(prevNrXs)
        % Obtain the batch size.
        [~,bSz] = size(prevNrXs);

        % We floor all entries. We mark unnecessary splits with decimal
        % numbers, thereby the split is not applied.
        prevNrXs = floor(prevNrXs);
        % Reduce redundancy by not add constraints for split neurons.
        if size(h,2) > size(prevNrXs,2)
            newSplits = size(h,2)/size(prevNrXs,2);
            prevNrXs_ = repelem(prevNrXs,1,newSplits);
        else
            prevNrXs_ = prevNrXs;
        end

        % Identify already split neurons.
        wasSplit = any(permute(abs(prevNrXs_),[3 2 1]) ...
            == repmat(neuronIds,bSz,1)',3);
        % There is no similarity; just prevent splitting the same
        % neuron twice by setting the heuristic to -inf.
        h(wasSplit) = -inf;
        if ~isempty(sim)
            % Specify a tolerance for similarity.
            tol = 1e-3;
            % Reduce the heuristic based on the similarity to already split
            % neurons.
            simSplit = any((sim > 1-tol) & permute(wasSplit,[1 3 2]),2);
            h(simSplit(:,:)) = -inf;
        else
            % There is no similarity; just prevent splitting the same
            % neuron twice by setting the heuristic to -inf.
            h(wasSplit) = -inf;
        end
    end

    if ~isempty(imgSz)
        % Obtain the patch size.
        ph = patchSz(1);
        pw = patchSz(2);
        % Obtain the scores.
        lowScore = patchScore(1);
        highScore = patchScore(2);
        % Compute a patch-wise weight mask to avoid splitting
        % many similar input dimensions.
        imgM = aux_patchWeightMask(imgSz,ph,pw,lowScore,highScore);
        % Convert to the correct data type.
        imgM = cast(imgM,'like',h);
        % Weight the heuristic by the patch-wise mask.
        h = h.*imgM(:);
    end
end

% Set Refinement ----------------------------------------------------------

function [l,u,nrXis] = aux_refineInputSet(nn,options, ...
    storeInputs,x,Gx,y,Gy,A,b,numUnionConst,safeSet, ...
    As,bs,newNrXs,prevNrXs,reluConstrHeuristic)

    % Specify improvement tolerance.
    improveTol = 1e-3;

    % Specify the minimum and maximum number of refinement iterations per 
    % layer.
    minRefIter = options.nn.refinement_min_iter;
    maxRefIter = options.nn.refinement_max_iter;

    % Extract type of refinement.
    layerwise = strcmp(options.nn.refinement_method,'zonotack-layerwise');

    % Specify whether approximation slopes are gradient optimized.
    gradientSlopeOptimization = ...
        strcmp(options.nn.input_split_heuristic,'zono-norm-gradient') ...
    || (options.nn.num_neuron_splits > 0 ...
        && any(strcmp(options.nn.neuron_split_heuristic, ...
                {'zono-norm-gradient','least-unstable-gradient'}))) ...
    || (options.nn.num_relu_constraints > 0 ...
        && any(strcmp(options.nn.relu_constraint_heuristic, ...
                {'zono-norm-gradient','least-unstable-gradient'})));

    % Enumerate the layers of the neural networks.
    [layers,ancIdx] = nn.enumerateLayers();

    % Obtain the indices of the activation layers.
    actIdxLayer = (1:length(layers));
    actIdxLayer = actIdxLayer(arrayfun(@(i) ...
        isa(layers{i},'nnActivationLayer'),actIdxLayer));

    if layerwise
        % TODO: we cannot refine through composite layer, because we loose
        % dependencies between the computation paths. We recompute
        % the output from the refined layer; we cannot recompute from the
        % input set, there we need to compensate for the approximation
        % errors.

        % We can only refine the top-level computation path (there are no
        % parallel paths).
        % layers = nn.layers;

        % We refine all activation layers.
        refIdxLayer = [1 actIdxLayer];
        % refIdxLayer = (1:length(layers));

        % Flip the layers for a backward refinement.
        refIdxLayer = fliplr(refIdxLayer);
    else       
        % We only refine the input.
        refIdxLayer = 1;
    end

    % Obtain number of generators and batchsize.
    [nK,q,bSz] = size(Gy);

    % Convert and join the general- & input-split constraints.
    [C,d,newSplits,constNrIdx] = aux_convertSplitConstraints(As,bs,newNrXs);

    % Replicate set for split constraints.
    [x,Gx] = aux_matchBatchSize(x,Gx,bSz*newSplits,options);
    [y,Gy] = aux_matchBatchSize(y,Gy,bSz*newSplits,options);
    % Update the batch size.
    bSz = bSz*newSplits;
    
    if ~isempty(constNrIdx) && size(bs,2) == 1
        % Compute the indices of the split neurons.
        nrXis = constNrIdx;
        % Remove all indices that do not correspond to the split of a neuron.
        nrXis(all(isnan(nrXis),2),:) = [];
        nrXis = reshape(nrXis,[],bSz);
        % Duplicate the indices for previously split neurons.
        prevNrXs = repelem(prevNrXs,1,newSplits);
        % Combine the newly split neurons with the previously split ones.
        nrXis = [prevNrXs; nrXis];
    else
        % Currently, we can only remember splits into two pieces.
        nrXis = zeros([0 bSz]);
    end

    % Initialize scale and offset of the generators.
    bc = zeros([q bSz],'like',Gy);
    br = ones([q bSz],'like',Gy);

    % Initialize loop variables.
    refIdx = 1; % index into refIdxLayer
    refIter = 1; % Counter for number of refinement iterations of the 
    % current layer.

    % Keep track of empty sets.
    isEmpty = zeros([1 bSz],'logical');

    % Keep track of which inputs sets of which layers need scaling.
    scaleInputSets = ones([1 length(layers)],'logical');

    % Iterate layers in a backward fashion to propagate the constraints
    % through the layers.
    while refIdx <= length(refIdxLayer)
        % Obtain layer index.
        i = refIdxLayer(refIdx);

        % Append the index of the current layer to update its input set 
        % in the next iterations.
        idxLayer = ancIdx(i):length(nn.layers);
   
        % Construct the unsafe output set.
        uYi = aux_constructUnsafeOutputSet(options,y,Gy,A,b,safeSet,numUnionConst);

        % Scale and offset constraints with current hypercube.
        [d_,C_] = aux_scaleAndOffsetZonotope(d,C,-bc,br);

        if ~isempty(C_)
            % Append split constraints.
            uYi.A = [uYi.A; C_];
            % Append the offset.
            uYi.b = [uYi.b; d_]; 
        end

        if options.nn.num_relu_constraints > 0
            % Compute tightening constraints for unstable ReLU neurons.
            [At,bt,~] = aux_reluTriangleConstraints(nn,options, ...
                [],reluConstrHeuristic,bc,br,scaleInputSets,nrXis);

            % Append ReLU constraints.
            uYi.A = [uYi.A; At];
            % Append the offset.
            uYi.b = [uYi.b; bt];
        end

        % Compute the bounds of the unsafe inputs (hypercube).
        options.nn.conzonotope_bounding_method = 'fourier-motzkin';
        [ly,uy,bli,bui] = conZonotope.approximateBoundsWithGPU(uYi, ...
            numUnionConst,options);

        % Update empty sets.
        isEmpty = isEmpty | any(isnan(ly),1) | any(isnan(uy),1);

        % Compute the center and radius of the new inner hypercube 
        % (the new hypercube is relative to the current hypercube).
        bci = 1/2*(bui + bli);
        bri = 1/2*(bui - bli);

        % Update the hypercube.
        bc = bc + br.*bci;
        br = br.*bri;

        % We have to the refine the input set of an ancestor layer.
        % Obtain the input set of the ancestor of the i.th layer.
        layerAnc = nn.layers{ancIdx(i)};
    
        % Obtain number of input generators of the i-th layer.
        ancQiIds = layerAnc.genIds;
        if layerwise
            % We refine the input set of the ancestor layer.
            cAnc = layerAnc.backprop.store.inc;    
            GAnc = layerAnc.backprop.store.inG(:,ancQiIds,:);
            % Replicate sets to match the batch size.
            [cAnc,GAnc] = aux_matchBatchSize(cAnc,GAnc,bSz,options);
        else
            % We only refine the input set.
            cAnc = x;    
            GAnc = Gx(:,ancQiIds,:);
        end

        % Update scale and offset of the input set to compute a smaller 
        % output set.
        if scaleInputSets(i) || ~layerwise
            [cAnc,GAnc] = aux_scaleAndOffsetZonotope(cAnc,GAnc,bc,br);
        else
            [cAnc,GAnc] = aux_scaleAndOffsetZonotope(cAnc,GAnc,bci,bri);
        end

        % Store computed bounds in the layers for tighter approximations.
        bndIdxLayer = actIdxLayer(arrayfun(@(j) ...
            ancIdx(i) <= ancIdx(j),actIdxLayer));
        for j=bndIdxLayer
            % Obtain the j-th layer.
            layerj = layers{j};
            % Obtain the constraint matrices.
            Aj = uYi.A;
            bj = uYi.b;

            % Compute the bounds of the j-th layer.
            if scaleInputSets(j)
                % The factors of the stored input set are w.r.t. to the
                % initial hypercube. Therefore, we have to scale the
                % constraints.
                brj = br./bri;
                bcj = bc - brj.*bci;
                [bj,Aj] = aux_scaleAndOffsetZonotope(bj,Aj,bcj,1./brj);
                % Compute the bounds of the current input set.
                [lj,uj,~,~,~] = aux_computeBoundsOfInputSet(layerj, ...
                    options,Aj,bj,numUnionConst,bc,br,constNrIdx,d,prevNrXs);
            else
                % The factors of the stored input set are w.r.t. to the
                % current hypercube.
                [lj,uj,~,~,~] = aux_computeBoundsOfInputSet(layerj, ...
                    options,Aj,bj,numUnionConst,bci,bri,constNrIdx,d,prevNrXs);
            end

            % Store the computed bounds in the layers.
            layerj.backprop.store.l = lj;
            layerj.backprop.store.u = uj;
        end

        % Store inputs for each layer by enabling backpropagation. 
        options.nn.train.backprop = storeInputs;
        % Compute a new output enclosure.
        [y,Gy] = nn.evaluateZonotopeBatch_(cAnc,GAnc,options,idxLayer);
        if gradientSlopeOptimization
            % Update the gradients to optimize slope.
            aux_updateGradients(nn,options,idxLayer,Gy,A,b,true);
        end
        options.nn.train.backprop = false;

        if storeInputs
            % New input sets are computed for the layers; update scaling 
            % index.
            scaleInputSets(ancIdx >= ancIdx(i)) = false;
        end

        % Check if we can further refine the current layer.
        if refIter <= minRefIter || (refIter < maxRefIter && ... 
                any(~isEmpty & min(bri,[],1) < 1 - improveTol,'all'))
            % Do another refinement iteration on the current layer.
            refIter = refIter + 1;
        else
            % No more refinement possible (either maximum number of
            % iteration reached or last iteration did not further refine
            % the hypercube).
            refIdx = refIdx + 1;
            % Reset refinement iteration counter.
            refIter = 1;
        end
    end

    % Remove the stored bounds.
    for i=actIdxLayer
        % Obtain the i-th layer.
        layeri = layers{i};
        % Clear the stored bounds and gradients.
        layeri.backprop.store.slope_gradients = 0;
        layeri.backprop.store = rmfield(layeri.backprop.store, ...
            {'l','u','slope_gradients'});
        if isfield(layeri.backprop.store,'dm')
            layeri.backprop.store = rmfield(layeri.backprop.store,'dm');
        end
    end

    % The factors of the stored input set are w.r.t. to the
    % initial hypercube. Therefore, we have to scale the
    % constraints.
    brx = br./bri;
    bcx = bc - brj.*bci;
    [bx,Ax] = aux_scaleAndOffsetZonotope(uYi.b,uYi.A,bcx,1./brx);
    % Compute bounds of the refined input set.
    [l,u,~,~,~,~,~,~] = aux_computeBoundsZonotope(x,Gx,options, ...
        Ax,bx,numUnionConst,bc,br);

    % Update empty sets.
    isEmpty = isEmpty | any(l > u,1);

    % Update bounds to represent empty sets.
    l(:,isEmpty) = NaN;
    u(:,isEmpty) = NaN;
end

function uYi = aux_constructUnsafeOutputSet(options,y,Gy,A,b,safeSet, ...
    numUnionConstraint)
    % Obtain the number of output dimensions and batch size.
    [nK,~,bSz] = size(Gy);

    if options.nn.interval_center 
        % Compute center and center radius.
        yc = reshape(1/2*(y(:,2,:) + y(:,1,:)),[nK bSz]);
        yr = reshape(1/2*(y(:,2,:) - y(:,1,:)),[nK bSz]);
    else
        % The radius is zero.
        yc = y;
        yr = 0;
    end

    % Compute the output constraints.
    [ld_yi,ld_Gyi,ld_Gyi_err,~,~,~] = ... 
        aux_computeLogitDifference(y,Gy,A,options);
    % Compute output constraints.
    if safeSet
        % safe iff all(A*y <= b) ...
        % <--> unsafe iff any(A*y > b) <--> unsafe iff any(-A*y < -b)
        % Thus, unsafe if any(-A*Gy*\beta < -b + A*y)
        A_ = -ld_Gyi;
        b_ = ld_yi - b;
        % Invert the sign for the union constraints.
        A_((numUnionConstraint+1):end,:,:) = -A_((numUnionConstraint+1):end,:,:);
        b_((numUnionConstraint+1):end,:) = -b_((numUnionConstraint+1):end,:);
    else
        % unsafe iff all(A*y <= b)
        % Thus, unsafe if all(A*Gy*\beta <= b - A*y)
        A_ = ld_Gyi;
        b_ = b - ld_yi;
    end

    % Construct a struct for the output set.
    uYi = struct('c',yc,'dr',yr,'G',Gy); 
    % Apply the output constraints to the input set of the i-th layer.
    uYi.A = A_;
    % Offset by refinement errors.
    uYi.b = b_ + ld_Gyi_err(:,:);
end

function [c,G] = aux_scaleAndOffsetZonotope(c,G,bc,br)
    % Obtain indices of generator.
    qiIds = 1:min(size(G,2),size(bc,1));
    % Extract the relevant entries.
    G_ = G(:,qiIds,:);
    bc_ = permute(bc(qiIds,:),[1 3 2]);
    br_ = permute(br(qiIds,:),[3 1 2]);
    % Scale and offset the zonotope to a new hypercube with center bic and 
    % radius bir.
    offset = pagemtimes(G_,bc_);
    % Offset the center.
    if ndims(c) > 2 % iff options.nn.interval_center
        c = c + offset;
    else
        c = c + offset(:,:);
    end
    % Scale the generators.
    G(:,qiIds,:) = G(:,qiIds,:).*br_;
end

function [A,b,newSplits,constNrIdx] = aux_convertSplitConstraints( ...
    As,bs,nrXis)
    % Consider all combinations between the given constraints, A*x <= b.
    if ~isempty(As)
        % Obtain the number of split-constraints.
        [ps,q,bSz] = size(As);
        % Obtain the number of pieces.
        [~,pcs,~] = size(bs);
        % Compute number of new splits.
        newSplits = (pcs+1)^ps;

        % We flip the signs of the constraints to realize splitting; -1
        % represents a lower bound, i.e., -A*x > -b, whereas 1 represents
        % an upper bound, i.e., A*x <= b.
        constrSign = [-1; 1];

        % Duplicate each halfspace for a lower and an upper bound.
        As_ = repelem(As,2,1,1);
        % Duplicate offsets for lower and upper bound.
        bs_ = repelem(bs,1,2,1);
        % Duplicate the indices for the split neurons.
        constNrIdx = repelem(nrXis,2,1);

        % Scale the constraints; -1 for upper bound and 1 for lower bound.
        As_ = repmat(constrSign,ps,1).*As_;
        % Duplicate the constraint for the new splits.
        A_ = permute(repelem(As_,1,1,1,newSplits),[2 1 4 3]);

        % Mark unused bounds by NaN.
        bs_ = cat(2,NaN(ps,1,bSz),bs_,cat(2,NaN(ps,1,bSz)));
        % Scale the offsets; -1 for upper bound and 1 for lower bound.
        bs_ = repmat(constrSign',1,pcs+1).*bs_;
        % Reshape and combine the lower and upper bounds.
        bs_ = reshape(permute(reshape(permute(bs_,[4 2 1 3]), ...
            [2 pcs+1 ps bSz]),[1 3 2 4]),[2*ps pcs+1 bSz]);
        % Extend the offsets.
        b_ = [bs_ zeros([2*ps newSplits - (pcs+1) bSz],'like',bs_)];
        % Compute all combinations of the splits.
        idx = pcs+1;
        for i=1:(ps-1)
            % Increase the index.
            idx_ = idx*(pcs+1);
            % Repeat the current combined splits.
            b_(1:2*i,1:idx_,:) = repmat(b_(1:2*i,1:idx,:),1,pcs+1,1);
            % Repeat the elements of the next split and append them.
            b_(2*i + (1:2),1:idx_,:) = ...
                repelem(b_(2*i + (1:2),1:(pcs+1),:),1,(pcs+1)^i,1);
            % Update the index of the combined splits.
            idx = idx_;
        end

        % Compute the neuron indices of each constraint. Then we can track
        % from which neuron the constraint stems; to extract the exact
        % bounds of the set.
        % Scale the indices; -1 for upper bound and 1 for lower bound.
        constNrIdx = repmat(constrSign,ps,1).*constNrIdx;
        % Duplicate the indices for the new splits.
        constNrIdx = repelem(constNrIdx,1,newSplits);

        % Find all unused constraints.
        nanIdx = isnan(b_);
        % Set all not needed constraints to zero.
        A_(:,nanIdx) = 0;
        b_(nanIdx) = 0;        
        % Mark all unused constraints.
        constNrIdx(nanIdx) = NaN;

        % Reshape the constraint matrix and offset.
        A = reshape(permute(A_,[2 1 3 4]),[2*ps q newSplits*bSz]);
        b = reshape(b_,[2*ps newSplits*bSz]);
    else
        % There are no additional constraints.
        newSplits = 1;
        A = zeros([0 size(As,[2 3])],'like',As);
        b = zeros([0 size(bs,2)],'like',bs);
        constNrIdx = zeros([0 size(nrXis,2)],'like',nrXis);
    end
end

% Constraints & Splitting -------------------------------------------------

function [xis,ris,dimId] = aux_split(xi,ri,hi,nSplits)
    % Split one input dimension into nSplits pieces.
    [n,bSz] = size(xi);
    % Split each input in the batch into nSplits parts.
    % 1. Find the input dimension with the largest heuristic.
    [~,sortDims] = sort(hi,1,'descend');
    dimIds = sortDims(1,:); 
    % Construct indices to use sub2ind to compute the offsets.
    splitsIdx = repmat(1:nSplits,1,bSz);
    bSzIdx = repelem((1:bSz)',nSplits);

    dimId = dimIds(1,:);
    linIdx = sub2ind([n bSz nSplits], ...
        repelem(dimId,nSplits),bSzIdx(:)',splitsIdx(:)');
    % 2. Split the selected dimension.
    xi_ = xi;
    ri_ = ri;
    % Shift to the lower bound.
    dimIdx = sub2ind([n bSz],dimId,1:bSz);
    xi_(dimIdx) = xi_(dimIdx) - ri(dimIdx);
    % Reduce radius.
    ri_(dimIdx) = ri_(dimIdx)/nSplits;
   
    xis = repmat(xi_,1,1,nSplits);
    ris = repmat(ri_,1,1,nSplits);
    % Offset the center.
    xis(linIdx(:)) = xis(linIdx(:)) + (2*splitsIdx(:) - 1).*ris(linIdx(:));
    
    % Flatten.
    xis = xis(:,:);
    ris = ris(:,:);
    % Replicate the dimension index.
    dimId = repmat(dimId,1,nSplits);
end

function [Ai,bi,dimIds,hi] = aux_dimSplitConstraints(hi,nSplits,nDims)
    % Construct dimension split constraints that splits #nDims dimensions 
    % into #nSplits pieces for subsequent refinement.

    % Obtain the number of dimensions and batch size.
    [n,bSz] = size(hi);
    nDims = min(nDims,n);

    % Split each input in the batch into nSplits parts.
    % 1. Find the input dimension with the largest heuristic.
    [hi,sortDims] = sort(hi,1,'descend');
    dimIds = sortDims(1:nDims,:); 
    hi = hi(1:nDims,:);

    % Compute dimension indices.
    dimIdx = sub2ind([nDims n bSz],repelem((1:nDims)',1,bSz), ...
        dimIds,repelem(1:bSz,nDims,1));

    % 2. Construct the constraints.
    Ai = zeros([nDims n bSz],'like',hi);
    Ai(dimIdx) = 1;
    bi = repelem(-1 + (1:(nSplits-1)).*(2/nSplits),nDims,1,bSz); % Specify offsets.
end

function [Ao,bo] = aux_outputDimSplitConstraints(y,Gy,nSplits,nDims)
    % Construct output split constraints that splits #nDims dimensions 
    % into #nSplits pieces for subsequent refinement.

    % Obtain the number of dimensions and batch size.
    [nK,qK,bSz] = size(Gy);
    nDims = min(nDims,nK);

    % Compute the radius of each output dimension.
    r = reshape(sum(abs(Gy),2),size(Gy,[1 3]));

    % 1. Find the output dimensions with the largest radius.
    [~,sortDims] = sort(r,1,'descend');
    dimIds = sortDims(1:nDims,:); 

    % 2. Construct the constraints.
    Ao = Gy(sub2ind([nK qK bSz], ...
        permute(repelem(dimIds,1,1,qK),[1 3 2]), ...
        repelem((1:qK),nDims,1,bSz), ...
        repelem(permute(1:bSz,[1 3 2]),nDims,qK,1) ...
    ));
    bo = permute(r(sub2ind(size(r),dimIds,repelem(1:bSz,nDims,1))),[1 3 2])...
        .*repelem(-1 + (1:(nSplits-1)).*(2/nSplits),nDims,1,bSz); % Specify offsets.
end

function [As,bs,constNrIdx,h] = aux_neuronConstraints(nn,options, ...
    idxLayer,heuristic,nSplits,nNeur,numInitGens,prevNrXs)
    % Assume: input was propagated and stored including sensitivity.
    % Output: 
    % - As, bs: individual constraints for neuron splits nrConst
    %   e.g. A(i,:)*beta <= b(i) and -A(i,:)*beta >= -b(i)
    % - constNrIdx: indices of neuron splits

    % Compute batch size.
    % bSz = nnz(unknown);
    
    % Initialize constraints; we insert dummy splits to prevent splitting 
    % a dimension twice.
    As = zeros([0 nNeur 1],'like',prevNrXs);
    bs = zeros([nSplits-1 nNeur 1],'like',prevNrXs);
    q = 0; % Number of considered generators.
    % Initial heuristics.
    h = -ones([nNeur 1],'like',prevNrXs);
    % Initialize indices of neuron split.
    constNrIdx = Inf([nNeur 1],'like',prevNrXs);
    
    % Enumerate the layers of the neural networks.
    [layers,~,~,succIdx] = nn.enumerateLayers();
    
    if isempty(idxLayer)
        idxLayer = 1:length(layers);
    end
    % Compute the indices of ReLU layers.
    idxLayer = idxLayer(arrayfun(@(i) ...
        isa(layers{i},'nnActivationLayer'),idxLayer));
    
    % Iterate through the layers and find max heuristics and propagate
    % constrains.
    for i=idxLayer
        % Obtain i-th layer.
        layeri = layers{i};
        % Obtain the input set and compute the bounds.
        [li,ui,cil,ciu,Gi] = aux_computeBoundsOfInputSet(layeri, ...
            options,[],[],0,[],[],[],[],prevNrXs);
        % Compute the center.
        ci = 1/2*(ciu + cil);
        % Obtain number of hidden neurons.
        [nk,qi,bSz] = size(Gi);

        % Obtain the indices of the approximation error generators.
        % approxErrGenIds = layeri.approxErrGenIds;
        % Obtain the indices of the neurons of the current layer.
        neuronIds = layeri.neuronIds;

        % Obtain the approximation errors.
        dl = layeri.backprop.store.dl;
        du = layeri.backprop.store.du;
        % Compute center and radius of approximation errors.
        % dc = 1/2*(du + dl);
        dr = 1/2*(du - dl);
    
        % Obtain the sensitivity for heuristic.
        Si_ = max(abs(layeri.sensitivity),1e-6);
        sens = reshape(max(Si_,[],1),nk,[]);

        if size(As,3) < bSz
            padBSz = bSz-size(As,3);
            % Pad to the correct batch size.
            As = cat(3,As,zeros([q nNeur padBSz],'like',As));
            bs = cat(3,bs,zeros([(nSplits-1) nNeur padBSz],'like',bs));
            h = [h -ones([nNeur padBSz],'like',h)];
            constNrIdx =  [constNrIdx,Inf([nNeur padBSz],'like',constNrIdx)];
        end

        if q < qi
            % Pad constraints with zeros.
            As = [As; zeros([qi-q nNeur bSz],'like',As)];
            % Update number of constraints.
            q = qi;
        else
            % Pad generators with zeros.
            Gi = [Gi zeros([nk qi-q bSz],'like',Gi)];
        end
    
        % Append new constraints.
        Asi = Gi;
        As = cat(2,As,permute(Asi,[2 1 3]));

        switch options.nn.split_position
            case 'zero'
                % Split into #nSplits pieces around 0.
                nSplits_ = floor((nSplits-1)/2);
                splitEnum = 1/(nSplits_+1).*(1:floor((nSplits-1))/2)';
                bil = flip(splitEnum).*permute(li,[3 1 2]);
                biu = splitEnum.*permute(ui,[3 1 2]);
                if mod(nSplits,2) == 0
                    % Include the center in the lower bounds.
                    bil = [bil; zeros([1 nk bSz],'like',ci)];
                end
                % Combine the bounds.
                bsi = [bil; biu];
                % Subtract the center.
                bsi = reshape(bsi - permute(ci,[3 1 2]),[nSplits-1 nk bSz]);
            case 'middle'
                % Split into #nSplits pieces around the middle.
                splitEnum = linspace(-1,1,nSplits+1)';
                splitEnum = splitEnum(2:end-1);
                bsi = splitEnum.*permute(ri,[3 1 2]);
            otherwise
                % Invalid option.
                throw(CORAerror('CORA:wrongFieldValue', ...
                    'options.nn.split_position',{'zero','middle'}));
        end

        % Append the new offsets.
        bs = cat(2,bs,bsi);

        % Obtain the gradient of the zonotope norm.
        if isfield(layeri.backprop.store,'approx_error_gradients')
            % Obtain the stored gradient.
            grad = layeri.backprop.store.approx_error_gradients;
        else
            % There is no stored gradient.
            grad = 0;
        end        
        % Obtain the neuron similarity based on the sensitivity.
        if isfield(layeri.backprop.store,'similarity')
            % Obtain the stored similarity.
            sim = layeri.backprop.store.similarity;
        else
            % There is no stored similarity.
            sim = [];
        end
        % Compute the heuristic.
        hi = aux_computeHeuristic(heuristic,i,li,ui,dr,sens,grad, ...
            sim,prevNrXs,neuronIds,true,0.7);
    
        % Append heuristic and sort.
        [h,idx] = sort([h; hi(:,:)],1,'descend');
    
        % Only keep the constraints for the top neurons.
        h = h(1:nNeur,:);

        % Obtain the indices for the relevant constraints.
        sIdx = sub2ind(size(As,2:3), ...
            idx(1:nNeur,:),repmat(1:bSz,nNeur,1));
    
        % Extract constraints.
        As = reshape(As(:,sIdx),[q nNeur bSz]);
        bs = reshape(bs(:,sIdx),[nSplits-1 nNeur bSz]);
    
        % Update indices.
        constNrIdx = [constNrIdx; repmat(neuronIds,bSz,1)'];
        constNrIdx = reshape(constNrIdx(sIdx),[nNeur bSz]);
    end

    % Transpose constraint matrix.
    As = permute(As,[2 1 3]);
    bs = permute(bs,[2 1 3]);

    if options.nn.add_orth_neuron_splits
        % Add the orthogonal neuron splits.

        % Obtain the number of constraints.
        [p,~,~] = size(As);

        % Extract the most important input dimension; make the constraint
        % orthogonal w.r.t. that dimension constraints.
        As_ = max(abs(As(:,1:numInitGens,:)),1e-6);
        [~,dimIds] = max(As_,[],2);

        % 1. Generate unit vector along most important dimension.
        v = zeros([p numInitGens bSz],'like',As_);
        dimIdx = sub2ind([p numInitGens bSz], ...
            repmat((1:p)',1,bSz),dimIds(:,:),repelem(1:bSz,p,1));
        v(dimIdx) = 1;
        % Move everything into the batch for easier computations.
        As_ = reshape(permute(As_,[2 1 3]),[1 p*numInitGens*bSz]);
        v = reshape(permute(v,[2 1 3]),[1 p*numInitGens*bSz]);
        
        % 2. Make the vector orthogonal to the input dimensions 
        % of the split constraints.
        proj = As_.*pagemtimes(v,'none',As_,'transpose') ...
            ./pagemtimes(As_,'none',As_,'transpose');
        vOrth = permute(reshape(v - proj,[numInitGens p bSz]),[2 1 3]);
        
        % 3. Normalize the orthogonal vector and embed into the full space.
        AsOrth = As;
        vOrthNorm = sqrt(sum(sum(vOrth.^2,1),2)); % pagenorm(vOrth)
        AsOrth(:,1:numInitGens,:) = vOrth./vOrthNorm;
        
        % 4. Append the orthogonal constraints (the offsets are identical
        % because we rotate the constraints along the origin).
        As = [As; AsOrth];
        bs = repmat(bs,2,1,1); % [bs; zeros(size(bs),'like',bs)];

        % Append NaN for the orthogonal constraints, because they are not
        % directly related to a neuron.
        constNrIdx = [constNrIdx; NaN(size(constNrIdx),'like',constNrIdx)];
        % Append -1 for the orthogonal constraints, because they do not
        % have a heuristic
        h = [h; -ones(size(h),'like',h)];
    end
end

function [At,bt,nrCt] = aux_reluTriangleConstraints(nn,options, ...
    idxLayer,heuristic,bc,br,scaleInputSets,prevNrXs)
    % Assume: input was propagated and stored.
    % Output: 
    % - At, bt: constraints for unstable neurons, 
    %   i.e., (i) ReLU(x) >= 0, (ii) ReLU(x) >= x, and
    %   (iii) ReLU(x) <= u/(u-l)*(x - l)

    % Specify a tolerance to avoid numerical issues.
    tol = 1e-8;

    % Obtain the number of constraints.
    numConstr = options.nn.num_relu_constraints;
    
    % Initialize constraints.
    q = 0; % Number of considered generators.
    p = 0; % Number of constraints.
    % (i) ReLU(x) >= 0
    At0 = [];
    bt0 = [];
    % (ii) ReLU(x) >= x
    Atd = [];
    btd = [];
    % (iii) ReLU(x) <= u/(u-l)*(x-l)
    % Atd2 = [];
    % btd2 = [];
    % % (iv) ReLU(x) >= ReLU(li)
    % Atl = [];
    % btl = [];
    % % (iv) ReLU(x) <= ReLU(ui)
    % Atu = [];
    % btu = [];
    % Initial heuristics.
    h = [];
    % Initialize indices of neuron constraints.
    % layerIdx = [];
    nrCt = [];
    
    % Enumerate the layers of the neural networks.
    [layers,~,~,succIdx] = nn.enumerateLayers();

    if isempty(scaleInputSets)
        scaleInputSets = zeros([1 length(layers)],'logical');
    end

    if isempty(idxLayer)
        idxLayer = 1:length(layers);
    end
    % Compute the indices of ReLU layers.
    idxLayer = idxLayer( ...
        arrayfun(@(i) isa(layers{i},'nnReLULayer'),idxLayer));

    % Obtain the batch size.
    [~,bSz] = size(bc);
    
    % Iterate through the layers and find maximal unstable neurons.
    for i=idxLayer
        % Obtain i-th layer.
        layeri = layers{i}; 
        % Obtain the input set and compute the bounds.
        if scaleInputSets(i)
            [li,ui,cil,ciu,Gi] = aux_computeBoundsOfInputSet(layeri, ...
                options,[],[],0,bc,br,[],[],prevNrXs);
        else
            [li,ui,cil,ciu,Gi] = aux_computeBoundsOfInputSet(layeri, ...
                options,[],[],0,zeros(size(bc),'like',bc), ...
                    ones(size(br),'like',br),[],[],prevNrXs);
        end
        % Obtain the indices for the generators containing the
        % approximation errors.
        approxErrGenIds = layeri.approxErrGenIds;
        % Obtain the indices of the neurons of the current layer.
        neuronIds = layeri.neuronIds;
        % Check is neuron is stable.
        isUnstable = li < 0 & 0 < ui;
    
        % Obtain the slope and approximation errors.
        dl = layeri.backprop.store.dl;
        du = layeri.backprop.store.du;
        % Compute center and radius of approximation errors.
        dr = 1/2*(du - dl);

        if size(dr,2) < bSz
            % Duplicate the approximation error (there was neuron 
            % splitting involved).
            newSplits = bSz/size(dr,2);
            dr = repelem(dr,1,newSplits);
        end

        % Obtain successor of the i-th layer.
        layerj = layers{succIdx(i)};  
        % Obtain the input set and compute the bounds.
        if scaleInputSets(succIdx(i))
            [~,~,cjl,cju,Gj] = aux_computeBoundsOfInputSet(layerj, ...
                options,[],[],0,bc,br,[],[],prevNrXs);
        else
            [~,~,cjl,cju,Gj] = aux_computeBoundsOfInputSet(layerj, ...
                options,[],[],0,zeros(size(bc),'like',bc), ...
                    ones(size(br),'like',br),[],[],prevNrXs);
        end
        % Obtain the number of generators and the batch size.
        [nk,qj,~] = size(Gj);

        % Obtain the sensitivity for heuristic.
        Si_ = max(abs(layeri.sensitivity),1e-6);
        sens = reshape(max(Si_,[],1),nk,[]);
        if ~isempty(sens) && size(sens,2) < bSz
            % Duplicate the sensitivity (there was neuron splitting involved).
            newSplits = bSz/size(sens,2);
            sens = repelem(sens,1,newSplits);
        end

        if size(At0,3) < bSz
            % Match the batch size.
            At0 = repelem(At0,1,1,bSz/size(At0,3));
            Atd = repelem(Atd,1,1,bSz/size(Atd,3));
            % Atd2 = repelem(Atd2,1,1,bSz/size(Atd2,3));
            % Atl = repelem(Atl,1,1,bSz/size(Atl,3));
            % Atu = repelem(Atu,1,1,bSz/size(Atu,3));
        end

        if q < qj
            % Pad constraints with zeros.
            At0 = [At0; zeros([qj-q p bSz],'like',At0)];
            Atd = [Atd; zeros([qj-q p bSz],'like',Atd)];
            % Atd2 = [Atd2; zeros([qj-q p bSz],'like',Atd2)];
            % Atl = [Atl; zeros([qj-q p bSz],'like',Atl)];
            % Atu = [Atu; zeros([qj-q p bSz],'like',Atu)];
            % Update number of constraints.
            q = qj;
        end

        if q > size(Gi,2)
            % Pad generators with zeros.
            Gi = [Gi zeros([nk q-size(Gi,2) bSz],'like',Gi)];
        end
        if q > qj
            % Pad generators with zeros.
            Gj = [Gj zeros([nk q-qj bSz],'like',Gj)];
        end

        % Reverse the sign of the approximation errors.
        Gj_ = Gj;
        Gj_(:,approxErrGenIds,:) = -Gj_(:,approxErrGenIds,:);

        % (i) ReLU(x) >= 0 
        % --> cj + Gj*\beta - dr >= ReLU(x) >= 0 
        % <--> -Gj*\beta + dr <= cj
        Ati0 = -Gj_;
        bti0 = cju + tol;
        % Permute the constraints s.t. the generators are in the first 
        % dimension.
        Ati0 = permute(Ati0,[2 1 3]);
        % Invalidate constraints for stable neurons.
        Ati0(:,~isUnstable) = NaN;
        bti0(~isUnstable) = NaN;
        % Append new constraints.
        At0 = cat(2,At0,Ati0);
        bt0 = [bt0; bti0];
    
        % (ii) ReLU(x) >= x 
        % --> cj + Gj*\beta - dr >= ReLU(x) >= x = ci + Gi*\beta
        % <--> (Gi-Gj)*\beta + dr <= cj - ci
        % Compute difference of generator matrices.
        Atid = Gi - Gj_;
        btid = cju - cil + tol;
        % Permute the constraints s.t. the generators are in the first 
        % dimension.
        Atid = permute(Atid,[2 1 3]);
        % Invalidate constraints for stable neurons.
        Atid(:,~isUnstable) = NaN;
        btid(~isUnstable) = NaN;
        % Append new constraints.
        Atd = cat(2,Atd,Atid);
        btd = [btd; btid];

        % Constraints are redundant.
        % % (iii) ReLU(x) <= m*(x-li) + ReLU(li)
        % % --> cj + Gj*\beta + dr <= ReLU(x) <= m*(x-li) + ReLU(li)
        % % <--> (Gj - m*Gi)*\beta + dr <= m*(ci-li) - cj + ReLU(li),
        % % where m = (ReLU(ui) - ReLU(li))/(ui-li).
        % % Compute the approximation slope; add 1e-10 to avoid numerical issues.
        % m_ = (max(ui,0) - max(li,0))./(ui - li);
        % Atid2 = Gj - permute(m_,[1 3 2]).*Gi;
        % btid2 = m_.*(ciu - li) - cjl + max(li,0) + tol;
        % % Permute the constraints s.t. the generators are in the first 
        % % dimension.
        % Atid2 = permute(Atid2,[2 1 3]);
        % % Avoid numerical issues if bounds are too close.
        % boundsAreEqual = (ui - li) < tol;
        % % Invalidate constraints for stable neurons.
        % Atid2(:,~isUnstable | boundsAreEqual) = NaN;
        % btid2(~isUnstable | boundsAreEqual) = NaN;
        % % Append new constraints.
        % Atd2 = cat(2,Atd2,Atid2);
        % btd2 = [btd2; btid2];
        
        % % (iv)ReLU(x) >= ReLU(li)
        % % --> cj + Gj*\beta - dr >= ReLU(li)
        % % <--> -Gj*\beta + dr <= cj - ReLU(li),
        % Atil = -Gj_;
        % btil = cju - max(li,0);
        % % Permute the constraints s.t. the generators are in the first 
        % % dimension.
        % Atil = permute(Atil,[2 1 3]);
        % % Append new constraints.
        % Atl = cat(2,Atl,Atil);
        % btl = [btl; btil];
        % 
        % % (v) ReLU(x) <= ReLU(ui)
        % % --> cj + Gj*\beta + dr <= ReLU(ui)
        % % <--> Gj*\beta + dr <= ReLU(ui) - cj,
        % Atiu = Gj;
        % btiu = max(ui,0) - cjl;
        % % Permute the constraints s.t. the generators are in the first 
        % % dimension.
        % Atiu = permute(Atiu,[2 1 3]);
        % % Append new constraints.
        % Atu = cat(2,Atu,Atiu);
        % btu = [btu; btiu];

        % Obtain the gradient of the zonotope norm.
        if isfield(layeri.backprop.store,'approx_error_gradients')
            % Obtain the stored gradient.
            grad = layeri.backprop.store.approx_error_gradients;
            if size(grad,2) < bSz
                % Duplicate the gradient (there was neuron splitting involved).
                newSplits = bSz/size(grad,2);
                grad = repelem(grad,1,newSplits);
            end
        else
            % There is no stored gradient.
            grad = 0;
        end
        % Compute the heuristic.
        hi = aux_computeHeuristic(heuristic,i,li,ui,dr,sens,grad, ...
            [],prevNrXs,neuronIds,true,0.7);

        % Append heuristic and sort.
        [h,idx] = sort([h; hi(:,:)],1,'descend');
        % Only keep the constraints for the top neurons.
        numConstr_ = min(numConstr,size(h,1));
        h = h(1:numConstr_,:);
    
        % Obtain the indices for the relevant constraints.
        cIdx = sub2ind(size(At0,2:3), ...
            idx(1:numConstr_,:),repmat(1:bSz,numConstr_,1));
    
        % Select the relevant constraints.
        At0 = reshape(At0(:,cIdx),[q numConstr_ bSz]);
        bt0 = reshape(bt0(cIdx),[numConstr_ bSz]);
        Atd = reshape(Atd(:,cIdx),[q numConstr_ bSz]);
        btd = reshape(btd(cIdx),[numConstr_ bSz]);
        % Atd2 = reshape(Atd2(:,cIdx),[q numConstr_ bSz]);
        % btd2 = reshape(btd2(cIdx),[numConstr_ bSz]);
        % Atl = reshape(Atl(:,cIdx),[q numConstr_ bSz]);
        % btl = reshape(btl(cIdx),[numConstr_ bSz]);
        % Atu = reshape(Atu(:,cIdx),[q numConstr_ bSz]);
        % btu = reshape(btu(cIdx),[numConstr_ bSz]);

        % Update number of constraints.
        p = size(At0,2);

        % % Update indices.
        nrCt = [nrCt; repmat(neuronIds,bSz,1)'];
        nrCt = reshape(nrCt(cIdx),[numConstr_ bSz]);
    end
    % Concatenate the constraints.
    At = cat(2,At0,Atd); % ,Atd2,Atl,Atu); 
    bt = [bt0; btd]; % ; btd2; btl; btu]; 
    nrTypesOfConstr = size(At,2)/size(nrCt,1);
    nrCt = repmat(nrCt,nrTypesOfConstr,1);
    % Find the minimal number of invalid constraints across the batch.
    minNumInvalidConstraints = min(sum(any(isnan(At),1),2),[],3);
    % Sort the constraints and remove invalidated constraints.
    [~,sortIds] = sort(isnan(bt),1,'descend');
    sortIdx = sub2ind(size(At,2:3),sortIds,repmat(1:bSz,size(At,2),1));
    % Reorder the constraints.
    At = reshape(At(:,sortIdx),size(At));
    bt = reshape(bt(sortIdx),size(bt));
    nrCt = reshape(nrCt(sortIdx),size(nrCt));
    % % Remove the invalid constraints.
    At(:,1:minNumInvalidConstraints,:) = [];
    bt(1:minNumInvalidConstraints,:) = [];
    nrCt(1:minNumInvalidConstraints,:) = [];
    % Set all remaining invalid constraints to 0.
    At(isnan(At)) = 0;
    bt(isnan(bt)) = 0;
    % Transpose constraint matrix.
    At = permute(At,[2 1 3]);
end

% Plotting ----------------------------------------------------------------

function cZeq = aux_2ConZonoWithEqConst(cZineq,apprErr)
    % Extract parameters of the constraint zonotope.
    c = double(gather(cZineq.c));
    G = double(gather(cZineq.G));
    r = double(gather(cZineq.r));
    A = double(gather(cZineq.A));
    b = double(gather(cZineq.b));

    % We convert the inequality constraints to equality constraints by 
    % adding a slack variable.

    % Obtain number of dimensions, generators, and batch size.
    [n,q,bSz] = size(G);
    % Obtain number of constraints.
    [p,~] = size(A);

    cZeq.c = c;
    % Add the radius to the generators.
    if any(r ~= 0,'all')
        G = cat(2,G,r.*eye(n));
        A = cat(2,A,zeros([p n bSz]));
    end
    % Add a slack variable.
    cZeq.G = cat(2,G,zeros([n p bSz]));
    % Compute scale for the slack variable.
    s = 1/2*(sum(abs(A),2) + permute(b,[1 3 2]));
    cZeq.A = cat(2,A,eye(p).*s);
    % Compensate for the slack variable.
    cZeq.b = b - s(:,:);
    % Set the approximation errors.
    cZeq.apprErr = double(gather(apprErr));
end

function [fig,hx0,hspec] = aux_initPlot(fig,plotDims,xs,ys,x0,r0,A,b,safeSet)
    % Plot the initial input set.
    subplot(1,2,1); hold on;
    title('Input Space')
    % Plot the initial input set.
    % plotPoints(xs,plotDims(1,:),'.k');
    hx0 = plot(interval(x0 - r0,x0 + r0),plotDims(1,:), ...
        'DisplayName','Input Set', ...
        'FaceColor',CORAcolor('CORA:simulations'),'FaceAlpha',0.0, ...
        'EdgeColor',CORAcolor('CORA:simulations'),'LineWidth',2);

    % Construct the halfspace specification.
    spec = polytope(double(gather(A)),double(gather(b)));

    % Plot the specification.
    subplot(1,2,2); hold on;
    title('Output Space')
    if safeSet
        safeSetStr = 'safe';
    else
        safeSetStr = 'unsafe';
    end

    plotPoints(ys,plotDims(2,:),'.k');
    hspec = plot(spec,plotDims(2,:),...
        'DisplayName',sprintf('Specification (%s)',safeSetStr), ...
        'FaceColor',CORAcolor(sprintf('CORA:%s',safeSetStr)),'FaceAlpha',0.2, ...
        'EdgeColor',CORAcolor(sprintf('CORA:%s',safeSetStr)),'LineWidth',2);
end

function [fig,hxi,hx,hxv,hy,hyv] = aux_plotInputAndOutputSets(fig, ...
    plotDims,x0,r0,res)
    % Obtain number of dimensions.
    [n,~] = size(x0);
    % Small interval to avoid plotting errors.
    pI = 1e-8*interval(-ones(n,1),ones(n,1));

    % Plot the input sets.
    subplot(1,2,1); hold on;
    % Plot the initial input set.
    hxi = plot(interval(x0 - r0,x0 + r0),plotDims(1,:), ...
        'DisplayName','Input Set', ...
        'FaceColor',CORAcolor('CORA:simulations'),'FaceAlpha',0.0, ...
        'EdgeColor',CORAcolor('CORA:simulations'),'LineWidth',2);
    % Store plot handles for potential deletion.
    hx = {};
    hxv = {};
    for j=1:size(res.Xs{end}.c,2)
        Xij = zonotope(res.Xs{end}.c(:,j),res.Xs{end}.G(:,:,j)) + pI;
        if res.Xs{end}.verified(j)
            hxv{end+1} = plot(Xij,plotDims(1,:), ...
                'DisplayName','Input Set (verified)', ...
                'FaceColor',CORAcolor('CORA:safe'),'FaceAlpha',0.5, ...
                'EdgeColor',CORAcolor('CORA:safe'),'LineWidth',2);
        else
            hx{end+1} = plot(Xij,plotDims(1,:), ...
                'DisplayName','Input Set', ...
                'FaceColor',CORAcolor('CORA:reachSet'),'FaceAlpha',0.2, ...
                'EdgeColor',CORAcolor('CORA:reachSet'),'LineWidth',2);
        end
    end
    % Plot the output sets.
    subplot(1,2,2); hold on;
    % Store plot handles for potential deletion.
    hy = {};
    hyv = {};
    for j=1:size(res.Ys{end}.c,2)
        Yij = zonotope(res.Ys{end}.c(:,j),res.Ys{end}.G(:,:,j)) + pI;
        if res.Xs{end}.verified(j)
            hyv{end+1} = plot(Yij,plotDims(2,:),'DisplayName','Output Set', ...
                'FaceColor',CORAcolor('CORA:safe'),'FaceAlpha',0.2, ...
                'EdgeColor',CORAcolor('CORA:safe'),'LineWidth',2);
        else
            hy{end+1} = plot(Yij,plotDims(2,:),'DisplayName','Output Set', ...
                'FaceColor',CORAcolor('CORA:reachSet'),'FaceAlpha',0.2, ...
                'EdgeColor',CORAcolor('CORA:reachSet'),'LineWidth',2);
        end
    end
end

function [fig,hxs_,hys_] = aux_plotCounterExampleCandidates(fig, ...
    plotDims,res)
    % Plot inputs.
    subplot(1,2,1); hold on;
    hxs_ = plotPoints(res.xs_{end},plotDims(1,:),'or', ...
        'DisplayName','Counterexample Candidate');
    % Plot outputs.
    subplot(1,2,2); hold on;
    hys_ = plotPoints(res.ys_{end},plotDims(2,:),'or', ...
        'DisplayName','Counterexample Candidate');
end

function [fig,huy,huy_,hux,hx,hx_] = ...
    aux_plotUnsafeOutputAndNewInputSets(fig,plotDims,res,lis,uis, ...
        splitsPerUnsafeSet)
    % Obtain number of dimensions.
    [n,~] = size(lis);
    % Small interval to avoid plotting errors.
    pI = 1e-8*interval(-ones(n,1),ones(n,1));

    % Store plot handles for potential deletion.
    huy = {};
    huy_ = {};
    if isfield(res,'uYs')
        % Plot unsafe output constraint zonotope.
        subplot(1,2,2); hold on;
        for j=1:size(res.uYs{end}.c,2)
            % Plot with approximation error.
            % uYij_ = conZonotope( ...
            %     res.uYs{end}.c(:,j),res.uYs{end}.G(:,:,j),...
            %     res.uYs{end}.A(:,:,j),res.uYs{end}.b(:,j) ...
            %         + res.uYs{end}.apprErr(:,j)) + pI;
            % huy_{end+1} = plot(uYij_,plotDims(2,:),'--', ...
            %     'DisplayName','Output Set (unsafe, w. Approx. Err.)', ...
            %     'EdgeColor',CORAcolor('CORA:highlight1'),'LineWidth',1, ...
            %     'FaceColor',CORAcolor('CORA:reachSet'),'FaceAlpha',0.2 ...
            %     );
            % Plot without approximation error.
            uYij = conZonotope( ...
                res.uYs{end}.c(:,j),res.uYs{end}.G(:,:,j),...
                res.uYs{end}.A(:,:,j),res.uYs{end}.b(:,j)) + pI;
            huy{end+1} = plot(uYij,plotDims(2,:), ...
                'DisplayName','Output Set (unsafe)', ...
                'FaceColor',CORAcolor('CORA:highlight1'),'FaceAlpha',0.2, ...
                'EdgeColor',CORAcolor('CORA:highlight1'),'LineWidth',2);
        end
    end
    % Plot new input sets.
    subplot(1,2,1); hold on;
    % Store plot handles for potential deletion.
    hux = {};
    hx = {};
    hx_ = {};
    for j=1:size(lis,2)
        if isfield(res,'uXs') && mod(j-1,splitsPerUnsafeSet) == 0
            j_ = (j-1)/splitsPerUnsafeSet + 1;
            % Plot unsafe input constraint zonotope.
            uXij = conZonotope( ...
                res.uXs{end}.c(:,j_),res.uXs{end}.G(:,:,j_), ...
                res.uXs{end}.A(:,:,j_),res.uXs{end}.b(:,j_)) + pI;
            hux{end+1} = plot(uXij,plotDims(1,:), ...
                'DisplayName','Input Set (unsafe)', ...
                'EdgeColor',CORAcolor('CORA:highlight1'),'LineWidth',1, ...
                'FaceColor',CORAcolor('CORA:reachSet'),'FaceAlpha',0.2 ...
                );
        end
        % Obtain new input set.
        Xij = interval(double(gather(lis(:,j))),double(gather(uis(:,j)))) + pI;
        hx{end+1} = plot(Xij,plotDims(1,:), ...
            'DisplayName','Input Set', ...
            'EdgeColor',CORAcolor('CORA:simulations'),'LineWidth',2);
    end
end

% Splitting Tree Visualization --------------------------------------------

function aux_plotSplittingTree(tree,labels)
    % % Plot the splitting tree.
    % treeplot(tree)
    % % Create position for the labels.
    % [x,y] = treelayout(tree);
    % % Add the labels.
    % text(x,y+0.01,labels);

    % Build a directed graph, that shows the hierachy.
    G = digraph;
    for i = 2:length(tree)
        G = addedge(G,tree(i),i);
    end
    % Plot the graph.
    h = plot(G, 'Layout', 'layered');
    % Add the node labels.
    h.NodeLabel = labels;
end

function [labels,idx] = aux_markVerifiedInTheSplittingTree(labels,idx, ...
    isVerified,postfix)
    % Update the tree labels for verified batch entries; idx indexes into
    % the cell of labels.

    % Update the splitting tree. Mark the nodes as verified.
    labels(idx(isVerified)) = repmat( ...
        {sprintf('Verified (%s)',postfix)},1,nnz(isVerified));
    % We only keep the parent indices of the unknown entries.
    idx(isVerified) = [];
end

function [tree,labels,newIdx] = aux_addNewSplitsToTheSplittingTree( ...
    tree,labels,idx,dimIds,nrIds,newSplits)
    % Obtain the batch size.
    [~,bSz] = size(dimIds);
    % Create a function handle to convert the dimension and neuron indices 
    % to a readable string, skipping NaNs.
    ids2str = @(prefix,ids) arrayfun(@(col) ...
        aux_joinSplitIds(prefix,ids(:,col)),1:size(ids,2), ...
            'UniformOutput',false);
    % Create strings for the input dimensions and neurons.
    inputSplitStr = ids2str('Input Dim',dimIds);
    neuronSplitStr = ids2str('Neuron',nrIds);
    % Update labels only for non-empty strings.
    labels(idx) = cellfun(@(s1,s2) cell2mat(join([{},s1,s2],', ')), ...
        inputSplitStr,neuronSplitStr,'UniformOutput',false);
    % Compute the indices for the new nodes.
    newIdx = length(tree) + (1:newSplits*bSz);
    % Add new nodes for the new splits.
    tree = [tree repelem(idx,1,newSplits)];
    labels = [labels repmat({'Unknown'},1,newSplits*bSz)];
end

function label = aux_joinSplitIds(prefix,ids)
    % Convert a vector of split ids to a readable format.
    if all(isnan(ids))
        % There are not split dimension so the string is empty.
        label = [];
    else
        % Join the ids and prepend the prefix.
        label = sprintf('%s: %s',prefix,join(string(ids(~isnan(ids))),','));
    end
end

% ------------------------------ END OF CODE ------------------------------
