function C_struct = pose_prepareCandOffline(V_prop,PZ_pose, PZ_l_PCF, w, h)
% pose_prepareCandOffline - prepares the computed pose candidate offline
%
% Syntax:
%    C_struct = pose_prepareCandOffline(V_prop,PZ_pose, PZ_l_PCF)
%
% Inputs:
%    V_prop - struct, properties of vertices
%    PZ_pose - polyZonotope, uncertain pose candidate
%    PZ_l_PCF - polyZonotope, in PCF
%
% Outputs:
%    C_struct - struct, contains prepared values for fast online execution
%        .PZ_pose - polyZonotope
%        .isContained - logical, w x h
%        .V_PCF - struct, contains prepared values for each vertex
%            .isContained - logical, w x h x nrVertex 
%            .idxValid - logical, nrVertex x 1
%            .dirCPixel - numeric, nrVertex x 2
%            .invCPixel - logical, nrVertex x 2, whether the inverse should be computed
%            .conC - numeric, nrConstaints x 6 x nrVertex
%            .conD_pre - numeric, nrConstaints x 1 x nrVertex
%            .conDirs - numeric, nrConstaints x 2 x nrVertex
%            .conDiscError - numeric, 1 x 1 x nrVertex
%               s.t. given a pixel p: C <= (conDir * p + conDiscError) + conD_pre [1, Prop. 2]
%           
% References:
%    [1] Koller et al. Out of the shadows: Exploring a latent space for 
%        neural network verification. arxiv. 2025
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: neuralNetwork/verify

% ------------------------------ BEGIN CODE -------------------------------

% init C_struct
C_struct = struct;
C_struct.PZ_pose = PZ_pose;
C_struct.PZ_l_PCF = PZ_l_PCF;
C_struct.V_prop = V_prop;

% compute contained pixels
convRegions = pose_encloseSupportFunc(V_prop, PZ_l_PCF);
[isContained,~,R_PCFcontained] = pose_getContainedPixels(w,h,convRegions);
C_struct.isContained = isContained;

% compute contained pixel for each vertex
PZ_l_PCF_project = arrayfun(@(i) project(PZ_l_PCF,(i-1)*2 + (1:2)),1:V_prop.nrRVertices,'UniformOutput',false);
vertexRegions = arrayfun(@(i) aux_computeVertexRegion(PZ_l_PCF_project{i}),1:V_prop.nrRVertices,'UniformOutput',false);
V_PCFcontained = cellfun(@(vertexRegion) pose_getContainedPixels(w,h,{vertexRegion}), vertexRegions,'UniformOutput',false);
V_PCFcontained = cat(3,V_PCFcontained{:});
C_struct.V_PCF.isContained = V_PCFcontained;

% check validity of each vertex ---
idxValid = true(V_prop.nrRVertices,1);
% check if non-overlapping (with other vertices)
idxValid = idxValid & arrayfun(@(i) all(sum(V_PCFcontained(:,:,i) & (V_PCFcontained(:,:,i) == V_PCFcontained),3) <= 1,"all"), 1:V_prop.nrRVertices)';
% check if only on regions where the vertex is also present
convRegionsIdx = cell2mat(V_prop.convRegionsIdx);
convRegionsIdx = convRegionsIdx(:,~V_prop.jumps);
idxValid = idxValid & arrayfun(@(i) all(sum(V_PCFcontained(:,:,i) & (V_PCFcontained(:,:,i) == R_PCFcontained(:,:,~convRegionsIdx(:,i))),3) <= 0,"all"), 1:V_prop.nrRVertices)';
% keep always valid points
idxValid(V_prop.alwaysValid) = true;
% except if not within frame all together
idxValid = idxValid & reshape(any(V_PCFcontained,1:2),[],1);
% store validity
C_struct.V_PCF.idxValid = idxValid;

% gather vertices of center
l_PCF = nan(2,V_prop.nrVertices);
l_PCF(:,~V_prop.jumps) = reshape(PZ_l_PCF.c,2,V_prop.nrRVertices);

% get support func dirs
l_PCF = [l_PCF, nan(2,1)];
lastNaN = 0;
C_struct.V_PCF.dirCPixel = nan(1,2,V_prop.nrRVertices);
C_struct.V_PCF.invCPixel = nan(V_prop.nrRVertices,1);

for k=2:(size(l_PCF,2))
    if all(isnan(l_PCF(:,k)))
        % process first vertex of last sequence
        idx = [k-1,lastNaN+1,lastNaN+2];
        lastNaN = k;
    elseif all(isnan(l_PCF(:,k+1)))
        % last vertex
        idx = [k-1,k,lastNaN+1];
    elseif all(isnan(l_PCF(:,k-1)))
        % process first vertex when end is known
        continue
    else
        % normal case
        idx = [k-1,k,k+1];
    end

    cnt = idx(2);
    l_PCF_km1 = l_PCF(:,idx(1));
    l_PCF_k = l_PCF(:,idx(2));
    l_PCF_kp1 = l_PCF(:,idx(3));

    % support dir is average of vectors to current point
    v1 = (l_PCF_k - l_PCF_km1); v2 = (l_PCF_k - l_PCF_kp1);
    dir = v1 / vecnorm(v1) + v2 / vecnorm(v2);
    
    % compute inverse problem if support is larger for neighboring points
    p = (l_PCF_km1 + l_PCF_kp1) / 2;
    inv = aux_isPointContained(l_PCF_km1,l_PCF_k,p) & aux_isPointContained(l_PCF_k,l_PCF_kp1,p);

    % save
    C_struct.V_PCF.dirCPixel(:,:,cnt) = dir'; 
    C_struct.V_PCF.invCPixel(cnt) = inv;
end

% make invCPixel logical
C_struct.V_PCF.invCPixel = C_struct.V_PCF.invCPixel == 1;
% filter out nans
C_struct.V_PCF.dirCPixel = C_struct.V_PCF.dirCPixel(:,:,~V_prop.jumps);
C_struct.V_PCF.invCPixel = C_struct.V_PCF.invCPixel(~V_prop.jumps);

% pre-compute constraints C,d using [1, Prop. 2]
conCell = arrayfun(@(i) aux_computeConstraints(idxValid(i),PZ_l_PCF_project{i},C_struct.V_PCF.invCPixel(i)),1:V_prop.nrRVertices,'UniformOutput',false);
conCell = cat(1,conCell{:});
C_struct.V_PCF.conC = cat(3,conCell{:,1});
C_struct.V_PCF.conD_pre = cat(3,conCell{:,2});
C_struct.V_PCF.conDirs = cat(3,conCell{:,3});
C_struct.V_PCF.conDiscError = cat(3,conCell{:,4});


end


% Auxiliary functions -----------------------------------------------------

function res = aux_isPointContained(v1,v2,p)
    % check if point is inside or outside of the given line of a polygon
    res = ((v2(1)-v1(1))*(p(2)-v1(2)) - (v2(2)-v1(2))*(p(1)-v1(1))) > 0;
end

function P = aux_computeVertexRegion(PZ_l_PCF_project)
    % compute the region surrounding a vertex as a polytope
    P = polytope(reduce(zonotope(PZ_l_PCF_project),'pca',1));
end

function conCell = aux_computeConstraints(idxValid,PZ_proj,invCPixel)
    % quick exit if given dim are not valid
    if ~idxValid
        conCell = {nan(4,6),nan(4,1),nan(4,2),nan(4,1)}; return
    end

    % extract generators with the same exponent as input set (identity)
    % PZ_proj = PZ_proj.relaxExponents(1);
    E = PZ_proj.E;
    idx = any(E == 1,1) & sum(E,1) == 1;
    PZ_proj_core = polyZonotope(0*PZ_proj.c,PZ_proj.G(:,idx),[],PZ_proj.E(:,idx),PZ_proj.id);
    PZ_proj_approx = exactPlus(PZ_proj,-PZ_proj_core);
    Z_proj_approx = zonotope(PZ_proj_approx);
    
    % find dominant directions of approx error
    PZ_proj_pca = reduce(PZ_proj_approx,'pca',1);
    conDirs = PZ_proj_pca.GI;
    % PZ_proj = zonotope(PZ_proj_core) + zonotope(PZ_proj_pca);
    
    % compute constraints in these directions
    conDirs = [conDirs'; -conDirs']; % positive & negative dir
    C = conDirs*PZ_proj_core.G * PZ_proj_core.E';
    d_pre = -conDirs*Z_proj_approx.c + sum(abs(conDirs*Z_proj_approx.G),2);

    % discretization error (TODO double-check!)
    conDiscError = 0.5*sum(abs(conDirs),2);

    % save
    conCell = {C,d_pre,conDirs,conDiscError};

end

% ------------------------------ END OF CODE ------------------------------
