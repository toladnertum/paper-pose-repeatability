function convRegions = pose_encloseSupportFunc(V_prop, PZ_l_PCF)
% pose_encloseSupportFunc - encloses all convex regions
%
% Syntax:
%    R = pose_forward(pose)
%
% Inputs:
%    V_prop - struct
%    PZ_l_PCF - polyZonotope, in PCF
%
% Outputs:
%    convRegions - cell of polytopes, convex regions
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: -

% ------------------------------ BEGIN CODE -------------------------------

% compact and convert PZ to Z for speed
% PZ_l_PCF = compact(PZ_l_PCF);
Z_l_PCF = zonotope(PZ_l_PCF);
Z_l_PCF = compact(Z_l_PCF);

% compute regions in l_PCF
convRegions = cell(size(V_prop.convRegionsIdx));
for i=1:numel(V_prop.convRegionsIdx)
    % for each convex region
    idx = V_prop.convRegionsIdx{i};
    idx = find(idx(~V_prop.jumps));
    nrVertices_i = numel(idx);

    % sort vertices clock-wise
    cs_idx = 2*(idx-1) + (1:2)';
    cs = Z_l_PCF.c(cs_idx);
    cs_mean = mean(cs,2);
    cs_cent = cs - cs_mean;
    angles = atan2(cs_cent(2,:),cs_cent(1,:));
    [~, order] = sort(angles, 'descend');
    cs_idx = cs_idx(:,order);
    % cs_cent = cs_cent(:,order);

    % init polytope
    nrSFevals = 2;
    A = nan(nrSFevals*nrVertices_i,2);

    % iterate through all subsequent vertices
    for k=1:nrVertices_i

        % find dimensions in l_PCF
        p1_dims = cs_idx(:,k);
        p2_dims = cs_idx(:,mod(k,nrVertices_i)+1);

        % read out center
        p1_k = Z_l_PCF.c(p1_dims);
        p2_k = Z_l_PCF.c(p2_dims);

        for j = 1:max(nrSFevals-1,1)
            % determine support function direction
            % (rotate p1 -> p2 by 90°, assumes clock-wise ordering)
            dir = p2_k - p1_k;
            rot90 = [0 -1; 1 0];
            dir_support = rot90 * dir;
    
            if j+1<nrSFevals
                % determine points for next iteration
                [~,p1_k] = supportFunc(project(Z_l_PCF,p1_dims),dir_support,'upper');
                [~,p2_k]= supportFunc(project(Z_l_PCF,p2_dims),dir_support,'upper');
            end

            % save dir
            A(nrSFevals*(k-1)+j,:) = dir_support';
        end

        if nrSFevals > 1
            % additionally run center -> vertices for sharp corners
            dir_support = Z_l_PCF.c(p1_dims) - cs_mean;
            A(nrSFevals*(k-1)+nrSFevals,:) = dir_support';
        end
    end

    % compute b (max over all vertices)
    SFevals = arrayfun(@(k) supportFunc_(project(Z_l_PCF,cs_idx(:,k)),A','upper'),1:nrVertices_i,'UniformOutput',false);
    b = max(cat(2,SFevals{:}),[],2);

    % init polytope
    P_i = polytope(A,b);
    % compact and normalize (for pixel inclusion check w/ tol)
    % P_i = compact(P_i);
    % P_i = normalizeConstraints(P_i,'A');

    % save
    convRegions{i} = P_i;
end

end


% Auxiliary functions -----------------------------------------------------


% ------------------------------ END OF CODE ------------------------------
