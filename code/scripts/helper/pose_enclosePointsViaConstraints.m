function [P_total,Ps] = pose_enclosePointsViaConstraints(critPixels,C_struct)
% pose_encloseSupportFunc - encloses all convex regions
%
% Syntax:
%    R = pose_forward(pose)
%
% Inputs:
%    critPixels - numeric, to be enclosed, 2 x nrVertex
%    C_struct - struct
%
% Outputs:
%    P_total - polytope, final constraints
%    Ps - cell of polytope, constraints from each point
%           
% References:
%    [1] Koller et al. Out of the shadows: Exploring a latent space for 
%        neural network verification. arxiv. 2025
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: pose_prepareCandOffline

% ------------------------------ BEGIN CODE -------------------------------


% compute remaining part of [1, Prop. 2]
conD_post = arrayfun(@(i) max(C_struct.V_PCF.conDirs(:,:,i) * critPixels{i},[],2),1:numel(critPixels),'UniformOutput',false);
conD_post = cat(3,conD_post{:});
C = C_struct.V_PCF.conC;
d = C_struct.V_PCF.conD_pre + ... % pre-computed 
    conD_post + ... % critical pixels
    C_struct.V_PCF.conDiscError; % error due to discretization

% filter valid constraints
idxValid = ~any(isnan(conD_post),1);
numValid = nnz(idxValid);
C = C(:,:,idxValid);
d = d(:,:,idxValid);

% init individual constraints per (valid) vertex if desired
I_cube = interval(-ones(6,1),ones(6,1));
if nargout == 2
    Ps = arrayfun(@(i) I_cube & polytope(C(:,:,i),d(:,:,i)), 1:numValid, 'UniformOutput',false);
end

% gather all constraints
C = reshape(permute(C,[1,3,2]),4*numValid,6);
d = reshape(d, 4*numValid,1);
P_total = I_cube & polytope(C,d);

end


% Auxiliary functions -----------------------------------------------------


% ------------------------------ END OF CODE ------------------------------
