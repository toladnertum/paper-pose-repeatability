function [V_CCF,l_CCF,l_PCF] = pose_forward(V_prop, V_TCF, pose, f, w, h, lambdas)
% pose_forward - computes the forward pass of the given vertices and pose
%
% Syntax:
%    R = pose_forward(pose)
%
% Inputs:
%    V_prop - struct
%    V_TCF - numeric, vertices or boundary/reference in TCF
%    pose - numeric or polyZonotope
%    f, w, h - numeric, intrinsic parameters
%    lambdas - numeric, weighting used for boundary points
%
% Outputs:
%    V_CCF - vertices in camera lens coordinate frame 
%    l_CCF - vertices in camera sensor coordinate frame
%    l_PCF - vertices in projected coordinate frame
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: -

% ------------------------------ BEGIN CODE -------------------------------

if nargin < 7
    lambdas = [];
end

% intrinsic parameters
K = [
    f 0 w/2; 
    0 f h/2; 
    0 0 1
];

switch class(pose)
    case {'double','interval'}
        [V_CCF,l_CCF,l_PCF] = aux_forwardNumeric(V_prop, V_TCF, pose, K, lambdas);

    case 'polyZonotope'
        [V_CCF,l_CCF,l_PCF] = aux_forwardPZ(V_prop, V_TCF, pose, K, lambdas);

    otherwise
        throw(CORAerror('CORA:wrongValue','first',{'numeric','interval','polyZonotope'}))

end

end


% Auxiliary functions -----------------------------------------------------

function [V_CCF,l_CCF,l_PCF] = aux_forwardNumeric(V_prop, V_TCF, pose, K, lamdbas)
% compute the forward pass for a numeric pose

    % get transformation matrices
    R = pose_getRotationMatrix(pose(4:6));
    T = pose(1:3);
    
    % compute transformation
    V_CCF = R * V_TCF + T;
    l_CCF = K * V_CCF;

    % apply weighting
    if ~isempty(lamdbas)
        lB_CCF = l_CCF; % (vertices are actually boundary/reference points)
        l_CCF = lB_CCF(:,1:V_prop.nrBpoints) * lamdbas;
    end

    l_PCF = l_CCF(1:2,:)./l_CCF(3,:);
end


function [PZ_CCF,PZ_l_CCF,PZ_l_PCF] = aux_forwardPZ(V_prop, V_TCF, PZ_pose, K, PZ_lambdas)
% compute the forward pass for a polyZonotope pose

    % compute rotation matrix
    PZ_rot = project(PZ_pose,4:6);
    R_PZ = pose_getRotationMatrix(PZ_rot);

    nrPoints = size(V_TCF,2);

    % V_CCF
    PZ_CCF = exactPlus( ...
        matMap( ...
            R_PZ, ...
            polyZonotope(reshape(V_TCF, 3*nrPoints, 1)), ...
            3,3,nrPoints ...
        ), ...
        kron(ones(nrPoints,1),eye(3))*project(PZ_pose,1:3) ...
    );

    % l_CCF
    PZ_l_CCF = matMap( ...
        polyZonotope(reshape(K,9,1)), ...
        PZ_CCF, ...
        3,3,nrPoints ...
    );

    % apply weighting
    if ~isnumeric(PZ_lambdas)
        PZ_lb_CCF = PZ_l_CCF; % (vertices are actually boundary/reference)
        PZ_lb_1_CCF = project(PZ_l_CCF,1:3);
        assert(isempty(compact(PZ_lb_1_CCF).GI),'First reference point should be origin, which is mapped exactly.');
        PZ_lb_CCF = exactPlus(PZ_lb_CCF, -kron(ones(nrPoints,1),eye(3))*PZ_lb_1_CCF);
        PZ_l_CCF = matMap(PZ_lb_CCF,PZ_lambdas,3,3,V_prop.nrRVertices);
        PZ_l_CCF = exactPlus(PZ_l_CCF, +kron(ones(V_prop.nrRVertices,1),eye(3))*PZ_lb_1_CCF);
        PZ_l_CCF = compact(PZ_l_CCF);
    end
    
    % l_PCF ---
    % compute division of x,y and z
    PZ_l_CCF_xy = project(PZ_l_CCF,setdiff(1:3*V_prop.nrRVertices,3:3:3*V_prop.nrRVertices));
    PZ_l_CCF_z = project(PZ_l_CCF,3:3:3*V_prop.nrRVertices);

    % compute 1/z
    persistent options
    persistent invlayer
    if isempty(options)
        options = struct;
        options.nn.add_approx_error_to_GI = true;
        options = nnHelper.validateNNoptions(options);
        invlayer = neuralNetwork({nnActLayerFromHandle(@(x) 1./x,'inverse',inf)});
    end
    assert(all(interval(PZ_l_CCF_z).inf > 0), 'Target too close to camera; or, at least the outer approximation is.')
    PZ_l_CCF_z_inv = evaluate_(invlayer,PZ_l_CCF_z,options);
    PZ_l_CCF_z_inv = projectHighDim( ...
        PZ_l_CCF_z_inv, ...
        V_prop.nrRVertices*V_prop.nrRVertices, ...
        1:(V_prop.nrRVertices+1):V_prop.nrRVertices*V_prop.nrRVertices ...
    );
    
    % compute x,y * 1/z
    PZ_l_PCF = matMap( ...
        PZ_l_CCF_xy, ...
        PZ_l_CCF_z_inv, ...
        2,V_prop.nrRVertices,V_prop.nrRVertices ...
    );
   
end

% ------------------------------ END OF CODE ------------------------------
