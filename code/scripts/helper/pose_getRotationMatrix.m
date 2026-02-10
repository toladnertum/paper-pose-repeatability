function R = pose_getRotationMatrix(pose_rot)
% pose_getRotationMatrix - computes the rotation matrix from the given pose
%
% Syntax:
%    R = pose_getRotationMatrix(pose_rot)
%
% Inputs:
%    pose_rot - rotation pose; numeric, interval, polyZonotope
%
% Outputs:
%    R - rotation matrix; numeric, interval, polyZonotope
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: -

% ------------------------------ BEGIN CODE -------------------------------

switch class(pose_rot)
    case {'double','interval'}
        R = aux_getRotationMatrixNumeric(pose_rot);

    case 'polyZonotope'
        R = aux_getRotationMatrixPZ(pose_rot);

    otherwise
        throw(CORAerror('CORA:wrongValue','first',{'numeric','interval','polyZonotope'}))
end

end


% Auxiliary functions -----------------------------------------------------

function R = aux_getRotationMatrixNumeric(pose_rot)
% compute rotation matrix using the given numeric pose
    
    % compute cosine
    c = cos(pose_rot);
    s = sin(pose_rot);

    % compute rotation matrix
    R = [
        c(3)*c(2), c(3)*s(2)*s(1)-s(3)*c(1), c(3)*s(2)*c(1)+s(3)*s(1);
        s(3)*c(2), s(3)*s(2)*s(1)+c(3)*c(1), s(3)*s(2)*c(1)-c(3)*s(1);
        -s(2),     c(2)*s(1),                c(2)*c(1)
    ];
end


function R_PZ = aux_getRotationMatrixPZ(PZ_rot)
% compute rotation matrix using the given polyZonotope pose

    % set up transformation matrix
    persistent options
    persistent coslayer
    persistent sinlayer
    if isempty(options)
        options = struct;
        options.nn.add_approx_error_to_GI = true;
        options = nnHelper.validateNNoptions(options);
        coslayer = neuralNetwork({nnActLayerFromHandle(@cos,'cos')});
        sinlayer = neuralNetwork({nnActLayerFromHandle(@sin,'sin')});
    end

    % evaluate cos and sin
    PZ_c = coslayer.evaluate_(PZ_rot,options);
    PZ_s = sinlayer.evaluate_(PZ_rot,options);

    % compute rotation matrix
    R_z = exactPlus( ...
        [1;0;0; 0;1;0; 0;0;0] * project(PZ_c,3), ...
        [0;1;0; -1;0;0; 0;0;0] * project(PZ_s,3) ...
    ) + [0;0;0; 0;0;0; 0;0;1];
    R_y = exactPlus( ...
        [1;0;0; 0;0;0; 0;0;1] * project(PZ_c,2), ...
        [0;0;-1; 0;0;0; 1;0;0] * project(PZ_s,2) ...
    ) + [0;0;0; 0;1;0; 0;0;0];
    R_x = exactPlus( ...
        [0;0;0; 0;1;0; 0;0;1] * project(PZ_c,1), ...
        [0;0;0; 0;0;1; 0;-1;0] * project(PZ_s,1) ...
    ) + [1;0;0; 0;0;0; 0;0;0];
    R_PZ = matMap( matMap(R_z, R_y, 3,3,3), R_x, 3,3,3);
    R_PZ = compact(R_PZ);
end

% ------------------------------ END OF CODE ------------------------------
