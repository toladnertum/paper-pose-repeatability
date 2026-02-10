function [lambdas, PZ_lambdas] = pose_getLambdasScaling(B, V, V_prop, I_scale)
% pose_getLambdasScaling - returns the scaling factors between B and V
%
% Syntax:
%    [lambdas, PZ_lambdas] = pose_getLambdasScaling(B, V, V_prop, I_scale)
%
% Inputs:
%    B - numeric, boundary/reference points
%    V - numeric, vertices
%    V_prop - struct
%    I_scale - interval, 2d
%
% Outputs:
%    lambdas - numeric
%    PZ_lambdas - polyZonotope
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: -

% ------------------------------ BEGIN CODE -------------------------------

% solve linear system of equations
lambdas = [B(1:2,1:3); ones(1, 3)] \ [V(1:2,:); ones(1, V_prop.nrVertices)];

% construct uncertainty lambda object
PZ_lambdas = polyZonotope( ...
    reshape([1;center(I_scale)].*lambdas(:,~V_prop.jumps),3*V_prop.nrRVertices,1), [ ...
        reshape([0;rad(I_scale(1));0].*lambdas(:,~V_prop.jumps),3*V_prop.nrRVertices,1), ...
        reshape([0;0;rad(I_scale(2))].*lambdas(:,~V_prop.jumps),3*V_prop.nrRVertices,1) ...
    ]);
PZ_lambdas = PZ_lambdas.replaceId([1;2],[7;8]);


end

% ------------------------------ END OF CODE ------------------------------
