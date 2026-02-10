function errRatio = pose_computeErrorRatio(PZ_l_PCF)
% pose_computeErrorRatio - computes error ratio of the given PZ with
%    respect to the linearized term
%
% Syntax:
%    errRatio = pose_computeErrorRatio(PZ_l_PCF)
%
% Inputs:
%    PZ_l_PCF - polyZonotope
%
% Outputs:
%    errorRatio - numeric
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: -

% ------------------------------ BEGIN CODE -------------------------------

E = PZ_l_PCF.E;
idx = any(E == 1,1) & sum(E,1) == 1;
PZ_core = polyZonotope(0*PZ_l_PCF.c,PZ_l_PCF.G(:,idx),[],PZ_l_PCF.E(:,idx),PZ_l_PCF.id);
PZ_approx = exactPlus(PZ_l_PCF,-PZ_core);
Z_proj_approx = zonotope(PZ_approx);

rad_G = rad(interval(zonotope(PZ_core.c, PZ_core.G)));
rad_GI = rad(interval(zonotope(Z_proj_approx.c, Z_proj_approx.G)));
errRatio = mean(rad_GI ./ rad_G);

end

% ------------------------------ END OF CODE ------------------------------
