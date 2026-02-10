function pose_printUncertainty(pose)
% pose_printUncertainty - prints uncertainty of given pose to command window
%
% Syntax:
%    pose_printUncertainty(pose)
%
% Inputs:
%    pose - numeric, contSet: 8-dim
%
% Outputs:
%    -
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: -

% ------------------------------ BEGIN CODE -------------------------------

% convert to interval
pose = interval(pose);

fprintf('Pose:\n')
fprintf('- Position: x=%s, y=%s, z=%s\n', ...
    aux_formatValue(pose(1)), aux_formatValue(pose(2)), aux_formatValue(pose(3)))
fprintf('- Angles:   θ_x=%s, θ_y=%s, θ_z=%s\n', ...
    aux_formatValue(pose(4)), aux_formatValue(pose(5)), aux_formatValue(pose(6)))
if dim(pose) == 8
    fprintf('- Scaling:  s_x=%s, s_y=%s\n', ...
        aux_formatValue(pose(7)), aux_formatValue(pose(8)))
end

end


% Auxiliary functions -----------------------------------------------------

function str = aux_formatValue(value)
    if rad(value) == 0
        str = sprintf('%g', center(value));
    else
        str = sprintf('[%g,%g]', value.inf, value.sup);
    end
end

% ------------------------------ END OF CODE ------------------------------

