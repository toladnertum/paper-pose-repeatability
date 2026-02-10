function [isContained,I_pixels,isContaineds] = pose_getContainedPixels(w,h,convRegions)
% pose_getContainedPixels - checks which pixels are turned on in an image
%
% Syntax:
%    [isContained,pixels] = pose_getContainedPixels(w,h,convRegions)
%
% Inputs:
%    w,h - numeric, image resolution
%    convRegions - cell of polytopes
%
% Outputs:
%    isContained - logical, w x h
%    I_pixels - numeric, contained pixels
%    isContained - logical, w x h x numRegions
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: -

% ------------------------------ BEGIN CODE -------------------------------

% check containment
ws = 1:w;
hs = 1:h;
[ws,hs] = meshgrid(ws,hs);
pixels = [reshape(ws,1,[]);reshape(hs,1,[])];
pixels = pixels - 0.5;
% frame = polygon(interval([0;0],[w;h]));
% pixelError = interval(-0.5*ones(2,1),0.5*ones(2,1));
convRegions = cellfun(@(convRegion) polygon(convRegion),convRegions,'UniformOutput',false);

isContaineds = cell(numel(convRegions),1);
for i=1:numel(convRegions)
    convRegion = convRegions{i};
    % expand region to compensate for partially intersecting pixels
    % (expands orthogonal to edge, 
    % so at most sqrt(0.5^2*2) to reach pixel center)
    convRegionExpanded = expandBoundaries(convRegion,sqrt(0.5^2*2),'JointType','square');
    idxContainedPixels = contains(convRegionExpanded,pixels);
    % invConvRegion = subtract(frame,convRegionExpanded);
    % idxNotContainedPixels = contains(invConvRegion,pixels);
    % idxUnknownPixels = ~idxContainedPixels & ~idxNotContainedPixels;
    % unknownPixels = pixels(:,idxUnknownPixels);
    % 
    % idxContainedPixels(:,idxUnknownPixels) = arrayfun(@(i) ...
    %         isIntersecting(unknownPixels(:,i)+pixelError,convRegion), ...
    %     1:size(unknownPixels,2));
    isContaineds{i} = idxContainedPixels;
end
isContaineds = cell2mat(isContaineds);
isContained = any(isContaineds,1);
I_pixels = pixels(:,isContained);

% prepare for output
isContained = reshape(isContained,h,w);
isContaineds = reshape(isContaineds',h,w,[]);

end


% Auxiliary functions -----------------------------------------------------


% ------------------------------ END OF CODE ------------------------------
