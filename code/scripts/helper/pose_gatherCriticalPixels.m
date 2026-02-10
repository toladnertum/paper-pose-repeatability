function [critPixels,sfPixels] = pose_gatherCriticalPixels(I,C_struct,maxNoisyPixels)
% pose_gatherCriticalPixels - get critical pixels for given pose
%
% Syntax:
%    sfPixels = pose_gatherCriticalPixels(I,C_struct)
%
% Inputs:
%    I - logical, image, h x w
%    C_struct - struct of a pose
%    maxNoisyPixels - numeric, max number of noisy pixels
%
% Outputs:
%    critPixels - cell, critical pixels per vertex
%    sfPixels - numeric, support function pixels, 2 x numRVertices
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

% get image dimensions
[h,w] = size(I);
ws = 1:w;
hs = 1:h;

% gather all pixels
[ws,hs] = meshgrid(ws,hs);
pixels = [reshape(ws,1,[]);reshape(hs,1,[])];
pixels = pixels - 0.5; % center of pixel

% filter relevant pixels for each vertex
C_struct.V_PCF.invCPixel = reshape(C_struct.V_PCF.invCPixel,1,1,[]);
I_isContained = C_struct.V_PCF.isContained & I; % & (...
    ... % normal: filter 'turned on' pixels 
    % ~C_struct.V_PCF.invCPixel .* (C_struct.V_PCF.isContained == I)  ...
    ... % inverse: filter 'turned off' pixels 
    % | C_struct.V_PCF.invCPixel .* (C_struct.V_PCF.isContained == ~I) ...
%);

% find critical pixels via computed support function
pixelsSF = pagemtimes(C_struct.V_PCF.dirCPixel,pixels);
% filter not included pixels
pixelsSF(~I_isContained) = -inf; 
% check if vertex region extends over polygon
% (relevant for inverse direction as additional '0's appear "above" / 
% two dis-connected regions with "0")
% pixelsSF_sort = sort(pixelsSF,2);
% idxGap = ((pixelsSF_sort(:,2:end,:) - pixelsSF_sort(:,1:(end-1),:)) >= 2*sum(abs(C_struct.V_PCF.dirCPixel),2)) & ~isinf(pixelsSF_sort(:,1:(end-1),:));
% idxGap = cat(2,false(1,1,C_struct.V_prop.nrRVertices),idxGap);
% pixelsSF_sort(~idxGap) = inf;
% pixelsSF(pixelsSF >= min(pixelsSF_sort,[],2)) = -inf;
%
[~,idx] = max(pixelsSF,[],2);
sfPixels = pixels(:,idx);

% TODO: determine critical pixel uncertainty:
% check neighboring pixels (same row/col) with 1s

% DISABLED: for inverse computations, shift determined pixel according to SF dir
% dirCPixel_norm = squeeze(C_struct.V_PCF.dirCPixel) ./ vecnorm(squeeze(C_struct.V_PCF.dirCPixel));
% critPixels = critPixels + reshape(C_struct.V_PCF.invCPixel,1,[]) .* round(dirCPixel_norm);

% find critical pixels --

% set critical pixels for invalid indices to nan 
% (as determined offline, or if no valid pixel was found online; also disable inverse dirs)
iValid = 1:C_struct.V_prop.nrRVertices;
border = any((1:w) == [1;w]) | any((1:h) == [1;h])';

% determine valid and invalid indices
idxInvalid = ~C_struct.V_PCF.idxValid ...
    | reshape(any(I_isContained & border,1:2),[],1) ...
    | reshape(all(isinf(pixelsSF),1:2),[],1) ...
    | reshape(C_struct.V_PCF.invCPixel,[],1);
sfPixels(:,idxInvalid) = nan;
iValid = iValid(~idxInvalid);

% prepare critical pixels
critPixels=cell(C_struct.V_prop.nrRVertices,1);
critPixels(idxInvalid) = {nan(2,1)};

if maxNoisyPixels == 0
    % geometric considerations

    % read out quadrant via support func dir
    angleSF = atan2(C_struct.V_PCF.dirCPixel(:,2,:),C_struct.V_PCF.dirCPixel(:,1,:));
    angleSF = reshape(angleSF,[],1);
    quads = [
        pi/2 <= angleSF & angleSF <= pi, ...
        0 <= angleSF & angleSF < pi/2, ...
        -pi/2 <= angleSF & angleSF < 0, ...
        -pi <= angleSF & angleSF < -pi/2, ...
    ] * (1:4)';
    
    % construct masking in each quadrant (edge detector in resp. direction)
    M_nw = [0 0 1; 0 -1 0; 0 0 0];
    M_ne = [1 0 0; 0 -1 0; 0 0 0];
    M_se = [0 0 0; 0 -1 0; 1 0 0];
    M_sw = [0 0 0; 0 -1 0; 0 0 1];
    M = {M_nw,M_ne,M_se,M_sw};
    

    critPixels(~idxInvalid) = arrayfun(@(i) conv2(I_isContained(:,:,i),M{quads(i)},'same') < 0, iValid, 'UniformOutput', false);
    
    % make sure pixels are connected
    critPixels(~idxInvalid) = arrayfun(@(i) aux_filterCritPixels(idxInvalid(i),pixels,I,critPixels{i},C_struct.V_PCF.dirCPixel(:,:,i),sfPixels(:,i)), iValid, 'UniformOutput',false);
else
    % choose all turned-on pixels
    critPixels(~idxInvalid) = arrayfun(@(i) pixels(:,I_isContained(:,:,i)), iValid, 'UniformOutput', false);
end

end


% Auxiliary functions -----------------------------------------------------

function critPixels = aux_filterCritPixels(invalid,pixels,I,critPixels,dir,sfPixel)
    % determine critical pixels

    % quick exit: i is invalid
    if invalid
        critPixels = nan(2,1); return
    end
    % quick exit: sfPixel not in critPixels -> disable
    if ~any(all(sfPixel == pixels(:,critPixels),1),2)
        critPixels = nan(2,1); return
    end

    % quick exit: sfPixel is extreme point
    extreme = conv2(I,[0 1 0; 1 0 1; 0 1 0],"same") <= 1;
    idxSfPixel = all(pixels == sfPixel,1);
    if extreme(idxSfPixel)
        critPixels = sfPixel; return
    end
    idxSfPixel = reshape(idxSfPixel,size(critPixels));

    % filter all other extreme points (not sound!)
    % critPixels = critPixels & ~(extreme & ~idxSfPixel);

    % filter barely connected pixels
    ring = conv2(critPixels,[1 1 1; 1 0 1; 1 1 1],"same");
    corner = conv2(critPixels,[1 0 1; 0 0 0; 1 0 1],"same");
    critPixels = critPixels & ~(ring == 1 & corner == 1 & ~idxSfPixel);

    % select component containing sf pixel
    CC = bwconncomp(critPixels);
    idx = cellfun(@(list) any(all(pixels(:,list) == sfPixel,1),2), CC.PixelIdxList);
    critIdx = CC.PixelIdxList{idx};
    critPixels = pixels(:,critIdx);

    % filter all pixels where line to extreme critical pixels do not touch
    % support pixel
    [~,idx1] = max([-dir(2) dir(1)] * critPixels);
    critPixel_ext1 = critPixels(:,idx1);
    [~,idx2] = max([dir(2) -dir(1)] * critPixels);
    critPixel_ext2 = critPixels(:,idx2);

    sfpixel_corner = sfPixel - 0.5 * sign(dir');
    idx = arrayfun(@(i) ...
            contains( ...
                expandBoundaries( ...
                    polygon([critPixel_ext1,critPixels(:,i),critPixel_ext2]), ...
                    sqrt(2*0.5^2), 'JointType','square'), ...
                sfpixel_corner), ...
        1:size(critPixels,2));
    critPixels = critPixels(:,idx);
    
    % turn on neighboring pixels of sfPixel
    sfPixelNeighbor = conv2(idxSfPixel,[0 1 0; 1 0 1; 0 1 0],"same") & I;
    critPixels = unique([critPixels, pixels(:,sfPixelNeighbor)]','rows')';
end

% ------------------------------ END OF CODE ------------------------------
