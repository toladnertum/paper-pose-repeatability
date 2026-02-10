function [V_TCF, B_TCF, V_prop] = pose_getTarget(target, varargin)
% pose_getTarget - returns the vertices of the requested target
%
% Syntax:
%    V_TCF = pose_getTarget(target)
%    V_TCF = pose_getTarget(target, varargin)
%
% Inputs:
%    target - str, 'triangle', 'stop', 'taxi', 'runway', 'T', 'A', 'X', 'I'
%               '0', '3', '30', 'stripes', 'slowvehicle'
%    computeProps - logicals
%    scaling - numeric scalar
%
% Outputs:
%    V_TCF - vertices in target camera view
%    B_TCF - reference/boundary in target camera view
%    V_prop - struct, stores properties of V matrix
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: -

% ------------------------------ BEGIN CODE -------------------------------

% parse input
[computeProps, scaling] = setDefaultValues({true,1},varargin);

% init nans for complex objects
nans = nan(2,1);
P = [eye(2),[0;0]];

switch upper(target)
    case 'TRIANGLE'
        V_TCF = [
            0 0 1;
            0 1 1
        ];

    case 'STOP'
        V_TCF = [
            0 1 2 3 3 2 1 0;
            2 3 3 2 1 0 0 1;
        ] / 3;
        V_TCF = V_TCF - mean(V_TCF,2);
        V_TCF = [V_TCF nan(2,1) V_TCF*0.8];

    case 'TAXI'
        V_TCF_letter_T = P * pose_getTarget('T', false);
        V_TCF_letter_A = P * pose_getTarget('A', false);
        V_TCF_letter_X = P * pose_getTarget('X', false);
        V_TCF_letter_I = P * pose_getTarget('I', false);
        V_TCF = [V_TCF_letter_T, nans, [3;0] + V_TCF_letter_A, nans, [6.5;0] + V_TCF_letter_X, nans, [10;0] + V_TCF_letter_I];

    % letters ---
    case 'T'
        V_TCF = [
            0 3 3 2 2 1 1 0
            3 3 2 2 0 0 2 2
        ];
    case 'A'
        V_TCF = [
            0 1 2 3 2 1.5 1 nan 1   2   1.5
            0 3 3 0 0 1   0 nan 1.5 1.5 2  
        ];
    case 'X'
        V_TCF = [
            0 0 1   0 0 1 1.5 2 3 3 2   3 3 2 1.5 1
            0 1 1.5 2 3 3 2   3 3 2 1.5 1 0 0 1   0
        ];
    case 'I'
        V_TCF = [
            0 0 1 1
            0 3 3 0
        ];
    case '3'
        vthick = 0.8; % ?
        V_TCF = [
          0 0   3-vthick 3-vthick 3-vthick-1.4 3-vthick 3-vthick 0   0 3 3   1.6 3   3
          0 1.5 1.5      4.3      5.4          6.5      7.5      7.5 9 9 6.5 5.4 4.3 0
        ];
    case '0'
        V_TCF = [
          0 0 3 3 nan 0.8 2.2 2.2 0.8
          0 9 9 0 nan 1.5 1.5 7.5 7.5
        ];
    case '30'
        V_TCF_letter_3 = P * pose_getTarget('3', false);
        V_TCF_letter_0 = P * pose_getTarget('0', false);
        V_TCF = [V_TCF_letter_3 nans [3+2.2;0] + V_TCF_letter_0];
    case 'R'
        V_TCF = [
          0 0 3 3   2.2 3 2.2 1.4 0.8 0.8 nan 0.8 2.2 2.2 0.8
          0 9 9 3.7 3.7 0 0   3.7 3.7 0   nan 5.2 5.2 7.5 7.5
        ];
    case '|'
        V_TCF = [
            0  0  1.8 1.8
            0 30 30   0
        ];
    case 'STRIPES'
        a = 1.8;
        I = P * pose_getTarget('|',false);
        V_TCF = [
            [2*a;0] + I nans ...
            [0;0] + I nans ...
            [-2*a;0] + I ...
        ];
        % scale down to match letters
        V_TCF = 4.5/30 * V_TCF;
    case 'RUNWAY'
        a = 1.8;
        V_TCF_letter_30 = P * pose_getTarget('30',false);
        V_TCF_letter_R = P * pose_getTarget('R',false);
        I = P * pose_getTarget('|',false);
        V_TCF = [V_TCF_letter_30 nans [0;-15] + V_TCF_letter_R nans, ...
            [0;-46.5] + [...
                    % L
                    [-1.5*a;0] + I nans ...
                    [-3.5*a;0] + I nans ...
                    [-5.5*a;0] + I nans ...
                    [-7.5*a;0] + I nans ...
                    [-9.5*a;0] + I nans ...
                    ... % [-11.5*a;0] + I nans ...
                    ... % R
                    [1.5*a;0] + I nans ...
                    [3.5*a;0] + I nans ...
                    [5.5*a;0] + I nans ...
                    [7.5*a;0] + I nans ...
                    [9.5*a;0] + I nans ...
                    ... % [11.5*a;0] + I nans ...
            ] nans ...
            [0;31.5] + I nans [0;81.5] + I nans [0;131.5] + I nans ...
            [0;97.5] + [...
                    ... % L
                    [-10-0.5*a;0] + I nans ...
                    [-10-2.5*a;0] + I nans ...
                    [-10-4.5*a;0] + I nans ...
                    ... % R
                    [+10+0.5*a;0] + I nans ...
                    [+10+2.5*a;0] + I nans ...
                    [+10+4.5*a;0] + I nans ...
            ] ...
        ];
            
        % add runway boundary
        bmin = min(V_TCF(:,~any(isnan(V_TCF),1)),[],2);
        bmax = max(V_TCF(:,~any(isnan(V_TCF),1)),[],2);
        V_TCF = [V_TCF nans [
            bmin(1)-a bmin(1)-2*a bmin(1)-2*a bmin(1)-a 
            bmin(2)   bmin(2) bmax(2) bmax(2)     
        ] nans [
            bmax(1)+a bmax(1)+2*a bmax(1)+2*a bmax(1)+a 
            bmin(2)   bmin(2) bmax(2) bmax(2)     
        ]];

    case 'SLOWVEHICLE' % sign from yasser's paper
        x = 0; y = 0;
        div = 100;
        
        % Define vertices of the Hexagon
        hex = [ ...
            (x - 15)/div, (y + 25.93)/div; ...
            (x - 20)/div, (y + 34.59)/div; ...
            (x - 5)/div, (y + 60.31)/div; ...
            (x + 5)/div, (y + 60.31)/div; ...
            (x + 20)/div, (y + 34.59)/div; ...
            (x + 15)/div, (y + 25.93)/div; ...
        ]'; 

        % Define vertices of the Triangle
        % tri = [ ...
        %     (x + 0.001)/div, (y + 60.31)/div; ... 
        %     (x - 17.3879)/div, (y + 30.0659)/div; ...
        %     (x + 17.3879)/div, (y + 30.0659)/div; ...
        % ]';
        % place perfectly between vertices of hexagon
        tri = [
            mean(hex(:,1:2),2), mean(hex(:,3:4),2), mean(hex(:,5:6),2)
        ];

        % get final shape
        V_TCF = [hex nans tri];
        
    % unknown target ---
    otherwise 
        throw(CORAerror('CORA:wrongValue','first',sprintf('Unknown target ''%s''', target)))
end

% post process
V_TCF = aux_postProcessV(V_TCF, scaling);

% compute properties of V_TCF
if computeProps
    V_prop = aux_readPropertiesV(V_TCF);
end

% get boundary/reference points
B_TCF = [0 1 0; 0 0 1]; % important for target scaling
V_prop.nrBpoints = size(B_TCF,2);

% project into 3d
V_TCF = eye(3,2) * V_TCF;
B_TCF = eye(3,2) * B_TCF;

end


% Auxiliary functions -----------------------------------------------------

function V_TCF = aux_postProcessV(V_TCF, scaling)
    % center vertices
    B = interval.enclosePoints(V_TCF);
    % center both (just for visuals)
    c = center(B);
    V_TCF = (V_TCF - c);
    % scale
    V_TCF = scaling * V_TCF;
end

function V_prop = aux_readPropertiesV(V_TCF)
    % read out properties of selected target
    V_prop = struct;
    V_prop.nrVertices = size(V_TCF,2);

    % read out convex areas
    % get convex regions in TCF
    splits = splitIntoConvexSets(polygon(V_TCF));
    V_prop.nrConvRegions = numel(splits);
    % which vertices are part of which convex region?
    V_prop.convRegionsIdx = cellfun( ...
        @(split) contains(split, V_TCF(1:2,:)), ...
        splits, 'UniformOutput', false);
    % additional properties
    V_prop.jumps = all(isnan(V_TCF),1);
    V_prop.nrJumps = nnz(V_prop.jumps);
    V_prop.nrRVertices = V_prop.nrVertices - V_prop.nrJumps;

    % compute boundary (these vertices are always valid)
    lb = min(V_TCF(:,~V_prop.jumps),[],2);
    ub = max(V_TCF(:,~V_prop.jumps),[],2);

    alwaysValid = all(lb == V_TCF | [lb(1);ub(2)] == V_TCF ...
        | ub == V_TCF | [lb(2);ub(1)] == V_TCF);
    V_prop.alwaysValid = alwaysValid(:,~V_prop.jumps);
end

% ------------------------------ END OF CODE ------------------------------
