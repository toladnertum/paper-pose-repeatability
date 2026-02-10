function han = pose_plot(V_prop,type,S,dims,varargin)
% pose_plotSetup - helper with plotting sets in harmonized style
%
% Syntax:
%    han = pose_plot(type,S,dims,varargin)
%
% Inputs:
%    V_prop - struct
%    type - str, 'T', 'B','T_SAMPLE', 'CAM', 'CAM_DIR','POSE_DIR',
%    'I_POINTS', 'I_PGON', 'H', 'C', 'C_I'
%    S - numeric or contSet, to be plotted
%    dims - dimensions to plot
%    varargin - additional name-value pairs for plotting
%
% Outputs:
%    han - graphics
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: -

% ------------------------------ BEGIN CODE -------------------------------

switch class(S)

    case {'double'} % -----------------------------------------------------
        switch upper(type)
            case 'T' % target
                han = plotMultipleSetsAsOne( ...
                    aux_getTargetPolytopes(V_prop,S), dims, ...
                    [{'Color', CORAcolor('CORA:blue')}, varargin]);
           
            case 'B' % boundary/reference points
                han = plot(polytope(S), dims, ...
                    '--', 'Color', CORAcolor('CORA:red'), 'Filled', false, varargin{:});

            case 'T_SAMPLE' % target samples
                han = plotMultipleSetsAsOne( ...
                    aux_getTargetPolytopes(V_prop,S), dims, ...
                    [{'Color', CORAcolor('CORA:purple'), 'Filled', false}, varargin]);

                % camera and orientation directions ---
            case 'CAM'
                han = plotPoints(S,1:3,'ok',varargin{:});

            case 'CAM_DIR'
                CAMERA_DIR = [0;0;S(3)];
                han = plot(zonotope(CAMERA_DIR/2,CAMERA_DIR/2),1:3,'--k',varargin{:});

            case 'POSE_DIR'
                han = plot(zonotope(S(1:3)/2,S(1:3)/2),1:3,'--','Color',CORAcolor('CORA:blue'),varargin{:});
        
                % final image ---
            case 'I_POINTS'
                plotPoints(S,dims,'k',varargin{:})

            case 'I_PGON'
                % create polygon (expensive!)
                pgon = polygon();
                for i=1:size(S,2)
                    pgon = pgon | polygon(interval(floor(S(:,i)), ceil(S(:,i))));
                end
                plot(pgon,1:2,'k',varargin{:})

                % critical pixels
            case 'CRIT_PIXELS'
                plotPoints(S,dims,'x','Color',CORAcolor('CORA:yellow'),varargin{:})

            otherwise
                throw(CORAerror('CORA:wrongValue','second',{'T','B','T_SAMPLE','CAM','CAM_DIR','POSE_DIR','I_points','I_pgon'}))
        end

    case 'interval' % -----------------------------------------------------
        switch upper(type)
            case 'T' % target
                han = plotMultipleSetsAsOne( ...
                    arrayfun(@(i) S(:,i), 1:V_prop.nrRVertices,'UniformOutput',false), ...
                    dims, [{'--k', 'Filled', false}, varargin]);

            case 'B' % boundary/reference points
                han = plotMultipleSetsAsOne( ...
                    arrayfun(@(i) S(:,i), 1:V_prop.nrBpoints,'UniformOutput',false), ...
                    dims, [{'--k', 'Filled', false}, varargin]);

            case 'H' % hypercube
                han = plot(S,dims,'k','Filled',false,varargin{:});
        
            otherwise
                throw(CORAerror('CORA:wrongValue','second',{'T','B','H'}))
        end

    case {'polyZonotope','zonotope'} % ------------------------------------
        nrSplits = 0;
        switch upper(type)
            case 'T' % target
                han = plotMultipleSetsAsOne( ...
                    arrayfun(@(i) project(S, numel(dims)*(i-1) + dims), 1:V_prop.nrRVertices,'UniformOutput',false), ...
                    dims, [{'Color', CORAcolor('CORA:light-blue'),'Splits',nrSplits}, varargin]);

            case 'B' % boundary/reference points
                han = plotMultipleSetsAsOne( ...
                    arrayfun(@(i) project(S, numel(dims)*(i-1) + dims), 1:V_prop.nrBpoints,'UniformOutput',false), ...
                    dims, [{'Color', CORAcolor('CORA:yellow'),'Splits',nrSplits}, varargin]);
        
            otherwise
                throw(CORAerror('CORA:wrongValue','second',{'T','B'}))
        end

    case 'polytope' % -----------------------------------------------------
        switch upper(type)
            case 'C' % all constraint
                han = plot(S,dims,varargin{:},'Color',CORAcolor('CORA:red'));
            
            case 'C_I' % (list of) individual constraints; should be given as cell
                han = pose_plot(V_prop,type,{S},dims,varargin{:});

            case 'H' % hypercube should be an interval
                han = pose_plot(V_prop,type,interval(S),dims,varargin{:});
        
            otherwise
                throw(CORAerror('CORA:wrongValue','second',{'C','C_I','H'}))

        end

    case 'cell' % ---------------------------------------------------------
        switch upper(type)
            case 'C_I' % (list of) individual constraints
                han = plotMultipleSetsAsOne(S,dims,[{'Color',CORAcolor('CORA:green')} varargin]);

            case 'CRIT_PIXELS'
                arrayfun(@(i)plotPoints( S{i},dims,'o','Color',CORAcolor('CORA:yellow'),varargin{:}),1:numel(S));
        
            otherwise
                throw(CORAerror('CORA:wrongValue','second',{'C_I'}))
        end

    otherwise % -----------------------------------------------------------
        throw(CORAerror('CORA:wrongValue','third', ...
            {'numeric','interval','(poly)zonotope','polytope','cell'}))
end

% clear output
if nargout == 0
    clear han
end

end


% Auxiliary functions -----------------------------------------------------

function Ps = aux_getTargetPolytopes(V_prop,V)
    Ps = arrayfun(@(i) polytope(V(:,V_prop.convRegionsIdx{i})), 1:V_prop.nrConvRegions, 'UniformOutput',false);
end

% ------------------------------ END OF CODE ------------------------------
