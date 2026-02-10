function ax = pose_plotSetup(type,init,w,h,varargin)
% pose_plotSetup - helper with init and get plot axis
%
% Syntax:
%    ax = pose_plotSetup(type,init,varargin)
%
% Inputs:
%    type - str, 'all', 'V_TCF', 'V_CCF', 'l_CCF', 'l_PCF', 'I', 
%           'H_POS', 'H_ANGLES'
%    init - logical, whether to initialize the subplot
%    w,h - numeric, image resolution
%    varargin - additional inputs to subplot(n,m,i,varargin)
%
% Outputs:
%    ax - axis
%
% Other m-files required: none
% Subfunctions: none
% MAT-files required: none
%
% See also: -

% ------------------------------ BEGIN CODE -------------------------------

% parse input
if nargin < 2
    init = false;
end

switch upper(type)
    case 'ALL'
        if init
            figure;
            pose_plotSetup('V_TCF',init,w,h,varargin{:});
            pose_plotSetup('V_CCF',init,w,h,varargin{:});
            pose_plotSetup('l_CCF',init,w,h,varargin{:});
            pose_plotSetup('l_PCF',init,w,h,varargin{:});
            pose_plotSetup('I',init,w,h,varargin{:});
        end
        ax = gcf;

    case 'V_TCF'
        ax = subplot(1,5,1); 
        if init
            hold on; box on; title('1. Target (v_{TCF})'); axis equal;
        end

    case 'V_CCF'
        ax = subplot(1,5,2); 
        if init 
            hold on; grid on; box on; title('2. Camera Lens (v_{CCF})')
            xlabel('x'); ylabel('y'); zlabel('z'); 
        end

    case 'L_CCF'
        ax = subplot(1,5,3); 
        if init 
            hold on; grid on; box on; title('3. Camera Sensor (λ_{CCF})')
            xlabel('x'); ylabel('y'); zlabel('z'); 
        end

    case {'L_PCF','I'}

        if strcmpi(type,'L_PCF')
            subplot(1,5,4); 
            title('4. Image (λ_{PCF})')
        else
            subplot(1,5,5); 
            title('Final Image')
        end
        
        if init
            hold on; box on; axis equal; 
            xlim([0,w]); xticks([]); ylim([0,h]); yticks([])
            % pixel grid
            grid('minor'); ax = gca();
            ax.XAxis.MinorTickValues = min(0):1:max(w);
            ax.YAxis.MinorTickValues = min(0):1:max(h);
        end

    case 'H_POS'
        ax = gca();
        if init
            subplot(1,3,1); 
            hold on; box on; hold on;
            title('Hypercube: Position')
            xlabel('x'); ylabel('y'); zlabel('z');
            xlim([-1,1]); ylim([-1,1]); zlim([-1,1]);
            pose_plot(struct,'H',interval(-ones(3,1),ones(3,1)),1:3)
            enlargeAxis();
            view(-35, 30);
        end

    case 'H_ANGLES'
        ax = gca();
        if init
            subplot(1,3,2); 
            hold on; box on; hold on;
            title('Hypercube: Angles')
            xlabel('θ_x'); ylabel('θ_y'); zlabel('θ_z');
            xlim([-1,1]); ylim([-1,1]); zlim([-1,1]);
            pose_plot(struct,'H',interval(-ones(3,1),ones(3,1)),1:3)
            enlargeAxis();
            view(-35, 30);
        end


    otherwise
        throw(CORAerror('CORA:wrongValue','first',{'all','v_TCF','v_CCF','l_CCF','l_PCF','I'}))

end

% clear output
if nargout == 0
    clear ax
end

end


% Auxiliary functions -----------------------------------------------------

% ------------------------------ END OF CODE ------------------------------
