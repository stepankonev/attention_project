function IM = Rload(FNAMEFMT, decrease, ratio, varargin)
% Function loads z-stack of numerated images.
% - Parameter 'decrease' is a positive number, which means how much
%   the resolution of loading 2d-stack images will be decreased.
% - Parameter 'ratio' is a positive number, which means the change of
%   ratio x/z resolutions
% Returns [array] with 3 dimensions. 

% ---------------------------------------------------------------------
% Create a structure in directory
files = dir(FNAMEFMT);
NUM = length(files);
step = decrease/ratio;

% Reading files
TMP = imread(files(1).name);
TMP = TMP(:, :, 1);
if decrease ~= 1
    TMP = imresize(TMP, 1 / decrease,'method','nearest');
end
[M, N] = size(TMP);
IM = zeros(M, N, ceil(NUM / decrease));

% Parcing input
defaultShow = true;
p = inputParser;
addParameter(p,'show',defaultShow);%,@isstring);
parse(p,varargin{:});
show = p.Results.show;

% Loading files
if show == true
    nf=NUM/step;
    loading_bar = waitbar(0,strcat('Done: ',sprintf('%1.0f',0),'/',sprintf('%4.0f',nf)),...
                      'Name', 'mLoading files');
end;
for i = 1 : step : NUM
    if show == true
    waitbar(i/nf,loading_bar, strcat('Done: ',sprintf('%3.0f',i),'/',sprintf('%4.0f',nf)));
    end
    TMP = imread(files(round(i)).name);
    TMP = TMP(:, :, 1);
    if decrease ~= 1
        TMP = imresize(TMP, 1 / decrease,'method','nearest');
    end
    if round(i / step)== 0
        IM(:, :, 1 + round(i / step)) = double(TMP);
    else
        IM(:, :, 0 + round(i / step)) = double(TMP);
    end;
end
if show == true
delete(loading_bar);
end;
%imagesc(max(IM, [], 3)); axis image; colormap hot; colorbar;


end

