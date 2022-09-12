function [ChannelMat, isSkip, strReport] = AlignChannel(ChannelMat, SurfaceMat, tolerance)
% CHANNEL_ALIGN_AUTO: Aligns the channels to the scalp using Polhemus points.
% MODIFIED with bst channel_align_auto
%
% USAGE:  [ChannelMat, R, T, isSkip, isUserCancel, strReport] = channel_align_auto(ChannelFile, ChannelMat=[], isWarning=1, isConfirm=1, tolerance=0)
%
% DESCRIPTION: 
%     Aligns the channels to the scalp using Polhemus points stored in channel structure.
%     We assume rough registration via the nasion (NAS), left preauricular (LPA) and right
%     preauricular (RPA) has already aligned the channels to the scalp. 
%     We then use the a Gauss-Newton algorithm to fine-tune that registration 
%     based on the "extra head points" representing the Polhemus data
%     The result will be that (new) ChannelMat.Loc = R * (old) ChannelMat.Loc + T and
%     similarly for the head points.
%
% INPUTS:
%     - ChannelMat  : If specified, do not read or write any information from/to ChannelFile
%     - tolerance   : Percentage of outliers head points, ignored in the final fit
%
% OUTPUTS:
%     - ChannelMat   : The same ChannelMat structure input in, with the head points and sensors rotated and translated to match the head points to the scalp.
%                      Returned value is [] if the registration was cancelled
%     - isSkip       : If 1, processing was skipped because there was not enough information in the file
%     - strReport    : Text description of the quality of the alignment (distance between headpoint and scalp surface)

if (nargin < 3) || isempty(tolerance)
    tolerance = 0;
end
R = [];T = [];
isSkip = 0;strReport = '';

%% ===== LOAD CHANNELS =====
HeadPoints = channel_get_headpoints(ChannelMat, 1, 1);
% Check number of HeadPoints
MIN_NPOINTS = 15;
if isempty(HeadPoints) || (length(HeadPoints.Label) < MIN_NPOINTS)
    % Warning
    disp('BST> Not enough digitized head points to perform automatic registration.');
    isSkip = 1;
    return;
end
% M x 3 matrix of head points
HP = double(HeadPoints.Loc');

%% ===== Check tolerance ======
if ~isempty(tolerance)
    % Number of points to remove
    nRemove = ceil(tolerance * size(HP,1));
    % Invalid tolerance value
    if (tolerance > 0.99) || (size(HP,1) - nRemove < MIN_NPOINTS)
        disp('BST> Invalid tolerance value or not enough head points left.');
        isSkip = 1;
        return
    end
end

%% ===== FIND OPTIMAL FIT =====
% Find best possible rigid transformation (rotation+translation)
[R,T,~,dist] = bst_meshfit(SurfaceMat.Vertices, SurfaceMat.Faces, HP);
% Remove outliers and fit again
if ~isempty(dist) && ~isempty(tolerance) && (tolerance > 0)
    % Sort points by distance to scalp
    [~, iSort] = sort(dist, 1, 'descend');
    iRemove = iSort(1:nRemove);
    % Remove from list of destination points
    HP(iRemove,:) = [];
    % Fit again
    [R,T,~,dist] = bst_meshfit(SurfaceMat.Vertices, SurfaceMat.Faces, HP);
else
    nRemove = 0;
end
% Current position cannot be optimized
if isempty(R)
    isSkip = 2;
    return;
end
% Distance between fitted points and reference surface
strReport = ['Distance between ' num2str(length(dist)) ' head points and head surface:' 10 ...
    ' |  Mean : ' sprintf('%4.1f', mean(dist) * 1000) ' mm  |  Distance >  3mm: ' sprintf('%3d points (%2d%%)\n', nnz(dist > 0.003), round(100*nnz(dist > 0.003)/length(dist))), ...
    ' |  Max  : ' sprintf('%4.1f', max(dist) * 1000)  ' mm  |  Distance >  5mm: ' sprintf('%3d points (%2d%%)\n', nnz(dist > 0.005), round(100*nnz(dist > 0.005)/length(dist))), ...
    ' |  Std  : ' sprintf('%4.1f', std(dist) * 1000)  ' mm  |  Distance > 10mm: ' sprintf('%3d points (%2d%%)\n', nnz(dist > 0.010), round(100*nnz(dist > 0.010)/length(dist))), ...
    ' |  Number of outlier points removed: ' sprintf('%d (%d%%)', nRemove, round(tolerance*100)), 10 ...
    ' |  Initial number of head points: ' num2str(size(HeadPoints.Loc,2))];

%% ===== ROTATE SENSORS AND HEADPOINTS =====
for i = 1:length(ChannelMat.Channel) 
    % Rotate and translate location of channel
    if ~isempty(ChannelMat.Channel(i).Loc) && ~all(ChannelMat.Channel(i).Loc(:) == 0)
        ChannelMat.Channel(i).Loc = R * ChannelMat.Channel(i).Loc + T * ones(1,size(ChannelMat.Channel(i).Loc, 2));
    end
    % Only rotate normal vector to channel
    if ~isempty(ChannelMat.Channel(i).Orient) && ~all(ChannelMat.Channel(i).Orient(:) == 0)
        ChannelMat.Channel(i).Orient = R * ChannelMat.Channel(i).Orient;
    end
end
% Rotate and translate head points
if isfield(ChannelMat, 'HeadPoints') && ~isempty(ChannelMat.HeadPoints) && ~isempty(ChannelMat.HeadPoints.Loc)
    ChannelMat.HeadPoints.Loc = R * ChannelMat.HeadPoints.Loc + ...
                                T * ones(1, size(ChannelMat.HeadPoints.Loc, 2));
end

%% ===== SAVE TRANSFORMATION =====
% Initialize fields
if ~isfield(ChannelMat, 'TransfEeg') || ~iscell(ChannelMat.TransfEeg)
    ChannelMat.TransfEeg = {};
end
if ~isfield(ChannelMat, 'TransfMeg') || ~iscell(ChannelMat.TransfMeg)
    ChannelMat.TransfMeg = {};
end
if ~isfield(ChannelMat, 'TransfMegLabels') || ~iscell(ChannelMat.TransfMegLabels) || (length(ChannelMat.TransfMeg) ~= length(ChannelMat.TransfMegLabels))
    ChannelMat.TransfMegLabels = cell(size(ChannelMat.TransfMeg));
end
if ~isfield(ChannelMat, 'TransfEegLabels') || ~iscell(ChannelMat.TransfEegLabels) || (length(ChannelMat.TransfEeg) ~= length(ChannelMat.TransfEegLabels))
    ChannelMat.TransfEegLabels = cell(size(ChannelMat.TransfEeg));
end
% Create [4,4] transform matrix
newtransf = eye(4);
newtransf(1:3,1:3) = R;
newtransf(1:3,4)   = T;
% Add a rotation/translation to the lists
ChannelMat.TransfMeg{end+1} = newtransf;
ChannelMat.TransfEeg{end+1} = newtransf;
% Add the comments
ChannelMat.TransfMegLabels{end+1} = 'refine registration: head points';
ChannelMat.TransfEegLabels{end+1} = 'refine registration: head points';
end
