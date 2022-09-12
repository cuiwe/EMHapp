function HeadModelMat = MakeLeadField(SourceGrid, ChannelMat, ProcessData, is_os)
    %% set current protocol
    gui_brainstorm('SetCurrentProtocol', bst_get('Protocol', ProcessData.subject_name));
    temp = bst_get('ProtocolSubjects');temp=temp.Subject;iSubject = find(contains({temp.Name}, ProcessData.subject_name));
    sSubject = bst_get('Subject', iSubject);
    
    %% Make OPTIONS
    % is_os: 0 -> localsphere; 1 -> 1-layer BEM; 2 -> 2-layer BEM; 3 -> 3-layer BEM
    if(nargin < 4)
        is_os = 0;
    end
    OPTIONS = bst_headmodeler();
    OPTIONS.GridLoc = SourceGrid;
    sMethod.Comment = 'Overlapping spheres (volume)';
    sMethod.HeadModelType = 'volume';
    sMethod.MEGMethod = 'os_meg';
    sMethod.EEGMethod = '';
    sMethod.ECOGMethod = '';
    sMethod.SEEGMethod = '';
    sMethod.SaveFile = 0;
    OPTIONS.Channel = ChannelMat.Channel;
    if isfield(ChannelMat, 'MegRefCoef')
        OPTIONS.MegRefCoef = ChannelMat.MegRefCoef;
    end
    OPTIONS.iMeg  = [good_channel(OPTIONS.Channel, [], 'MEG'), ...
                     good_channel(OPTIONS.Channel, [], 'MEG REF')];
    OPTIONS.iEeg  = [];OPTIONS.iEcog = []; OPTIONS.iSeeg = [];
    % get inner_skull
    inner_skull = sSubject.Surface(sSubject.iInnerSkull).FileName;
    if(~isempty(inner_skull))
        OPTIONS.InnerSkullFile = inner_skull;
    end
    if(is_os == 0)
        % localsphere
        sMethod.MEGMethod = 'os_meg';
        OPTIONS = struct_copy_fields(OPTIONS, sMethod, 1);
    else
        % bem
        sMethod.MEGMethod = 'openmeeg';
        OPTIONS = struct_copy_fields(OPTIONS, sMethod, 1);
        OPTIONS.BemFiles = {};
        OPTIONS.BemNames = {};
        OPTIONS.BemCond  = [];
        % Add the BEM layers definition to the OPTIONS structure
        % Get all the available layers: out -> in
        if ~isempty(sSubject.iScalp)
            OPTIONS.BemFiles{end+1} = file_fullpath(sSubject.Surface(sSubject.iScalp(1)).FileName);
            OPTIONS.BemNames{end+1} = 'Scalp';
            OPTIONS.BemCond(end+1)  = 1;
        end
        if ~isempty(sSubject.iOuterSkull)
            OPTIONS.BemFiles{end+1} = file_fullpath(sSubject.Surface(sSubject.iOuterSkull(1)).FileName);
            OPTIONS.BemNames{end+1} = 'Skull';
            OPTIONS.BemCond(end+1)  = 0.0125;
        end
        if ~isempty(sSubject.iInnerSkull)
            OPTIONS.BemFiles{end+1} = file_fullpath(sSubject.Surface(sSubject.iInnerSkull(1)).FileName);
            OPTIONS.BemNames{end+1} = 'Brain';
            OPTIONS.BemCond(end+1)  = 1;
        end
        OPTIONS.BemCond = [1 0.0125 1];
        OPTIONS.isAdjoint = 0;
        OPTIONS.isAdaptative = 1;
        OPTIONS.isSplit = 0;OPTIONS.SplitLength = 4000;
        % select layer (default inner_skull)
        OPTIONS.BemSelect = zeros(size(OPTIONS.BemCond));
        if(is_os == 3)
            OPTIONS.BemSelect(1:end) = 1;
        elseif(is_os == 2)
            OPTIONS.BemSelect(end-1:end) = 1;
        else
            OPTIONS.BemSelect(end) = 1;
        end
        OPTIONS.BemSelect = logical(OPTIONS.BemSelect);
        OPTIONS.BemFiles = OPTIONS.BemFiles(OPTIONS.BemSelect);
        OPTIONS.BemNames = OPTIONS.BemNames(OPTIONS.BemSelect);
        OPTIONS.BemCond  = OPTIONS.BemCond(OPTIONS.BemSelect);
    end
    
    %% COMPUTE HEADMODEL
    [OPTIONS, ~] = bst_headmodeler(OPTIONS);
    HeadModelMat = OPTIONS.HeadModelMat;
end