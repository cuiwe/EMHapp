function bstChannel = MakeSourceModel(MRI, VoxelSize, bstSeg, sInner, Posterior_Fossa_Remove) 
    if(nargin < 3)
        bstSeg = [];
    end
    if(nargin < 4)
        sInner = [];
    end
    if(nargin < 5 || isempty(Posterior_Fossa_Remove))
        Posterior_Fossa_Remove = 1;
    end
    %% =======Make Source Model=======
    % Covert MRI
    ftMri = out_fieldtrip_mri(MRI,'anatomy');
    ftMri = ft_convert_units(ftMri,'m');
    % Segment Brain tissue
    if(~isempty(bstSeg))
        % load from bst
        seg = removefields(ftMri, 'anatomy');
        % generate gray
        seg.gray = double(bstSeg.Cube);
        seg.gray(seg.gray ~= bstSeg.Labels{cellfun(@(x)strcmp(x, 'Gray'), bstSeg.Labels(:, 2)), 1}) = 0;
        seg.gray(seg.gray ~= 0) = 1;
        % generate white
        seg.white = double(bstSeg.Cube);
        seg.white(seg.white ~= bstSeg.Labels{cellfun(@(x)strcmp(x, 'White'), bstSeg.Labels(:, 2)), 1}) = 0;
        seg.white(seg.white ~= 0) = 1;
        % generate csf
        seg.csf = double(bstSeg.Cube);
        seg.csf(seg.csf ~= bstSeg.Labels{cellfun(@(x)strcmp(x, 'CSF'), bstSeg.Labels(:, 2)), 1}) = 0;
        seg.csf(seg.csf ~= 0) = 1;
        % generate bone;
        seg.bone = double(bstSeg.Cube);
        seg.bone(seg.bone ~= bstSeg.Labels{cellfun(@(x)strcmp(x, 'Skull'), bstSeg.Labels(:, 2)), 1}) = 0;
        seg.bone(seg.bone ~= 0) = 1;
        % generate softtissue;
        seg.softtissue = double(bstSeg.Cube);
        seg.softtissue(seg.softtissue ~= bstSeg.Labels{cellfun(@(x)strcmp(x, 'Scalp'), bstSeg.Labels(:, 2)), 1}) = 0;
        seg.softtissue(seg.softtissue ~= 0) = 1;
        % generate air;
        seg.air = double(bstSeg.Cube);
        seg.air(seg.air == 0) = 100;seg.air(seg.air ~= 100) = 0;seg.air(seg.air == 100) = 1;
    else
        % generate using ft
        cfg = [];
        cfg.tissue = 'brain';
        cfg.spmversion = 'spm12';
        cfg.spmmethod = 'new';
        seg = ft_volumesegment(cfg, ftMri);
    end
    % Mesh Brain tissue
    cfg = [];
    cfg.tissue = 'brain';
    cfg.spmversion = 'spm12';
    brain_mesh = ft_prepare_mesh(cfg, seg);
    % Build SourceModel (Use segmetation results, only grid belonging to gray matter)
    cfg = [];
    cfg.spmversion = 'spm12';
    cfg.resolution = VoxelSize;
    cfg.mri = seg;
    cfg.unit = 'm'; 
    ftSourcemodel = ft_prepare_sourcemodel(cfg);
    % exclude grid outsied innerskull
    if(~isempty(sInner))
        iOutside = find(~inpolyhd(ftSourcemodel.pos, sInner.Vertices, sInner.Faces));
        ftSourcemodel.inside(iOutside) = 0;
    end
    
    %% =======VS location Location Name=======
    Channel={};
    % Get transformation from MRI to MNI
    ftPath = which('ft_defaults');ftPath = bst_fileparts(ftPath);
    addpath(genpath(fullfile(ftPath,'external/spm12')));
    cfg = [];
    cfg.spmversion = 'spm12';
    cfg.nonlinear = 'yes';
    cfg.template = fullfile(ftPath,'external/spm8/templates/T1.nii');
    cfg.write = 'no';
    MRI2MNI = ft_volumenormalise(cfg, ftMri);
    % Transfomr grid to MNI
    ftSourcemodel = ft_convert_units(ftSourcemodel,'mm');
    mnipos=ft_warp_apply(MRI2MNI.params,ft_warp_apply(MRI2MNI.initial, ftSourcemodel.pos),'individual2sn');
    ftSourcemodelMNI = ftSourcemodel;
    ftSourcemodelMNI.pos = mnipos;
    % Get ALL lable for gird
    aalAtlas = ft_read_atlas(fullfile(ftPath,'template/atlas/aal/ROI_MNI_V4.nii'));
    se=strel('cube',5);aalAtlas.tissue=imdilate(aalAtlas.tissue,se);
    aalAtlas.tissuelabel = {'Left-Temporal','Left-Posterior_Fossa','Left-Insula/Cingulate','Left-Frontal','Left-Occipital','Left-Parietal','Left-Central',...
                            'Right-Temporal','Right-Posterior_Fossa','Right-Insula/Cingulate','Right-Frontal','Right-Occipital','Right-Parietal','Right-Central'};
    aalAtlas.tissue(ismember(aalAtlas.tissue,[37,39,41,55,79,81,83,85,87,89]))=1+1000;
    aalAtlas.tissue(ismember(aalAtlas.tissue,[38,40,42,56,80,82,84,86,88,90]))=length(aalAtlas.tissuelabel)/2+1+1000;
    aalAtlas.tissue(ismember(aalAtlas.tissue,[91,93,95,97,99,101,103,105,107]))=2+1000;
    aalAtlas.tissue(ismember(aalAtlas.tissue,[92,94,96,98,100,102,104,106,108]))=length(aalAtlas.tissuelabel)/2+2+1000;
    aalAtlas.tissue(ismember(aalAtlas.tissue,[29,31,33,35]))=3+1000;
    aalAtlas.tissue(ismember(aalAtlas.tissue,[30,32,34,36]))=length(aalAtlas.tissuelabel)/2+3+1000;
    aalAtlas.tissue(ismember(aalAtlas.tissue,[1,3,5,7,9,11,13,15,17,19,21,23,25,27,69]))=4+1000;
    aalAtlas.tissue(ismember(aalAtlas.tissue,[2,4,6,8,10,12,14,16,18,20,22,24,26,28,70]))=length(aalAtlas.tissuelabel)/2+4+1000;
    aalAtlas.tissue(ismember(aalAtlas.tissue,[43,45,47,49,51,53]))=5+1000;
    aalAtlas.tissue(ismember(aalAtlas.tissue,[44,46,48,50,52,54]))=length(aalAtlas.tissuelabel)/2+5+1000;
    aalAtlas.tissue(ismember(aalAtlas.tissue,[57,59,61,63,65,67]))=6+1000;
    aalAtlas.tissue(ismember(aalAtlas.tissue,[58,60,62,64,66,68]))=length(aalAtlas.tissuelabel)/2+6+1000;
    aalAtlas.tissue(ismember(aalAtlas.tissue,[71,73,75,77]))=7+1000;
    aalAtlas.tissue(ismember(aalAtlas.tissue,[72,74,76,78]))=length(aalAtlas.tissuelabel)/2+7+1000;
    aalAtlas.tissue(ismember(aalAtlas.tissue,[109,110,111,112,113,114,115,116]))=0;
    aalAtlas.tissue(aalAtlas.tissue~=0) = aalAtlas.tissue(aalAtlas.tissue~=0)-1000; 
    % Remove Posterior_Fossa
    if(Posterior_Fossa_Remove == 1)
        Posterior_Fossa = [find(aalAtlas.tissue==2);find(aalAtlas.tissue==length(aalAtlas.tissuelabel)/2+2)];
        aalAtlas.tissue(Posterior_Fossa)=0;
        aalAtlas.tissue(aalAtlas.tissue>length(aalAtlas.tissuelabel)/2+2)=aalAtlas.tissue(aalAtlas.tissue>length(aalAtlas.tissuelabel)/2+2)-1;
        aalAtlas.tissue(aalAtlas.tissue>2)=aalAtlas.tissue(aalAtlas.tissue>2)-1;
        aalAtlas.tissuelabel([2,length(aalAtlas.tissuelabel)/2+2])=[];
    end 
    % Align Label
    cfg = [];
    cfg.interpmethod = 'nearest';
    cfg.parameter = 'tissue';
    ftSourcemodelNew = ft_sourceinterpolate(cfg, aalAtlas, ftSourcemodelMNI);
    Label = aalAtlas.tissuelabel(ftSourcemodelNew.tissue(ftSourcemodel.inside==1&ftSourcemodelNew.tissue(:)~=0));
    Idx = find(ftSourcemodel.inside==1&ftSourcemodelNew.tissue(:)~=0);
    Channel(:,1) = num2cell(find(ftSourcemodel.inside));
    uniqueLable = unique(Label);
    for i=1:length(uniqueLable)
        temp = contains(Label,uniqueLable(i));
        Channel(ismember([Channel{:,1}],Idx(temp)),2) = cellfun(@(x,y)strcat(x(x<'Z'& x>'A'),'-',num2str(y)),Label(temp),num2cell(1:length(Label(temp))),'UniformOutput',false);
    end
    Channel(ismember([Channel{:,1}],Idx),3) =  Label;
    Channel(cellfun(@(x)isempty(x),Channel(:,3)),:) = [];
    ftSourcemodel = ft_convert_units(ftSourcemodel, 'm');
    % Save VS channel as bst mat
    bstChannel=db_template('channelmat');
    for chan=1:size(Channel,1)
        bstChannel.Channel(chan).Name = Channel{chan,2};
        bstChannel.Channel(chan).Comment = [];
        bstChannel.Channel(chan).Type = 'ECOG';
        bstChannel.Channel(chan).Group = Channel{chan,3};
        bstChannel.Channel(chan).Loc = ftSourcemodel.pos(Channel{chan,1},:)';
        bstChannel.Channel(chan).Orient = [];
        bstChannel.Channel(chan).Weight = [];   
    end
    bstChannel.Comment = ['SEEG/ECOG (',num2str(size(Channel,1)),')'];  
end