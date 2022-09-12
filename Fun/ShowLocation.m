function ShowLocation(varargin)
% BST_COLORMAPS: Show source locations.
% 
% USAGE:
%  ShowLocation('CreatFigure', FigureName, ShowResult, MRI, Surface, Position)
%  ShowLocation('UpdateFigure', ShowResult)
    eval(macro_method);
end

function CreatFigure(FigureName, ShowResult, MRI, Surface, Position) %#ok<*DEFNU>
    global hdl
    % initial Param
    hdl.rand_num = rand(1);
    SurfAlpha = 0.75;DataThres = 0.8;DataAlpha=0.5;SurfSmooth = 0;ResAlpha = 0.6;
    % Add Path
    PrivateFigure3dPath = which('Main.mlapp');
    addpath(genpath([fileparts(fileparts(PrivateFigure3dPath)),'/ExternalFun']));
    % Get Param
    hdl.ColorLimit = [0.6 1];
    Protocols = load(bst_get('BrainstormDbFile'));Protocol = bst_get('ProtocolInfo');
    hdl.ProtocolFile = load(fullfile(Protocols.ProtocolsListInfo(bst_get('iProtocol')).STUDIES,'protocol.mat'));
    hdl.Loc = ShowResult.Loc;hdl.AmpMap = ShowResult.AmpMap;hdl.ChanNumMap = ShowResult.ChanNumMap;hdl.Map = hdl.AmpMap;
    hdl.MRI = MRI;hdl.Surface = Surface;
    hdl.MRI_Mat = load(fullfile(Protocol.SUBJECTS, hdl.MRI));
    hdl.ProtocolPath = fileparts(fileparts(strrep(file_fullpath(hdl.MRI), hdl.MRI, '')));
    hdl.FigureName = FigureName;
    hdl.SeegChannelMat = [];
    % Add surf and Show Results
    ShowSurface(DataThres, SurfSmooth, SurfAlpha, Position);
    % set surface figure on the top
    jFrame = get(hdl.Fig,'JavaFrame');drawnow;
    jFrame.fHG2Client.getWindow.setAlwaysOnTop(true);
    Thres = round(peak2peak(double(hdl.Map)) * min(max(DataThres, 0.01), 0.99) * 100) / 100;
    % Add pannel
    hdl.Pannel_ShowParam = uipanel('Parent',hdl.Fig,'Position',[0.75 0.00 0.25 0.50],'Units','normalized');
    hdl.Pannel_Resection = uipanel('Parent',hdl.Fig,'Position',[0.75 0.50 0.25 0.15],'Units','normalized');
    hdl.Pannel_SEEG = uipanel('Parent',hdl.Fig,'Position',[0.75 0.65 0.25 0.35],'Units','normalized');
    % Pannel_Resection: Get Resection Anat
    Item = {hdl.ProtocolFile.ProtocolSubjects(1).Subject.Anatomy.Comment};
    Item = cat(2, 'None', Item);
    hdl.Pop_Res = uicontrol('Style', 'popup','String', Item,'Parent',hdl.Pannel_Resection,'Units','normalized','Position', [0 0.34 1 0.6]);
    hdl.Pop_Res.Callback = @Callback_PopRes;
    % Pannel_Resection: Resection Trans
    hdl.ResAlphaFactor = uicontrol('Style','text','Parent',hdl.Pannel_Resection,'Units','normalized','Position',[0 0.3 1 0.3],'String',['Resection Transp: ', num2str(ResAlpha)],'FontWeight','bold');
    hdl.Slider_ResAlpha = uicontrol('Style','Slider','Parent',hdl.Pannel_Resection,'Units','normalized','Position',[0.03 0 0.94 0.3],'Value',0);
    set(hdl.Slider_ResAlpha,'Max',1,'Min',0,'Value',ResAlpha,'SliderStep',[0.05, 0.1]);
    hdl.Slider_ResAlpha.Callback = @Callback_ResAlpha; addlistener(hdl.Slider_ResAlpha, 'Value', 'PostSet',@Callback_ResAlpha);
    % Pannel_SEEG
    Item = cat(2, hdl.ProtocolFile.ProtocolStudies(1).Study.Condition);
    Item = cat(2, 'None',Item(contains(Item,'Implantation')),Item(~contains(Item,'Implantation')));
    hdl.Pop_SEEG = uicontrol('Style', 'popup','String', Item,'Parent',hdl.Pannel_SEEG,'Units','normalized','Position', [0 0.85 1 0.1],'Min',1,'Max',length(Item));
    hdl.List_SEEG = uicontrol('Style','listbox','Parent',hdl.Pannel_SEEG,'Units','normalized','String',{},'position',[0.05 0.05 0.90 0.73]);   
    hdl.Pop_SEEG.Callback = @Callback_PopSEEG;
    hdl.List_SEEG.Callback = @Callback_ListSEEG;
    % Pannel_ShowParam: Show VS Num or Amp    
    hdl.ShowAmp = uicontrol('Style','radiobutton','Parent',hdl.Pannel_ShowParam,'Units','normalized','Position',[0 0.8700 0.4 0.125],'String','Amp','FontWeight','bold','Value', 1);
    hdl.ShowChannel = uicontrol('Style','radiobutton','Parent',hdl.Pannel_ShowParam,'Units','normalized','Position',[0.4 0.8700 0.6 0.125],'String','Numbers','FontWeight','bold','Value', 0);
    hdl.ShowAmp.Callback = @Callback_ShowAmp;hdl.ShowChannel.Callback = @Callback_ShowChannel;
    % Pannel_ShowParam: Overlap Threshold
    hdl.ResultsThreshold = uicontrol('Style','text','Parent',hdl.Pannel_ShowParam,'Units','normalized','Position',[0 0.920 - 0.125 1 0.075],'String',['Overlap Thres: ', num2str(Thres)],'FontWeight','bold');
    hdl.Slider_ResultsThreshold = uicontrol('Style','Slider','Parent',hdl.Pannel_ShowParam,'Units','normalized','Position',[0.03 0.845 - 0.125 0.94 0.075],'Value',1);
    set(hdl.Slider_ResultsThreshold,'Max',1,'Min',0,'Value',DataThres,'SliderStep',[0.05, 0.1]);
    hdl.Slider_ResultsThreshold.Callback = @CallFun_ResultsThreshold;addlistener(hdl.Slider_ResultsThreshold, 'Value', 'PostSet',@CallFun_ResultsThreshold);
    % Pannel_ShowParam: Surface Trans    
    hdl.SurfaceAlphaFactor = uicontrol('Style','text','Parent',hdl.Pannel_ShowParam,'Units','normalized','Position',[0 0.765 - 0.125 1 0.075],'String',['Surface Transp: ', num2str(round(SurfAlpha*100)/100)],'FontWeight','bold');
    hdl.Slider_SurfaceAlpha = uicontrol('Style','Slider','Parent',hdl.Pannel_ShowParam,'Units','normalized','Position',[0.03 0.690 - 0.125 0.94 0.075],'Value',0);
    set(hdl.Slider_SurfaceAlpha,'Max',1,'Min',0,'Value',SurfAlpha,'SliderStep',[0.05, 0.1]);
    hdl.Slider_SurfaceAlpha.Callback = @Callback_SurfaceAlpha; addlistener(hdl.Slider_SurfaceAlpha, 'Value', 'PostSet',@Callback_SurfaceAlpha);
    % Pannel_ShowParam: Overlap Trans
    hdl.HFOAlphaFactor = uicontrol('Style','text','Parent',hdl.Pannel_ShowParam,'Units','normalized','Position',[0 0.610 - 0.125 1 0.075],'String',['Overlap Transp: ', num2str(round(DataAlpha*100)/100)],'FontWeight','bold');
    hdl.Slider_HFOAlpha = uicontrol('Style','Slider','Parent',hdl.Pannel_ShowParam,'Units','normalized','Position',[0.03 0.535 - 0.125 0.94 0.075],'Value',0);
    set(hdl.Slider_HFOAlpha,'Max',1,'Min',0,'Value',DataAlpha,'SliderStep',[0.05, 0.1]);
    hdl.Slider_HFOAlpha.Callback = @Callback_HFOAlpha; addlistener(hdl.Slider_HFOAlpha, 'Value', 'PostSet',@Callback_HFOAlpha);
    % Pannel_ShowParam: Surface Smooth  
    hdl.SurfaceSmoothFactor = uicontrol('Style','text','Parent',hdl.Pannel_ShowParam,'Units','normalized','Position',[0 0.455 - 0.125 1 0.075],'String',['Surface Smooth: ', num2str(round(SurfSmooth*100)/100)],'FontWeight','bold');
    hdl.Slider_SurfaceSmooth = uicontrol('Style','Slider','Parent',hdl.Pannel_ShowParam,'Units','normalized','Position',[0.03 0.380 - 0.125 0.94 0.075],'Value',0);
    set(hdl.Slider_SurfaceSmooth,'Max',1,'Min',0,'Value',SurfSmooth,'SliderStep',[0.05, 0.1]);
    hdl.Slider_SurfaceSmooth.Callback = @Callback_SurfaceSmooth; addlistener(hdl.Slider_SurfaceSmooth, 'Value', 'PostSet',@Callback_SurfaceSmooth);
    % Pannel_ShowParam: Show Left or Right hemi  
    hdl.ShowLeft = uicontrol('Style','togglebutton','Parent',hdl.Pannel_ShowParam,'Units','normalized','Position',[0 0.115 0.5 0.125],'String','Left','FontWeight','bold','Value', 1);
    hdl.ShowRight = uicontrol('Style','togglebutton','Parent',hdl.Pannel_ShowParam,'Units','normalized','Position',[0.5 0.115 0.5 0.125],'String','Right','FontWeight','bold','Value', 1);
    hdl.ShowLeft.Callback = @Callback_ShowLeft;hdl.ShowRight.Callback = @Callback_ShowRight;
    % Pannel_ShowParam: Show MRI
    hdl.ShowMRI = uicontrol('Style','pushbutton','Parent',hdl.Pannel_ShowParam,'Units','normalized','Position',[0 0.0 1 0.125],'String','Show MRI','FontWeight','bold');
    hdl.ShowMRI.Callback = @Callback_ShowMRI;
    
    % Label Size
    hdl.Fig.SizeChangedFcn = @CallFunSizeChange;
    hdl.Fig.CloseRequestFcn = @CallFunClose;
    
    CallFunSizeChange();
end

function UpdateFigure(ShowResult, FigureName)
    global hdl
    if(isfield(hdl, 'Fig')&&isvalid(hdl.Fig))
        hdl.Loc = ShowResult.Loc;hdl.AmpMap = ShowResult.AmpMap;hdl.ChanNumMap = ShowResult.ChanNumMap;hdl.Map = hdl.AmpMap;
        % update mri 
        if(isfield(hdl, 'hFig_MRI')&&isvalid(hdl.hFig_MRI))
            Callback_ShowMRI();
        end
        % update surface figure
        Callback_ShowAmp();
        CallFun_ResultsThreshold();
        Callback_HFOAlpha();
        % update figure name
        if(nargin == 2)
           hdl.FigureName = FigureName;
           if(isfield(hdl, 'hFig_MRI')&&isvalid(hdl.hFig_MRI))
              hdl.hFig_MRI.Name = hdl.FigureName;
           end
           set(hdl.Fig,'Name',hdl.FigureName,'NumberTitle', 'off'); 
        end
    end
end

function CallFunClose(~, ~ , ~)
    global GlobalData hdl
    if(isfield(hdl, 'Channel_MRIDir')&&exist(hdl.Channel_MRIDir, 'file'))
        delete(hdl.Channel_MRIDir)
    end
    if(exist(fullfile(hdl.ProtocolPath, 'data', hdl.ProtocolFile.ProtocolSubjects(1).Subject.Name, 'Temp'), 'file'))
       rmdir(fullfile(hdl.ProtocolPath, 'data', hdl.ProtocolFile.ProtocolSubjects(1).Subject.Name, 'Temp'), 's'); 
    end
    if(strcmp(GlobalData.DataBase.ProtocolStudies(GlobalData.DataBase.iProtocol).Study(end).Name, 'Temp'))
        GlobalData.DataBase.ProtocolStudies(GlobalData.DataBase.iProtocol).Study(end) = [];
    end
    if(isfield(hdl, 'MRI_TempDir')&&exist(hdl.MRI_TempDir, 'file'))
        delete(hdl.MRI_TempDir)
    end
    if(isfield(hdl, 'hFig_MRI')&&isvalid(hdl.hFig_MRI))
        bst_figures('DeleteFigure', hdl.hFig_MRI);
    end
    bst_figures('DeleteFigure', hdl.Fig);
    PrivateFigure3dPath = which('Main.mlapp');
    rmpath(genpath([fileparts(fileparts(PrivateFigure3dPath)),'/ExternalFun']));
end

function CallFunSizeChange(~,~,~)
    global hdl
    AxesPos = hdl.axes.Position(3)*hdl.Fig.Position(3);hdl.LabelSize = round(AxesPos/20);
    set(hdl.axes.Children(ismember(hdl.axes.Children.get('Tag'),'VS')),'SizeData',hdl.LabelSize*100);
    set(hdl.axes.Children(ismember(hdl.axes.Children.get('Tag'),'SEEG')),'MarkerSize',hdl.LabelSize);
    set(hdl.axes.Children(ismember(hdl.axes.Children.get('Tag'),'SEEGi')),'LineWidth',hdl.LabelSize/8);
    temp = hdl.axes.Children(contains(hdl.axes.Children.get('Type'),'text'));
    set(temp(ismember(temp.get('Tag'),'SEEGiTag')),'FontSize',hdl.LabelSize/1.5);
    set(temp(ismember(temp.get('Tag'),'SEEGTag')),'FontSize',hdl.LabelSize/4);
    set(hdl.hbar,'FontSize',hdl.LabelSize/2);
end

function Callback_ResAlpha(~, ~, ~)
    global hdl
    TessInfo = getappdata(hdl.Fig, 'Surface');iSurface = find(cellfun(@(x)strcmp(x, 'Other'),  {TessInfo.Name}), 1);
    if(~isempty(iSurface))
        Transp = hdl.Slider_ResAlpha.Value;Transp = round(Transp*100)/100;
        set(hdl.ResAlphaFactor,'String',['Resection Transp: ',num2str(Transp)]);
        TessInfo = getappdata(hdl.Fig, 'Surface');iSurface = find(cellfun(@(x)strcmp(x, 'Other'),  {TessInfo.Name}));
        TessInfo(iSurface).SurfAlpha = Transp;setappdata(hdl.Fig, 'Surface', TessInfo);
        figure_3d('UpdateSurfaceAlpha', hdl.Fig, iSurface);
    else
        hdl.Slider_ResAlpha.Value = 0;
    end
end

function Callback_PopRes(~, ~, ~)
    global hdl;
    % Remove Previous Resection
    TessInfo = getappdata(hdl.Fig, 'Surface');iSurface = find(cellfun(@(x)strcmp(x, 'Other'),  {TessInfo.Name}));
    if(~isempty(iSurface))
        TessInfo(iSurface) = [];setappdata(hdl.Fig, 'Surface', TessInfo);
        Temp = findobj(hdl.axes, 'Tag', 'AnatSurface'); delete(Temp(1));
    end
    if(~strcmp(hdl.Pop_Res.String{hdl.Pop_Res.Value}, 'None'))
        % Shwo Resection
        iSubject = 1;Protocol = bst_get('ProtocolInfo');
        Resection = load(fullfile(Protocol.SUBJECTS,hdl.ProtocolFile.ProtocolSubjects(iSubject).Subject.Anatomy(cellfun(@(x)strcmp(x, hdl.Pop_Res.String{hdl.Pop_Res.Value}), {hdl.ProtocolFile.ProtocolSubjects(iSubject).Subject.Anatomy.Comment})).FileName));
        Resection.Cube = Resection.Cube > 0.5;
        TessMat = in_tess_mrimask(Resection, 0);
        TessMat.Vertices = cs_convert(hdl.MRI_Mat, 'mri', 'scs', TessMat.Vertices);
        view_surface_matrix(TessMat.Vertices, TessMat.Faces, 0.5, [0 1 0], hdl.Fig);
        % Smooth
        TessInfo = getappdata(hdl.Fig, 'Surface'); iSurface = find(cellfun(@(x)strcmp(x, 'Other'),  {TessInfo.Name}));
        TessInfo(iSurface).SurfSmoothValue=1;setappdata(hdl.Fig, 'Surface', TessInfo);
        figure_3d('SmoothSurface', hdl.Fig, iSurface, 1);
        % Tran
        Transp = hdl.Slider_ResAlpha.Value;Transp = round(Transp*100)/100;
        TessInfo = getappdata(hdl.Fig, 'Surface');iSurface = find(cellfun(@(x)strcmp(x, 'Other'),  {TessInfo.Name}));
        TessInfo(iSurface).SurfAlpha = Transp;setappdata(hdl.Fig, 'Surface', TessInfo);
        figure_3d('UpdateSurfaceAlpha', hdl.Fig, iSurface);
    end
end

function Callback_PopSEEG(~,~,~)
    global hdl GlobalData
    Value = hdl.Pop_SEEG.Value;
    % Get SeegChannelMat
    iStudy = cellfun(@(x)strcmp(hdl.Pop_SEEG.String{Value},x),{hdl.ProtocolFile.ProtocolStudies(1).Study.Condition});
    if(any(iStudy))
        Protocol = bst_get('ProtocolInfo');hdl.SeegChannelMat = load(fullfile(Protocol.STUDIES,hdl.ProtocolFile.ProtocolStudies(1).Study(iStudy).Channel.FileName));
        channel = cellfun(@(x)[x,' '],{hdl.SeegChannelMat.Channel.Name},'UniformOutput',0);
        set(hdl.List_SEEG,'String',sort(channel),'Max',length(channel),'Value',1);
    else
        hdl.SeegChannelMat = 0;
        set(hdl.List_SEEG,'String',{},'Max',1,'Value',1);
    end
    if(strcmp(hdl.Pop_SEEG.String{Value}, 'None'))
        [~, iFig, iDS] = bst_figures('GetFigure', hdl.Fig);
        GlobalData.DataSet(iDS).Figure(iFig).SelectedChannels = [];
        GlobalData.DataSet(iDS).Channel = [];
        GlobalData.DataSet(iDS).IntraElectrodes = [];
        delete(findobj(hdl.Fig, 'Tag', 'ElectrodeGrid'));
        delete(findobj(hdl.Fig, 'Tag', 'ElectrodeSelect'));
        delete(findobj(hdl.Fig, 'Tag', 'ElectrodeDepth'));
        delete(findobj(hdl.Fig, 'Tag', 'ElectrodeWire'));
        delete(findobj(hdl.Fig, 'Tag', 'ElectrodeLabel'));
    end
end

function Callback_ListSEEG(~,~,~)
    global hdl GlobalData
    Value = hdl.Pop_SEEG.Value;
    if(~isempty(hdl.SeegChannelMat)&&isfield(hdl.SeegChannelMat, 'IntraElectrodes')&&~isempty(hdl.SeegChannelMat.IntraElectrodes))
        % Get SEEG Location
        iStudy = cellfun(@(x)strcmp(hdl.Pop_SEEG.String{Value},x),{hdl.ProtocolFile.ProtocolStudies(1).Study.Condition});
        Protocol = bst_get('ProtocolInfo');hdl.SeegChannelMat = load(fullfile(Protocol.STUDIES,hdl.ProtocolFile.ProtocolStudies(1).Study(iStudy).Channel.FileName));  
        Value = hdl.List_SEEG.Value;channel = sort(cellfun(@(x)[x,' '],{hdl.SeegChannelMat.Channel.Name},'UniformOutput',0));Value = cellfun(@(x)find(ismember({hdl.SeegChannelMat.Channel.Name},x(1:end-1))),channel(Value));
        
%         delete(hdl.axes.Children(contains(hdl.axes.Children.get('Tag'),'SEEG')));
%         delete(hdl.axes.Children(contains(hdl.axes.Children.get('Type'),'text')));
%         Value = hdl.List_SEEG.Value;channel = sort(cellfun(@(x)[x,' '],{hdl.SeegChannelMat.Channel.Name},'UniformOutput',0));Value = cellfun(@(x)find(ismember({hdl.SeegChannelMat.Channel.Name},x(1:end-1))),channel(Value));
%         Pos = cat(2,hdl.SeegChannelMat.Channel(Value).Loc);Name = {hdl.SeegChannelMat.Channel(Value).Name};
%         hold(hdl.axes,'on');plot3(hdl.axes,Pos(1,:),Pos(2,:),Pos(3,:),'.g','MarkerSize',hdl.LabelSize * 2,'Tag','SEEG');
%         if(~isempty(hdl.SeegChannelMat.IntraElectrodes))
%             cellfun(@(x)plot3(hdl.axes,x(1,:),x(2,:),x(3,:),'b','LineWidth',hdl.LabelSize/8,'Tag','SEEGi'),{hdl.SeegChannelMat.IntraElectrodes(cellfun(@(x)any(ismember(unique({hdl.SeegChannelMat.Channel(Value).Group}),x)),{hdl.SeegChannelMat.IntraElectrodes.Name})).Loc});hold(hdl.axes,'off');
%             Group = unique({hdl.SeegChannelMat.Channel(Value).Group},'stable');
%             temp = {hdl.SeegChannelMat.IntraElectrodes(cellfun(@(x)any(ismember(unique({hdl.SeegChannelMat.Channel(Value).Group}),x)),{hdl.SeegChannelMat.IntraElectrodes.Name})).Loc};
%             temp = cell2mat(cellfun(@(x)x(:,2),temp,'UniformOutput',false));
%             text(hdl.axes,temp(1,:),temp(2,:),temp(3,:)+0.002,Group,'color','w','FontSize',hdl.LabelSize/1.5,'FontWeight','bold','Tag','SEEGiTag');
%         end
%         text(hdl.axes,Pos(1,:),Pos(2,:),Pos(3,:)+0.001,Name,'color','w','FontSize',hdl.LabelSize/4,'FontWeight','bold','Tag','SEEGTag');
        
        % Show SEEG
        temp = hdl.SeegChannelMat;temp.IntraElectrodes = temp.IntraElectrodes(cellfun(@(x)any(contains(unique({temp.Channel(Value).Group}),x)),{temp.IntraElectrodes.Name}));temp.Channel = temp.Channel(Value);
        [~, iFig, iDS] = bst_figures('GetFigure', hdl.Fig);
        GlobalData.DataSet(iDS).Figure(iFig).SelectedChannels = 1:length(Value);
        GlobalData.DataSet(iDS).Channel = temp.Channel;
        GlobalData.DataSet(iDS).IntraElectrodes = temp.IntraElectrodes;
        TempDir = ['Channel_',num2str(round(hdl.rand_num*100000)),'.mat']; 
        hdl.Channel_MRIDir = fullfile(hdl.ProtocolPath, 'data', TempDir);
        save(fullfile(hdl.ProtocolPath, 'data', TempDir),'-struct','temp');
        view_channels(hdl.Channel_MRIDir, 'SEEG', 1, 0, hdl.Fig, 1);
        % Updata VS Colorbar due to the Change of CData in Axes Children
        if(~isempty(hdl.SeegChannelMat)&&~isempty(findobj(hdl.Fig, '-depth', 3, 'Tag', 'ElectrodeGrid')))
            VS_Data = findobj(hdl.Fig, '-depth', 3, 'Tag', 'VS');VS_Data = [min(VS_Data.CData), max(VS_Data.CData)];
            Grid_Data = findobj(hdl.Fig, '-depth', 3, 'Tag', 'ElectrodeGrid');Grid_Data = [min(Grid_Data.CData(:)), max(Grid_Data.CData(:))];
            Surf_Data = findobj(hdl.Fig, '-depth', 3, 'Tag', 'AnatSurface');Surf_Data =  cell2mat(cellfun(@(x)[min(x.CData(:)), max(x.CData(:))],num2cell(Surf_Data),'UniformOutput',false));
            hdl.ColorLimit = [max(min(min(VS_Data(1), Grid_Data(1)), Surf_Data(1)), 0.01), max(max(VS_Data(2), Grid_Data(2)), Surf_Data(2))];
            % Update VS Data 
            Map = double(hdl.Map);Loc = hdl.Loc'; 
            Thres = round(peak2peak(hdl.Map) * min(max(hdl.Slider_ResultsThreshold.Value, 0.01), 0.99) * 100) / 100;
            delete(hdl.axes.Children(contains(hdl.axes.Children.get('Tag'),'VS')));
            hdl.axes = findobj(hdl.Fig, '-depth', 2, 'tag', 'Axes3D');AxesPos = hdl.axes.Position(3)*hdl.Fig.Position(3);hdl.LabelSize = round(AxesPos/20);
            Loc = hdl.Loc';Map = double(hdl.Map);
            Loc(:,Map < double(Thres)) = [];Map(Map<double(Thres)) = [];
            % Make Sure VS in Whole range of Data in Axes
            Map = (double(Map) - min(double(Map))) / (max(double(Map)) - min(double(Map))) * hdl.ColorLimit(2) + hdl.ColorLimit(1) + 0.1;        
            hold(hdl.axes,'on');scatter3(hdl.axes,Loc(1,:),Loc(2,:),Loc(3,:),hdl.LabelSize * 100,Map,'.','Tag','VS');hold(hdl.axes,'off');
            % Update Colorbar
            hPan = findobj(hdl.Fig, '-depth', 1, 'Tag', 'SurfPan');
            delete(hPan.Children(contains(hPan.Children.get('Tag'),'Colorbar')));
            hPan = findobj(hdl.Fig, '-depth', 1, 'Tag', 'SurfPan');
            hdl.hbar = colorbar(hdl.axes, 'Units','normalized','position',[0.89 0.25 0.03 0.5],'Color',[1,1,1],'FontWeight','bold','Parent',hPan);colormap(hdl.axes, hot);
            set(hdl.hbar,'FontSize',hdl.LabelSize / 2,'FontWeight','bold');
            set(hdl.hbar,'Ticks',(min(Map) : peak2peak(Map) / 5 : max(Map)));
            set(hdl.hbar,'TickLabels',cellfun(@(x)num2str(x),num2cell(round(100 * (double(Thres) : (double(max(hdl.Map)) - double(Thres)) / 5 : double(max(hdl.Map)))) / 100),'UniformOutput',false));
            set(hdl.hbar,'Limit',[min(Map), max(Map)]);
        end
        % Updata MRI
        if(~isempty(hdl.SeegChannelMat) && (isfield(hdl, 'hFig_MRI')&&isvalid(hdl.hFig_MRI)))
            Value = hdl.List_SEEG.Value;channel = sort(cellfun(@(x)[x,' '],{hdl.SeegChannelMat.Channel.Name},'UniformOutput',0));Value = cellfun(@(x)find(ismember({hdl.SeegChannelMat.Channel.Name},x(1:end-1))),channel(Value));
            if(~isempty(Value))
                % Remove Previous SEEG
                [~, iFig, iDS] = bst_figures('GetFigure', hdl.hFig_MRI);
                GlobalData.DataSet(iDS).Figure(iFig).Handles = figure_mri('PlotElectrodes', iDS, iFig, GlobalData.DataSet(iDS).Figure(iFig).Handles, 1);
                figure_mri('PlotSensors3D', iDS, iFig);       
                temp = hdl.SeegChannelMat;temp.IntraElectrodes = temp.IntraElectrodes(cellfun(@(x)any(contains(unique({temp.Channel(Value).Group}),x)),{temp.IntraElectrodes.Name}));temp.Channel = temp.Channel(Value);
                TempDir = ['Channel_',num2str(round(hdl.rand_num*100000)),'.mat']; 
                hdl.Channel_MRIDir = fullfile(hdl.ProtocolPath, 'data', TempDir);
                save(fullfile(hdl.ProtocolPath, 'data', TempDir),'-struct','temp');
                GlobalData.DataSet(iDS).ChannelFile = [];
                figure_mri('LoadElectrodes', hdl.hFig_MRI, hdl.Channel_MRIDir, 'SEEG');
            end
        end
    end
end

function Callback_SurfaceAlpha(~,~,~)
    global hdl
    Transp = hdl.Slider_SurfaceAlpha.Value;Transp = round(Transp*100)/100;
    set(hdl.SurfaceAlphaFactor,'String',['Surface Transp: ',num2str(Transp)]);
    TessInfo = getappdata(hdl.Fig, 'Surface');iSurface = find(cellfun(@(x)strcmp(x, 'Cortex'),  {TessInfo.Name}));
    TessInfo(iSurface).SurfAlpha = Transp;setappdata(hdl.Fig, 'Surface', TessInfo);
    figure_3d('UpdateSurfaceAlpha', hdl.Fig, iSurface);
end

function Callback_HFOAlpha(~,~,~)
    global hdl
    Transp = hdl.Slider_HFOAlpha.Value;Transp = round(Transp*100)/100;
    set(hdl.HFOAlphaFactor,'String',['Overlay Transp: ',num2str(Transp)]);
    % Updata MRI Threshold
    if(isfield(hdl, 'hFig_MRI')&&isvalid(hdl.hFig_MRI))
        TessInfo = getappdata(hdl.hFig_MRI, 'Surface'); iSurface = getappdata(hdl.hFig_MRI, 'iSurface');
        TessInfo(iSurface).DataAlpha = Transp;setappdata(hdl.hFig_MRI, 'Surface', TessInfo);
        figure_mri('UpdateSurfaceColor', hdl.hFig_MRI, iSurface);
    end
end

function Callback_SurfaceSmooth(~,~,~)
    global hdl
    Smooth = hdl.Slider_SurfaceSmooth.Value;Smooth = round(Smooth*100)/100;
    set(hdl.SurfaceSmoothFactor,'String',['Surface Smooth: ',num2str(Smooth)]);
    TessInfo = getappdata(hdl.Fig, 'Surface');iSurface = find(cellfun(@(x)strcmp(x, 'Cortex'),  {TessInfo.Name}));
    TessInfo(iSurface).SurfSmoothValue=round(Smooth*100)/100;setappdata(hdl.Fig, 'Surface', TessInfo);
    figure_3d('SmoothSurface', hdl.Fig, iSurface, Smooth);
end

function Callback_ShowAmp(~,~,~)
    global hdl
    if(hdl.ShowAmp.Value)
       hdl.ShowChannel.Value = 0;
       hdl.Map = hdl.AmpMap;
    else
       hdl.ShowChannel.Value = 1;
       hdl.Map = hdl.ChanNumMap;
    end
    ShowSurface(min(max(hdl.Slider_ResultsThreshold.Value, 0.01), 0.99), min(max(hdl.Slider_SurfaceSmooth.Value, 0.01), 0.99), ...
                min(max(hdl.Slider_SurfaceAlpha.Value, 0.01), 0.99));
end

function Callback_ShowChannel(~,~,~)
    global hdl
    if(hdl.ShowChannel.Value)
        hdl.ShowAmp.Value = 0;
        hdl.Map = hdl.ChanNumMap;
    else
        hdl.ShowAmp.Value = 1;
        hdl.Map = hdl.AmpMap;
    end
    ShowSurface(min(max(hdl.Slider_ResultsThreshold.Value, 0.01), 0.99), min(max(hdl.Slider_SurfaceSmooth.Value, 0.01), 0.99), ...
                min(max(hdl.Slider_SurfaceAlpha.Value, 0.01), 0.99));
end

function Callback_ShowLeft(~,~,~)
    global hdl
    TessInfo = getappdata(hdl.Fig, 'Surface');iSurface = find(cellfun(@(x)strcmp(x, 'Cortex'),  {TessInfo.Name}));
    if(hdl.ShowLeft.Value && hdl.ShowRight.Value)
        SelectHemispheres(hdl.Fig, TessInfo, iSurface, 'none');
    elseif(hdl.ShowLeft.Value && ~hdl.ShowRight.Value)
        SelectHemispheres(hdl.Fig, TessInfo, iSurface, 'left');
    elseif(~hdl.ShowLeft.Value && hdl.ShowRight.Value)
        SelectHemispheres(hdl.Fig, TessInfo, iSurface, 'right');
    else
        hdl.ShowLeft.Value = 1;hdl.ShowRight.Value = 1;
        SelectHemispheres(hdl.Fig, TessInfo, iSurface, 'none');
    end
end


function Callback_ShowRight(~,~,~)
    global hdl
    TessInfo = getappdata(hdl.Fig, 'Surface');iSurface = find(cellfun(@(x)strcmp(x, 'Cortex'),  {TessInfo.Name}));
    if(hdl.ShowLeft.Value && hdl.ShowRight.Value)
        SelectHemispheres(hdl.Fig, TessInfo, iSurface, 'none');
    elseif(hdl.ShowLeft.Value && ~hdl.ShowRight.Value)
        SelectHemispheres(hdl.Fig, TessInfo, iSurface, 'left');
    elseif(~hdl.ShowLeft.Value && hdl.ShowRight.Value)
        SelectHemispheres(hdl.Fig, TessInfo, iSurface, 'right');
    else
        hdl.ShowLeft.Value = 1;hdl.ShowRight.Value = 1;
        SelectHemispheres(hdl.Fig, TessInfo, iSurface, 'none');
    end
end

function SelectHemispheres(hFig, TessInfo, iSurf, name)
    % Update surface Resect field
    TessInfo(iSurf).Resect = name;
    setappdata(hFig, 'Surface', TessInfo);
    % Update surface display
    figure_3d('UpdateSurfaceAlpha', hFig, iSurf);
end

function CallFun_ResultsThreshold(~,~,~)
    global hdl
    hPan = findobj(hdl.Fig, '-depth', 1, 'Tag', 'SurfPan');
    if(~isempty(hdl.Loc)&&((peak2peak(double(hdl.Map))>=1 && ~hdl.ShowAmp.Value) || hdl.ShowAmp.Value))       
        Map = double(hdl.Map);Loc = hdl.Loc'; 
        Thres = round(peak2peak(hdl.Map) * min(max(hdl.Slider_ResultsThreshold.Value, 0.01), 0.99) * 100) / 100;
        Loc(:,Map < double(Thres)) = [];Map(Map<double(Thres)) = [];
        % Make Sure VS in Whole range of Data in Axes
        Map = (double(Map) - min(double(Map))) / (max(double(Map)) - min(double(Map))) * hdl.ColorLimit(2) + hdl.ColorLimit(1) + 0.1; 
        delete(hdl.axes.Children(contains(hdl.axes.Children.get('Tag'),'VS')));
        delete(hPan.Children(contains(hPan.Children.get('Tag'),'Colorbar')));
        hold(hdl.axes,'on');scatter3(hdl.axes,Loc(1,:),Loc(2,:),Loc(3,:),hdl.LabelSize * 100,Map,'.','Tag','VS');hold(hdl.axes,'off');
        hdl.hbar = colorbar(hdl.axes, 'Units','normalized','position',[0.89 0.25 0.03 0.5],'Color',[1,1,1],'FontWeight','bold','Parent',hPan);colormap(hdl.axes, hot);
        set(hdl.hbar,'FontSize',hdl.LabelSize / 2,'FontWeight','bold');
        set(hdl.hbar,'Ticks',(min(Map) : peak2peak(Map) / 5 : max(Map)));
        set(hdl.hbar,'TickLabels',cellfun(@(x)num2str(x),num2cell(round(100 * (double(Thres) : (double(max(hdl.Map)) - double(Thres)) / 5 : double(max(hdl.Map)))) / 100),'UniformOutput',false));
        set(hdl.hbar,'Limit',[min(Map), max(Map)]);
        % Updata MRI Threshold
        if(isfield(hdl, 'hFig_MRI')&&isvalid(hdl.hFig_MRI))
            TessInfo = getappdata(hdl.hFig_MRI, 'Surface');
            iSurface = getappdata(hdl.hFig_MRI, 'iSurface');
            TessInfo(iSurface).DataThreshold = 0;
            TessInfo(iSurface).DataMinMax = [double(Thres), max(double(hdl.Map))];
            TessInfo(iSurface).DataLimitValue = [double(Thres), max(double(hdl.Map))];
            TessInfo(iSurface).Data = double(hdl.Map);TessInfo(iSurface).Data(double(hdl.Map)<double(Thres)) = 0;
            % Update current surface
            setappdata(hdl.hFig_MRI, 'Surface', TessInfo);
            FigureId = getappdata(hdl.hFig_MRI, 'FigureId');
            figure_mri('UpdateSurfaceColor', hdl.hFig_MRI, iSurface);
            % Modify Color bar
            hColorbar = findobj(hdl.hFig_MRI, '-depth', 1, 'Tag', 'Colorbar');
            set(hColorbar,'YTickLabels',cellfun(@(x)num2str(round(x*100)/100),num2cell(double(Thres):(1 - double(Thres)) / (length(hColorbar.YTick) - 1):1),'UniformOutput',false));
        end
    else
        Thres = peak2peak(double(hdl.Map));
    end
    set(hdl.ResultsThreshold,'String',['Overlap Thres: ',num2str(Thres)]);
end

function Callback_ShowMRI(~,~,~)
    global hdl GlobalData
    Protocols = load(bst_get('BrainstormDbFile'));
    Thres = round(peak2peak(hdl.Map) * min(max(hdl.Slider_ResultsThreshold.Value, 0.01), 0.99) * 100) / 100;
    Map = double(hdl.Map);Loc = hdl.Loc;
    % Get Mri and modify GlobalData
    TempDir = fullfile(hdl.ProtocolFile.ProtocolSubjects(1).Subject.Name, 'Temp',['results',num2str(round(hdl.rand_num*100000)),'.mat']); 
    % Modify GlobalData
    if(~strcmp(GlobalData.DataBase.ProtocolStudies(GlobalData.DataBase.iProtocol).Study(end).Name, 'Temp'))
        GlobalData.DataBase.ProtocolStudies(GlobalData.DataBase.iProtocol).Study = cat(2, GlobalData.DataBase.ProtocolStudies(GlobalData.DataBase.iProtocol).Study, GlobalData.DataBase.ProtocolStudies(GlobalData.DataBase.iProtocol).Study(end));
        GlobalData.DataBase.ProtocolStudies(GlobalData.DataBase.iProtocol).Study(end).Name = 'Temp';
        GlobalData.DataBase.ProtocolStudies(GlobalData.DataBase.iProtocol).Study(end).FileName = fullfile(hdl.ProtocolFile.ProtocolSubjects(1).Subject.Name, 'Temp', 'brainstormstudy.mat');
        mkdir(fullfile(hdl.ProtocolPath, 'data', hdl.ProtocolFile.ProtocolSubjects(1).Subject.Name, 'Temp'));
        copyfile(fullfile(hdl.ProtocolPath, 'data', GlobalData.DataBase.ProtocolStudies(GlobalData.DataBase.iProtocol).Study(end-1).FileName), ...
                 fullfile(hdl.ProtocolPath, 'data', hdl.ProtocolFile.ProtocolSubjects(1).Subject.Name, 'Temp', 'brainstormstudy.mat'));
        Temp = load(fullfile(hdl.ProtocolPath, 'data', hdl.ProtocolFile.ProtocolSubjects(1).Subject.Name, 'Temp', 'brainstormstudy.mat'));
        Temp.Name = 'Temp';save(fullfile(hdl.ProtocolPath, 'data', hdl.ProtocolFile.ProtocolSubjects(1).Subject.Name, 'Temp', 'brainstormstudy.mat'),'-struct','Temp');
    end
    GlobalData.DataBase.ProtocolStudies(GlobalData.DataBase.iProtocol).Study(end).Result(length(GlobalData.DataBase.ProtocolStudies(GlobalData.DataBase.iProtocol).Study(end).Data) + 1).FileName = TempDir;
    hdl.Reaults_MRIDir = fullfile(hdl.ProtocolPath, 'data', TempDir);
    % Get Result Mat
    results = db_template('resultsmat');
    results.ImageGridAmp =  Map;results.Time = 0;results.HeadModelType = 'volume';results.GridLoc = Loc;results.SurfaceFile = hdl.Surface;
    save(fullfile(hdl.ProtocolPath, 'data', TempDir),'-struct', 'results');
    % Show
    if(~isfield(hdl, 'hFig_MRI') || ~isvalid(hdl.hFig_MRI))
        [hdl.hFig_MRI, ~, ~] = view_mri(hdl.MRI, TempDir, 'MEG', 1);hdl.hFig_MRI.Name = hdl.FigureName;
        hdl.hFig_MRI.Position(3) = hdl.Fig.Position(3);
        hdl.hFig_MRI.Position(4) = hdl.Fig.Position(4);
        hdl.hFig_MRI.Position(1) = hdl.Fig.Position(1);
        hdl.hFig_MRI.Position(2) = hdl.Fig.Position(2) - hdl.Fig.Position(4)*1.07;
    else
        [hdl.hFig_MRI, ~, ~] = view_mri(hdl.MRI, TempDir, 'MEG', 0);hdl.hFig_MRI.Name = hdl.FigureName;
    end 
    % Update current surface
    TessInfo = getappdata(hdl.hFig_MRI, 'Surface');iSurface = getappdata(hdl.hFig_MRI, 'iSurface');
    TessInfo(iSurface).DataThreshold = 0;
    TessInfo(iSurface).DataMinMax = [double(Thres), max(Map)];
    TessInfo(iSurface).DataLimitValue = [double(Thres), max(Map)];
    TessInfo(iSurface).Data = Map;
    TessInfo(iSurface).Data(Map<double(Thres)) = 0;
    TessInfo(iSurface).DataAlpha = hdl.Slider_HFOAlpha.Value;
    setappdata(hdl.hFig_MRI, 'Surface', TessInfo);
    FigureId = getappdata(hdl.hFig_MRI, 'FigureId');
    figure_mri('UpdateSurfaceColor', hdl.hFig_MRI, iSurface);
    % Modify Color bar
    hColorbar = findobj(hdl.hFig_MRI, '-depth', 1, 'Tag', 'Colorbar');
    set(hColorbar,'YTickLabels',cellfun(@(x)num2str(round(x*100)/100),num2cell(double(Thres):(1 - double(Thres)) / (length(hColorbar.YTick) - 1):1),'UniformOutput',false));
    hColorbar.XLabel.String = '';
    % Show SEEG
    if(~isempty(hdl.SeegChannelMat))
        Value = hdl.List_SEEG.Value;channel = sort(cellfun(@(x)[x,' '],{hdl.SeegChannelMat.Channel.Name},'UniformOutput',0));Value = cellfun(@(x)find(ismember({hdl.SeegChannelMat.Channel.Name},x(1:end-1))),channel(Value));
        if(~isempty(Value))
            temp = hdl.SeegChannelMat;temp.IntraElectrodes = temp.IntraElectrodes(cellfun(@(x)any(contains(unique({temp.Channel(Value).Group}),x)),{temp.IntraElectrodes.Name}));temp.Channel = temp.Channel(Value);
            TempDir = ['Channel_',num2str(round(hdl.rand_num*100000)),'.mat']; 
            hdl.Channel_MRIDir = fullfile(hdl.ProtocolPath, 'data', TempDir);
            save(fullfile(hdl.ProtocolPath, 'data', TempDir),'-struct','temp');
            [~,~,iDS] = bst_figures('GetFigure', hdl.hFig_MRI);
            GlobalData.DataSet(iDS).ChannelFile = [];
            figure_mri('LoadElectrodes', hdl.hFig_MRI, hdl.Channel_MRIDir, 'SEEG');
        end
    end
    CallFun_ResultsThreshold();
    % set mri figure on the top
    jFrame = get(hdl.hFig_MRI,'JavaFrame');drawnow;
    jFrame.fHG2Client.getWindow.setAlwaysOnTop(true);
end

function ShowSurface(Thre, Smooth, Alpha, Position)
    global hdl
    % Add surf and Show Results
    % Add private figure_3d.m Path
    Thres = round(peak2peak(hdl.Map) * min(max(Thre, 0.01), 0.99) * 100) / 100;
    hdl.SurfPos = [0.00 0.000 0.75 1.00];
    if(~isfield(hdl, 'Fig') || ~isvalid(hdl.Fig))
        [hdl.Fig, ~, ~] = view_surface(hdl.Surface, 0, [.6,.6,.6], 'NewFigure');set(hdl.Fig,'Name',hdl.FigureName,'NumberTitle', 'off');  
    else
        delete(hdl.axes.Children(contains(hdl.axes.Children.get('Tag'),'VS')));
        hPan = findobj(hdl.Fig, '-depth', 1, 'Tag', 'SurfPan');
        delete(hPan.Children(contains(hPan.Children.get('Tag'),'Colorbar')));
    end 
    if(nargin == 4)
        hdl.Fig.Position(1) = Position(1) - hdl.Fig.Position(3);
%         hdl.Fig.Position(1) = Position(1);
        hdl.Fig.Position(2) = Position(2) + Position(4) - hdl.Fig.Position(4);
    end
    hdl.axes = findobj(hdl.Fig, '-depth', 2, 'tag', 'Axes3D');AxesPos = hdl.axes.Position(3)*hdl.Fig.Position(3);hdl.LabelSize = round(AxesPos/20);
    Loc = hdl.Loc';Map = double(hdl.Map);
    Loc(:,Map < double(Thres)) = [];Map(Map<double(Thres)) = [];
    % Make Sure VS in Whole range of Data in Axes
    if(length(unique(Map)) ~= 1)
        Map = (double(Map) - min(double(Map))) / (max(double(Map)) - min(double(Map))) * hdl.ColorLimit(2) + hdl.ColorLimit(1) + 0.1;  
    end
    hold(hdl.axes,'on');scatter3(hdl.axes,Loc(1,:),Loc(2,:),Loc(3,:),hdl.LabelSize * 100,Map,'.','Tag','VS');hold(hdl.axes,'off');
    % Update current surface (Smooth and Tranp)
    TessInfo = getappdata(hdl.Fig, 'Surface');iSurface = find(cellfun(@(x)strcmp(x, 'Cortex'),  {TessInfo.Name}));
    TessInfo(iSurface).SurfAlpha = Alpha;
    TessInfo(iSurface).SurfSmoothValue=round(Smooth*100)/100;setappdata(hdl.Fig, 'Surface', TessInfo);
    figure_3d('UpdateSurfaceAlpha', hdl.Fig, iSurface);figure_3d('UpdateSurfaceAlpha', hdl.Fig, iSurface);
    % Set Colorbar
    hPan = findobj(hdl.Fig, '-depth', 1, 'Tag', 'SurfPan');
    hdl.hbar = colorbar(hdl.axes, 'Units','normalized','position',[0.89 0.25 0.03 0.5],'Color',[1,1,1],'FontWeight','bold','Parent',hPan);colormap(hdl.axes, hot);
    set(hdl.hbar,'FontSize',hdl.LabelSize / 2,'FontWeight','bold');
    set(hdl.hbar,'Ticks',(min(Map) : peak2peak(Map) / 5 : max(Map)));
    set(hdl.hbar,'TickLabels',cellfun(@(x)num2str(x),num2cell(round(100 * (double(Thres) : (double(max(hdl.Map)) - double(Thres)) / 5 : double(max(hdl.Map)))) / 100),'UniformOutput',false));
    set(hdl.hbar,'Limit',[min(min(Map), 0.99), max(Map)]);
    figure_3d('SmoothSurface', hdl.Fig, iSurface, round(Smooth*100)/100);
    alpha(hdl.axes,1-round(Alpha*100)/100);
    if(isfield(hdl, 'ResultsThreshold')&&isvalid(hdl.ResultsThreshold))
        hdl.ResultsThreshold.String = ['Overlap Thres: ', num2str(Thres)];
    end
end