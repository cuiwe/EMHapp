function SpikeView = LoadSpikeForView(SpikedetectionResults, ChannelMat)
    global GlobalData
    if(~isempty(SpikedetectionResults.BsTime))
        % Add Channel Group Name
        MontagesIdx = (46:53);
        RegionIdx = cell(size(MontagesIdx));
        for i=1:length(MontagesIdx)
           ChannelName = {ChannelMat.Channel.Name};
           MontagesChannelName = GlobalData.ChannelMontages.Montages(MontagesIdx(i)).ChanNames;
           MontagesChannelName = cellfun(@(x)strrep(x,' ',''),MontagesChannelName,'UniformOutput',false);
           Idx = find(cellfun(@(x)any(contains(MontagesChannelName,x)), ChannelName));
           RegionIdx{i} = Idx;
           for j=1:length(Idx)
               ChannelMat.Channel(Idx(j)).Group = GlobalData.ChannelMontages.Montages(MontagesIdx(i)).Name;
           end
        end
        % Get the Channel Index of each brain regions
        Channel = ChannelMat.Channel;
        ChannelTmep = find(contains({ChannelMat.Channel.Type},'MEG'));
        RegionIdx = cellfun(@(x)[x(ismember(x,find(contains({Channel.Type},'MEG GRAD')))),...
                                 x(ismember(x,find(contains({Channel.Type},'MEG MAG'))))],RegionIdx,'UniformOutput',false);
        RegionName = cellfun(@(x){Channel(x).Name;Channel(x).Group},RegionIdx,'UniformOutput',false);
        RegionIdx = cellfun(@(x)x-min(ChannelTmep)+1,RegionIdx,'UniformOutput',false);
        % Get Slope, Sharp, Amplitude, Peak Index and Spike Time 
        SampleFreq = double(SpikedetectionResults.SampleFreq);
        SpikeTime = SpikedetectionResults.BsTime;
        EventRegionChannel = cell(size(SpikeTime, 1),1);
        EventRegionChannelName = cell(size(SpikeTime, 1),1);
        EventChannel = cell(size(SpikeTime, 1),1);
        EventPeaks = cell(size(SpikeTime, 1),1);
        EventPeakAmp = cell(size(SpikeTime, 1),1);
        EventPeakSlope = cell(size(SpikeTime, 1),1);
        EventPeakSharp = cell(size(SpikeTime, 1),1);
        EventPeakCorr = cell(size(SpikeTime, 1),1);
        EventPeakDist = cell(size(SpikeTime, 1),1);
        for i=1:size(SpikeTime, 1)
            % Get Channel Index in each brain Region (Sorted by Spike Channel number)
            EventRegionChannel{i} = RegionIdx(SpikedetectionResults.SpikeRegion(i,:));
            % Get Channel Name in each brain Region (Sorted by Spike Channel number)
            EventRegionChannelName{i} = RegionName(SpikedetectionResults.SpikeRegion(i,:));
            % Get Spike Channel Index in each brain Region (Sorted by Spike Channel number)
            EventChannel{i} = cellfun(@(x)find(ismember(x,SpikedetectionResults.SpikeChan{i})),EventRegionChannel{i},'UniformOutput',false);
            EventChannel{i}(cellfun(@(x)isempty(x),EventChannel{i})) = [];
            % Get Amp
            EventPeakAmp{i} = zeros(length(ChannelTmep), 1);EventPeakAmp{i}(SpikedetectionResults.SpikeChan{i}) = SpikedetectionResults.Amp{i};
            EventPeakAmp{i} = cellfun(@(x)EventPeakAmp{i}(x), EventRegionChannel{i}, 'UniformOutput', false);
            % Get Slope
            EventPeakSlope{i} = zeros(length(ChannelTmep), 1);EventPeakSlope{i}(SpikedetectionResults.SpikeChan{i}) = SpikedetectionResults.Slope{i};
            EventPeakSlope{i} = cellfun(@(x)EventPeakSlope{i}(x), EventRegionChannel{i}, 'UniformOutput', false);
            % Get Sharp
            EventPeakSharp{i} = zeros(length(ChannelTmep), 1);EventPeakSharp{i}(SpikedetectionResults.SpikeChan{i}) = SpikedetectionResults.Sharp{i};
            EventPeakSharp{i} = cellfun(@(x)EventPeakSharp{i}(x), EventRegionChannel{i}, 'UniformOutput', false);
            % Get Corr
            EventPeakCorr{i} = zeros(length(ChannelTmep), length(SpikedetectionResults.Template));EventPeakCorr{i}(SpikedetectionResults.SpikeChan{i}, :) = SpikedetectionResults.EventCorr{i};
            EventPeakCorr{i} = cellfun(@(x)EventPeakCorr{i}(x, :), EventRegionChannel{i}, 'UniformOutput', false);
            % Get Dist
            EventPeakDist{i} = 100*ones(length(ChannelTmep), length(SpikedetectionResults.Template));EventPeakDist{i}(SpikedetectionResults.SpikeChan{i}, :) = SpikedetectionResults.EventDist{i};
            EventPeakDist{i} = cellfun(@(x)EventPeakDist{i}(x, :), EventRegionChannel{i}, 'UniformOutput', false);
        end
        % Export to global value
        SampleEventLabel = cellfun(@(x)cat(2,'Event_',num2str(x)),num2cell(1:size(SpikeTime, 1)),'UniformOutput',false); 
        % For Show
        SpikeView.toShow.ReSample = SampleFreq;
        SpikeView.toShow.BaseLine = mat2cell(double(SpikedetectionResults.BsTime), ones(size(SpikedetectionResults.BsTime, 1),1),size(SpikedetectionResults.BsTime,2));
        SpikeView.toShow.SpikeChannel = EventChannel;
        SpikeView.toShow.RegionChannelName = EventRegionChannelName;
        SpikeView.toShow.RegionChannel = EventRegionChannel;
        SpikeView.toShow.EventTime = double(SpikedetectionResults.EventSpikeTime);
        SpikeView.toShow.EventTimeOffset = zeros(size(SpikeTime, 1),1);
        % For Similarity
        SpikeView.toShow.Template = SpikedetectionResults.Template;
        SpikeView.toShow.Dist = EventPeakDist;
        SpikeView.toShow.Corr = EventPeakCorr;
        % For Features
        SpikeView.toShow.EventSharp = EventPeakSharp;
        SpikeView.toShow.EventSlope = EventPeakSlope;
        SpikeView.toShow.EventAmp = EventPeakAmp;
        % For Event
        SpikeView.Event.EventName = SampleEventLabel;
        SpikeView.Event.EventValue = SampleEventLabel{1};
        SpikeView.Event.EventToDisp = true(length(SampleEventLabel),1);
        SpikeView.SpikeIdxToShow = cellfun(@(x)cellfun(@(y)y>0,x,'UniformOutput', false),SpikeView.toShow.EventAmp, 'UniformOutput', false);
    else
        SpikeView = [];
    end
end