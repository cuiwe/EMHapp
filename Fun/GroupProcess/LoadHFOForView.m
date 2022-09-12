function HFOView = LoadHFOForView(HFOdetectionResults, VS_ChannelMat, SampelFreq)
    if(isfield(HFOdetectionResults, 'EntHFO')&&~isempty(HFOdetectionResults.EntHFO))
        % Export to global value
        % For Show (BaseLine MEG sample, VS Weight, Channel Name and HFO time in BaseLine)
        HFOView.toShow.ReSample = SampelFreq;
        HFOView.toShow.BsTime = HFOdetectionResults.EntHFO_BsTime;
        HFOView.toShow.ArdBsTime = HFOdetectionResults.ArdEntHFO_BsTime;
        HFOView.toShow.Weight = HFOdetectionResults.EntHFO_Weight;
        HFOView.toShow.ArdWeight = HFOdetectionResults.ArdEntHFO_Weight;
        HFOView.toShow.Weight_S = HFOdetectionResults.EntHFO_Weight_S;
        HFOView.toShow.ArdWeight_S = HFOdetectionResults.ArdEntHFO_Weight_S;
        HFOView.toShow.Channel = cellfun(@(x)reshape(cat(1,{VS_ChannelMat.Channel(x(:,2)).Name},{VS_ChannelMat.Channel(x(:,2)).Group}), [], 1)',HFOdetectionResults.EntHFO,'UniformOutput',false);
        HFOView.toShow.ArdChannel = cellfun(@(x)reshape(cat(1,{VS_ChannelMat.Channel(x(:,2)).Name},{VS_ChannelMat.Channel(x(:,2)).Group}), [], 1)',HFOdetectionResults.ArdEntHFO,'UniformOutput',false);
        HFOView.toShow.EventTime = cellfun(@(x)round([min(x(:,3)),max(x(:,4))]),HFOdetectionResults.EntHFO,'UniformOutput',false);
        % For Feature
        HFOView.toShow.LL_HFO = HFOdetectionResults.Features.LL_HFO;
        HFOView.toShow.HilAmp_HFO = HFOdetectionResults.Features.HilAmp_HFO;
        HFOView.toShow.TFEntropy_HFO = HFOdetectionResults.Features.TFEntropy_HFO;
        HFOView.toShow.LL_Spike = HFOdetectionResults.Features.LL_Spike;
        HFOView.toShow.PeakAmp_Spike = HFOdetectionResults.Features.PeakAmp_Spike;
        HFOView.toShow.TFEntropy_Spike = HFOdetectionResults.Features.TFEntropy_Spike;
        % For Cluster
        HFOView.toShow.Bic = HFOdetectionResults.Bic;
        HFOView.toShow.ClsIdx = HFOdetectionResults.ClsIdx;
        HFOView.toShow.Cls = HFOdetectionResults.Cls;
        HFOView.toShow.Best_Cls = HFOdetectionResults.Best_Cls;
        % For Event
        SampleEventLabel = cellfun(@(x)cat(2,'Event_',num2str(x)),num2cell(1:size(HFOdetectionResults.EntHFO,2)),'UniformOutput',false);
        HFOView.Event.EventName = SampleEventLabel;
        HFOView.Event.EventValue = SampleEventLabel{1};
        HFOView.Event.EventToDisp = true(length(SampleEventLabel),1);
        % For HFO Index
        HFOView.HFOIdx = HFOdetectionResults.EntHFO;
        HFOView.ArdHFOIdx = HFOdetectionResults.ArdEntHFO;
        HFOView.HFOIdxToShow = 1 : size(HFOdetectionResults.EntHFO, 2);
        HFOView.SourceMaps = HFOdetectionResults.SourceMapsHFO;
    else
        HFOView.toShow = [];
        HFOView.Event = [];
        HFOView.Event.EventName={'none'};
        HFOView.HFOIdx = [];
        HFOView.Signal = [];
    end
end
