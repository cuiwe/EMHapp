function [smri, seg, inner, channel_mat, meg_files, ied_results_path, vs_results_path, hfo_results_path] = PrepareProcessParam(process_data)
    % set current protocol
    gui_brainstorm('SetCurrentProtocol', bst_get('Protocol', process_data.subject_name));
    % get smri
    temp = bst_get('ProtocolSubjects');temp=temp.Subject;iSubject = find(contains({temp.Name}, process_data.subject_name));
    sSubject = bst_get('Subject', iSubject);
    smri = bst_memory('LoadMri', sSubject.Anatomy(sSubject.iAnatomy).FileName);
    % get segment
    seg_index = contains({sSubject.Anatomy.Comment}, 'tissues_md');
    if(isempty(seg_index))
        seg = [];
    else
        seg = bst_memory('LoadMri', sSubject.Anatomy(seg_index).FileName);
    end
    % get segment
    inner_index = sSubject.iInnerSkull;
    if(isempty(inner_index))
        inner = [];
    else
        inner = bst_memory('LoadSurface', sSubject.Surface(inner_index).FileName);
    end
    % get channel mat 
    channel_mat = process_data.meg_channel_mat;
    % get meg file
    meg_files = process_data.meg_path;
    % get result path
    ied_results_path = cellfun(@(x)fullfile(x, 'SaveIEDDetectionResults.mat'), process_data.result_path, 'UniformOutput', false)';
    vs_results_path = cellfun(@(x)fullfile(x, 'SaveVirtualSensorResults.mat'), process_data.result_path, 'UniformOutput', false)';
    hfo_results_path = cellfun(@(x)fullfile(x, 'SaveHFODetectionResults.mat'), process_data.result_path, 'UniformOutput', false)';
end