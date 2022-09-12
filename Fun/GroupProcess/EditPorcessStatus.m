function EditPorcessStatus(EditIndex)
    % Edit process status to "downloaded"
    % Get cfg dir
    cfg_path = fullfile(pwd, 'cfg.mat');
    % Load Data
    if(~exist(cfg_path, 'file'))
        status = 'error';
        return 
    end
    % edit process_status in cfg.mat
    temp = load(cfg_path);
    for i=1:length(temp.db_process.process_info.process_data)
        if(EditIndex(i) == 1)
            temp.db_process.process_info.process_data(i).process_status = 2;
        end
    end
    save(cfg_path, '-struct', 'temp', '-v6');
end