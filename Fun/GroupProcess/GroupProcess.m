function status = GroupProcess()
    % head_method: 0 -> localsphere; 1 -> 1-layer BEM; 2 -> 2-layer BEM; 3 -> 3-layer BEM
    head_method = 0;
    % Get cfg dir
    cfg_path = fullfile(pwd, 'cfg.mat');
    % Load Data
    if(~exist(cfg_path, 'file'))
        status = 'error';
        return 
    end
    % add path
    addpath(genpath(fileparts(fileparts(which('GroupProcess')))));
    % Load Data
    temp = load(cfg_path);db_process = temp.db_process;
    BrainstormDbDir = fullfile(temp.ProjectDir, temp.BrainstormDbDir);
    % Start bst
    StartFcn(BrainstormDbDir);
    process_info = db_process.process_info;
    % Remove log files of all subjects
    stdout_files = process_info.stdout_dir;
    if(exist(stdout_files, 'file'))
        delete(stdout_files);
    end
    for subj=1:length(process_info.process_data)
        process_data = process_info.process_data(subj);
        log_files = process_data.log_dir;
        if(exist(log_files, 'file'))
            delete(log_files);
        end
    end
    % Process Data for each subject
    for subj=1:length(process_info.process_data)
        % STD_OUT to file
        diary(stdout_files);diary on;
        process_data = process_info.process_data(subj);
        log_files = process_data.log_dir;
        try
            % prepare process paraeters (smri, channel_mat, meg_file, results_file)
            [smri, seg, inner, channel_mat, meg_files, ied_results_path, vs_results_path, hfo_results_path] ...
                  = PrepareProcessParam(process_data);
            % generate source model (source grid)
            grid_size = process_info.virtual_sensor_opt.Volume_Resolutione / 1000;
            bstChannel = MakeSourceModel(smri, grid_size, seg, inner);
            % generate lf matrix
            source_grid = cat(2, bstChannel.Channel.Loc)';
            leadfields = [];
            for fif=1:length(channel_mat)
                head_model_path = fullfile(fileparts(ied_results_path{fif}), 'HeadModel.mat');
                % load previously generated HeadModel
                if(~exist(head_model_path, 'file'))
                    head_model_mat =  MakeLeadField(source_grid, channel_mat(fif), process_data, head_method);
                    save(head_model_path, '-struct', 'head_model_mat');
                else
                    temp = load(head_model_path);head_model_mat = temp;
                end
                temp = cat(3, head_model_mat.Gain(:, 1:3:end), head_model_mat.Gain(:, 2:3:end), head_model_mat.Gain(:, 3:3:end));
                temp = temp(good_channel(channel_mat(fif).Channel, [], 'MEG'), :, :);
                temp = permute(temp, [2, 3, 1]);
                temp = reshape(temp, 1, size(temp, 1), size(temp, 2), size(temp, 3));
                leadfields = cat(1, leadfields, temp);
            end

            % Prepare parameters to pipeline
            ParamPath = fullfile(process_info.project_dir ,process_data.subject_name, 'Parameters.mat');
            Files.RawFile = meg_files;Files.FileIED = [];Files.FileHFO = [];
            Files.LogFile = log_files;
            Files.IEDDetectionDirs = ied_results_path;
            Files.VirtualSensorDirs = vs_results_path;
            Files.HFODetectionDirs = hfo_results_path;
            PipelineParam.Files = Files;
            PipelineParam.PreprocessOpt = process_info.pre_process_opt;
            PipelineParam.IEDdetectionOpt = process_info.ied_detection_opt;
            PipelineParam.VirtualSensorOpt = process_info.virtual_sensor_opt;
            PipelineParam.HFOdetectionOpt = process_info.hfo_detection_opt;
            PipelineParam.bstChannel = bstChannel;
            PipelineParam.leadfields=leadfields;
            PipelineParam.cuda_device=process_info.device;
            save(ParamPath,'-struct','PipelineParam','-v6');

            % Call Python and Run
            tempDir = fullfile(process_info.project_dir,  'RunningData', 'Fun', 'EMHapp');
            cmd = ['source ~/.bashrc;source activate;conda activate EMHapp;python ', fullfile(tempDir, 'emhapp_run.py'), ' --mat ', ParamPath];
            status = system(cmd);
            
            % get log
            log = importdata(log_files);
            is_error = any(cellfun(@(x)contains(x, '[ERROR]'), log));
            
            % generate mat file for GUI
            if(status == 0 && is_error == 0)
                % write IEDView mat Files
                for i=1:size(process_data.result_path, 2)
                    result_path = fullfile(process_data.result_path{i}, 'SaveIEDDetectionResults.mat');
                    if(exist(result_path, 'file'))
                        % Generate spike view mat file
                        spike_detection_results = load(result_path);
                        spike_detection_view = LoadSpikeForView(spike_detection_results, process_data.meg_channel_mat(i));
                        % save mat file 
                        save(fullfile(process_data.result_path{i},'SaveSpikeViewResults.mat'),'-struct', 'spike_detection_view', '-v6');
                    end
                end
                % write HFOView mat Files
                for i=1:size(process_data.result_path, 2)
                    ied_result_path = fullfile(process_data.result_path{i}, 'SaveIEDDetectionResults.mat');
                    hfo_result_path = fullfile(process_data.result_path{i}, 'SaveHFODetectionResults.mat');
                    vs_result_path = fullfile(process_data.result_path{i}, 'SaveVirtualSensorResults.mat');
                    if(exist(ied_result_path, 'file') && exist(hfo_result_path, 'file') && exist(vs_result_path, 'file'))
                        vs_channel_mat = load(vs_result_path, 'bstChannel');vs_channel_mat = vs_channel_mat.bstChannel;
                        sfreq = load(ied_result_path, 'SampleFreq');sfreq = sfreq.SampleFreq;
                        hfo_detection_results = load(hfo_result_path);
                        % Generate hfo view mat file
                        hfo_detection_view = LoadHFOForView(hfo_detection_results, vs_channel_mat, sfreq);
                        % save mat file 
                        save(fullfile(process_data.result_path{i},'SaveHFOViewResults.mat'),'-struct', 'hfo_detection_view', '-v6');
                    end
                end
                % update process_status in cfg.mat (finished)
                temp = load(cfg_path);
                temp.db_process.process_info.process_data(subj).process_status = 1;
                save(cfg_path, '-struct', 'temp', '-v6');
                status = 'ok';
            else
                % update process_status in cfg.mat (error)
                temp = load(cfg_path);
                temp.db_process.process_info.process_data(subj).process_status = 3;
                save(cfg_path, '-struct', 'temp', '-v6');
                status = 'error';
            end
            % save STD_OUT to file
            diary off;
        catch exception
            % save error log
            message = ['[ERROR][',exception.identifier,'][', datestr(clock), '][', process_data.subject_name,  ']: ', exception.message];
            fid = fopen(log_files, 'w');
            fprintf(fid, '%s\n', message);
            fclose(fid);
            % update process_status in cfg.mat (error)
            temp = load(cfg_path);
            temp.db_process.process_info.process_data(subj).process_status = 3;
            save(cfg_path, '-struct', 'temp', '-v6');
            % save STD_OUT to file
            diary off;
            status = 'error';
        end
    end
end