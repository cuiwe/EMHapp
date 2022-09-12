sshpass_cmd = '/usr/local/bin/sshpass';
emhapp_path = strrep(which('EMHapp.m'), [filesep, 'EMHapp.m'], '');
addpath(genpath(fullfile(emhapp_path, 'Fun')));
addpath(genpath(fullfile(emhapp_path, 'GUI')));
Main(sshpass_cmd);