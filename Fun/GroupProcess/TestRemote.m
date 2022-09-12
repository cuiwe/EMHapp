function error = TestRemote(matlab_path, project_path, usr_name, passwd, ip, sshpass_cmd)
    error = [];
    if(nargin == 6)
        % step1: check ip
        cmd = ['ping -c 1 ', ip];
        [status, ~] = system(cmd);
        if(status ~= 0)
           error = 'ip is unreachable';
           return
        end

        % step2: check usrname and passwd
        cmd = [sshpass_cmd,' -p ', passwd,' ssh ', usr_name, '@', ip, ' ls'];
        [status, ~] = system(cmd);
        if(status ~= 0)
           error = 'usrname or passwd is error';
           return
        end

        % step3: check matlab path
        cmd = [sshpass_cmd,' -p ', passwd,' ssh ', usr_name, '@', ip, ' ls ', matlab_path];
        [status, ~] = system(cmd);
        if(status ~= 0)
           error = 'matlab path is not exist';
           return
        end

        % step4: check project path
        cmd = [sshpass_cmd,' -p ', passwd,' ssh ', usr_name, '@', ip,...
               ' "mkdir -p ', project_path, '; mkdir ', project_path, '/test; ', 'rm -r ', project_path, '/test"'];
        [status, ~] = system(cmd);
        if(status ~= 0)
           error = 'project path is not exist or not writeable';
           return
        end
    else
        error = 'input error';
    end
end

