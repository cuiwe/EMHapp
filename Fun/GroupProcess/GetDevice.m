function Out = GetDevice(IP, User, Passwd, sshpass_cmd)
    Out = [];
    if(nargin==0)
        [status, gpu] = dos('nvidia-smi --list-gpus');
    else
        [status, gpu] = dos([sshpass_cmd,' -p ',Passwd,' ssh ',User,'@',IP,' nvidia-smi --list-gpus']);
    end
    if(status==0)
        Out = struct('DeviceID',-1,'Device',{'CPU'});
        gpu = splitlines(gpu);
        for i=0:length(gpu) - 2
            Out.DeviceID = cat(1, Out.DeviceID, i);
            Out.Device = cat(1, Out.Device, {['GPU', num2str(i)]});
        end
    else
        Out = struct('DeviceID',-1,'Device',{'CPU'});
    end
end