function StartFcn(BrainstormDbDir)
    % bst Init
    brainstorm setpath
    if ~brainstorm('status')
        brainstorm server
    end
    ft_defaults();
    % Set Database folder
    global GlobalData;
    try
        % Select new database folder
        bst_set('BrainstormDbDir', BrainstormDbDir);
        % Get the current protocol: if not empty, there are existing protocols to unload
        iiProtocol = bst_get('iProtocol');
        if ~isempty(iiProtocol) && (iiProtocol >= 1)
            % Reset all the structures
            GlobalData.DataBase.ProtocolInfo(:)     = [];
            GlobalData.DataBase.ProtocolSubjects(:) = [];
            GlobalData.DataBase.ProtocolStudies(:)  = [];
            GlobalData.DataBase.isProtocolLoaded    = [];
            GlobalData.DataBase.isProtocolModified  = [];
            % Select current protocol in combo list
            gui_brainstorm('SetCurrentProtocol', 0);
            % Update interface
            gui_brainstorm('UpdateProtocolsList');
            panel_protocols('UpdateTree');
        end
        % Import new database
        db_import(BrainstormDbDir);
    catch
        bst_set('BrainstormDbDir', BrainstormDbDir);
        % Get the current protocol: if not empty, there are existing protocols to unload
        iiProtocol = bst_get('iProtocol');
        if ~isempty(iiProtocol) && (iiProtocol >= 1)
            % Reset all the structures
            GlobalData.DataBase.ProtocolInfo(:)     = [];
            GlobalData.DataBase.ProtocolSubjects(:) = [];
            GlobalData.DataBase.ProtocolStudies(:)  = [];
            GlobalData.DataBase.isProtocolLoaded    = [];
            GlobalData.DataBase.isProtocolModified  = [];
            % Select current protocol in combo list
            gui_brainstorm('SetCurrentProtocol', 0);
            % Update interface
            gui_brainstorm('UpdateProtocolsList');
            panel_protocols('UpdateTree');
        end
        % Import new database
        db_import(BrainstormDbDir);
    end
    db_save(1);
end

