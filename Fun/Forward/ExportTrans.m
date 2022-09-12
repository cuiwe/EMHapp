function trans = ExportTrans(trans, out_dir)
    [fid_raw,~, dir] = fiff_open(strrep(which('ExportTrans.m'), 'ExportTrans.m', 'template-trans.fif'));
    FIFF = fiff_define_constants;
    temp = find([dir.kind] == FIFF.FIFF_COORD_TRANS);
    pos = dir(temp(1)).pos;

    % copy smoething
    fid_targe = fopen(out_dir,'w','ieee-be');
    fseek(fid_targe,0,'bof');fseek(fid_raw,0,'bof');
    for i=1:pos+4*6
        temp = fread(fid_raw,1);
        fwrite(fid_targe, temp);
    end
    
    % write trans
    data = [single(reshape(trans(1:3, 1:3)', 9, 1)'), single(trans(1:3, end))'];
    count = fwrite(fid_targe, single(data), 'single',0);
    fread(fid_raw,9,'single=>double');
    fread(fid_raw,3,'single=>double');
    
    % copy smoething
    while ~feof(fid_raw)
        temp = fread(fid_raw,1);
        fwrite(fid_targe, temp);
    end
       
    fclose(fid_raw);
    fclose(fid_targe);
    trans = fiff_read_coord_trans(out_dir);
end