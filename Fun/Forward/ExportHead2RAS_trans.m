function trans = ExportHead2RAS_trans(sMri, ChannelMat, OutDir)
    % neuromag_head=>scs=>refine_scs=>mri=>surface_ras
    % get head2surface_ras trans
    mri2surface_ras = cat(1, cat(2, eye(3), - (size(sMri.Cube)' / 2 + [0 1 0]') ./ sMri.Voxsize' / 1000), ...
                             [0, 0, 0, 1]);
    scs2mri = eye(4) * inv([sMri.SCS.R, sMri.SCS.T./1000; 0 0 0 1]);
    head2scs = ChannelMat.TransfMeg{find(contains(ChannelMat.TransfMegLabels, 'neuromag_head=>scs'))};
    scs2scs = ChannelMat.TransfMeg{find(contains(ChannelMat.TransfMegLabels, 'refine registration: head points'))};
    head2mri = mri2surface_ras * scs2mri * scs2scs * head2scs;
    % save trans
    trans = ExportTrans(head2mri, OutDir);
end