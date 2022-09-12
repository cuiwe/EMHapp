function segment_modified = ModifiedSegment(segment_spm, segment_ft, headmask)
    %% step0: get segment data
    segment_spm.Cube(segment_spm.Cube==5)=0;segment_spm.Cube(1,1,1)=5;
    segment_ft.Cube(segment_ft.Cube==5)=0;segment_ft.Cube(1,1,1)=5;
    sphere = strel('sphere', 10);
    square = strel('square', 10);

    %% step1: get brain mask (CSF/WM/GM)
    brain_mask = segment_spm.Cube;
    brain_mask(brain_mask ~= 1 & brain_mask ~= 2 & brain_mask ~= 3) = 0;
    brain_mask(brain_mask == 1 | brain_mask == 2 | brain_mask == 3) = 1;
    brain_mask = imclose(brain_mask, sphere);

    %% step2: modified skull mask
    % exclude brain region from skull mask
    skull_mask = segment_spm.Cube;
    skull_mask(skull_mask ~= 4) = 0;
    skull_mask(skull_mask == 4) = 1;
    skull_mask(brain_mask == 1) = 0;
    % remove regions below brain using ft segment
    skull_mask_ft = segment_ft.Cube;
    skull_mask_ft(skull_mask_ft ~= 4) = 0;
    skull_mask_ft(skull_mask_ft == 4) = 1;
    % dilate ft_skull about 10mm.
    skull_mask_ft = imdilate(skull_mask_ft, strel('sphere', 10));
    skull_mask(skull_mask_ft == 0) = 0;
    % get connected region
    skull_label = bwlabeln(skull_mask);
    label_n = cellfun(@(x)length(find(skull_label==x)), num2cell(1:max(unique(skull_label))));
    % keep max region
    [~, label_max] = max(label_n);
    skull_mask(skull_label ~= label_max) = 0;
    % close image
    skull_mask = imclose(skull_mask, sphere);

    %% step3: "modified" CSF mask
    % close CSF and skull mask (fill region between CSF and skull)
    csf_skull_mask = double(brain_mask | bwperim(skull_mask));
    csf_skull_mask = imclose(csf_skull_mask, strel('square', 15));
    csf_skull_mask(skull_mask == 1) = 0;csf_skull_mask = imopen(csf_skull_mask, sphere);
    csf_skull_mask(segment_spm.Cube == 1 | segment_spm.Cube == 2) = 0;
    % add edge of brain_mask
    brain_edge = bwperim(brain_mask);
    csf_skull_mask(brain_edge == 1) = 1;csf_mask = csf_skull_mask;
    
    %% step4: modified head mask
    head_mask = headmask;
    head_mask = imclose(headmask, sphere);
    head_mask(csf_mask == 1 | skull_mask == 1 | segment_spm.Cube == 1 | segment_spm.Cube == 2) = 0;

    %% step5: export
    segment_modified = segment_spm;segment_modified_cube = segment_modified.Cube;
    segment_modified_cube(segment_modified_cube==3) = 0;segment_modified_cube(csf_mask==1) = 3;
    segment_modified_cube(segment_modified_cube==4) = 0;segment_modified_cube(segment_modified_cube==0 & skull_mask==1) = 4;
    segment_modified_cube(segment_modified_cube==5) = 0;segment_modified_cube(segment_modified_cube==0 & head_mask==1) = 5;
    segment_modified.Cube = segment_modified_cube;
    segment_modified.Comment = 'tissues_md';
end