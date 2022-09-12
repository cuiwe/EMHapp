# -*- coding:utf-8 -*-
# @Time    : 2022/2/4
# @Author  : cuiwei
# @File    : emhapp_run.py
# @Software: PyCharm
# @Script to:
#   - 运行HFO/IED detection pipeline
import os
import mne
import torch
import scipy.io
import argparse
import numpy as np
from mne.utils import logger
from logs import logs
from meg_preprocessing.meg_preprocessing import preprocess
from ied_detection.ied_detection_multi_files import ied_detection_threshold_multi
from ied_detection.ied_temporal_clustering import temporal_cluster
from virtual_sensor_recon.vs_reconstruction import vs_recon
from hfo_detection.hfo_clustering import hfo_cluster


def set_log_file(output_format, fname, overwrite=False):
    logs.set_log_file(fname=fname, output_format=output_format, overwrite=overwrite)
    mne.set_log_file(fname=fname, output_format=output_format, overwrite=overwrite)


def load_parameters(parameter_dir):
    """
    Description:
        读取从EMHapp输出的处理参数

    Input:
        :param parameter_dir: path
            从EMHapp输出的处理参数(.mat文件)
    Return:
        :return bst_channel: dict
            从mat文件读取的虚拟电极的bst_channel
        :return leadfields: list, [ndarray, double, shape(n_dipoles, 3, n_channels)], shape(n_files)
            leadfield矩阵
        :return file_dirs: dict
            MEG文件，以及输出mat文件路径
        :return file_dirs: dict
            MEG文件，以及输出mat文件路径
        :return preprocess_parameters: dict
            预处理参数
        :return ied_detection_parameters: dict
            ied检测参数
        :return vs_reconstruction_parameters: dict
            vs重建参数
        :return hfo_detection_parameters: dict
            hfo检测参数
    """
    # 读取mat文件
    parameters = scipy.io.loadmat(parameter_dir)

    # 读取cuda device
    cuda_device = parameters['cuda_device'][0][0]
    # 读取leadfield
    leadfields = [x for x in parameters['leadfields']]
    # 读取虚拟电极的bst_channel
    bst_channel = {'bstChannel': parameters['bstChannel']}

    file_dirs = \
        {  # 读取MEG数据路径
            'fif_files_raw': [x[0][0] for x in parameters['Files']['RawFile'][0][0]],
            'fif_files_ied': [x[0][0] for x in parameters['Files']['FileIED'][0][0]],
            'fif_files_hfo': [x[0][0] for x in parameters['Files']['FileHFO'][0][0]],
            # 读取EMHapp mat文件的输出路径
            'mat_files_hfo': [x[0][0] for x in parameters['Files']['HFODetectionDirs'][0][0]],
            'mat_files_vs_recon': [x[0][0] for x in parameters['Files']['VirtualSensorDirs'][0][0]],
            'mat_files_ied': [x[0][0] for x in parameters['Files']['IEDDetectionDirs'][0][0]],
            # 读取EMHapp log文件
            'log_file': parameters['Files']['LogFile'][0][0][0]}

    # 读取预处理的参数
    preprocess_parameters = \
        {  # TSSS
            'do_tsss': parameters['PreprocessOpt']['isTSSSandICA'][0][0][0][0] == 1,
            'tsss_auto_bad_lowpass': parameters['PreprocessOpt']['TSSSOpt'][0][0]['LowerTSSS'][0][0][0][0],
            'tsss_auto_bad_thre': parameters['PreprocessOpt']['TSSSOpt'][0][0]['ThreTSSS'][0][0][0][0],
            'fine_cal_file': parameters['PreprocessOpt']['TSSSOpt'][0][0]['FineCal'][0][0][0],
            'crosstalk_file': parameters['PreprocessOpt']['TSSSOpt'][0][0]['Crosstalk'][0][0][0],
            # ICA
            'do_ica': parameters['PreprocessOpt']['isTSSSandICA'][0][0][0][0] == 1,
            # 坏段检测
            'do_jump_detection': parameters['PreprocessOpt']['isBad'][0][0][0][0] == 1,
            'jump_threshold': parameters['PreprocessOpt']['BadEventOpt'][0][0]['JumpAmplitudeThreshold'][0][0][0][0],
            'do_noisy_detection': parameters['PreprocessOpt']['isBad'][0][0][0][0] == 1,
            'noisy_freq_range': [parameters['PreprocessOpt']['BadEventOpt'][0][0]['NoisyHighPass'][0][0][0][0],
                                 parameters['PreprocessOpt']['BadEventOpt'][0][0]['NoisyLowPass'][0][0][0][0]],
            'noisy_threshold': parameters['PreprocessOpt']['BadEventOpt'][0][0]['NoisyAmplitudeThreshold'][0][0][0][0],
            # 滤波
            'hfo_frequency_range': [parameters['PreprocessOpt']['Filter'][0][0]['HighPassHFO'][0][0][0][0],
                                    parameters['PreprocessOpt']['Filter'][0][0]['LowPassHFO'][0][0][0][0]],
            'ied_frequency_range': [parameters['PreprocessOpt']['Filter'][0][0]['HighPassIED'][0][0][0][0],
                                    parameters['PreprocessOpt']['Filter'][0][0]['LowPassIED'][0][0][0][0]]}

    # 读取IED检测的参数
    ied_detection_parameters = \
        {  # IED检测: 阈值法
            'z_win': parameters['IEDdetectionOpt']['ZscoreWindows'][0][0][0][0],
            'peak_amplitude_threshold': [parameters['IEDdetectionOpt']['AmplitudeThreshold'][0][0][0][0], None],
            'half_peak_slope_threshold': [parameters['IEDdetectionOpt']['SlopeThreshold'][0][0][0][0], None],
            # 'peak_sharpness_threshold': [parameters['IEDdetectionOpt']['SharpnessThreshold'][0][0][0][0], None],
            'peak_sharpness_threshold': [None, None],
            'chan_threshold': [parameters['IEDdetectionOpt']['ChannelThreshold'][0][0][0][0], 150],
            'exclude_duration': parameters['IEDdetectionOpt']['DurationThreshold'][0][0][0][0],
            # IED检测: 模版匹配法
            'data_segment_window': parameters['IEDdetectionOpt']['SegmentWindows'][0][0][0][0],
            'large_peak_amplitude_threshold': parameters['IEDdetectionOpt']['HighAmplitude'][0][0][0][0],
            'large_peak_half_slope_threshold': parameters['IEDdetectionOpt']['HighSlope'][0][0][0][0],
            'large_peak_sharpness_threshold': parameters['IEDdetectionOpt']['HighSharp'][0][0][0][0],
            'large_peak_duration_threshold': parameters['IEDdetectionOpt']['HighDuration'][0][0][0][0],
            'sim_template': parameters['IEDdetectionOpt']['TemplateSimilarityThreshold'][0][0][0][0],
            'dist_template': parameters['IEDdetectionOpt']['TemplateSimilarityThreshold'][0][0][0][0],
            'corr_template': parameters['IEDdetectionOpt']['TemplateSimilarityThreshold'][0][0][0][0],
            'sim_threshold': parameters['IEDdetectionOpt']['SimilarityThreshold'][0][0][0][0],
            'dist_threshold': parameters['IEDdetectionOpt']['SimilarityThreshold'][0][0][0][0],
            'corr_threshold': parameters['IEDdetectionOpt']['SimilarityThreshold'][0][0][0][0]}

    # 读取虚拟电机重建的参数
    vs_reconstruction_parameters = \
        {  # 虚拟电机重建
            'ied_segment_time': parameters['VirtualSensorOpt']['Signal_Win'][0][0][0][0],
            'ied_segment': parameters['VirtualSensorOpt']['IED_Win'][0][0][0][0],
            'ied_peak2peak_windows': parameters['VirtualSensorOpt']['IEDPeakWin'][0][0][0][0],
            'hfo_window': parameters['VirtualSensorOpt']['HFO_Win'][0][0][0][0],
            'hfo_segment': parameters['VirtualSensorOpt']['HFOSegmentWin'][0][0][0][0],
            'mean_power_windows': parameters['VirtualSensorOpt']['HFOPowerWin'][0][0][0][0]}

    # 读取HFO检测的参数
    hfo_detection_parameters = \
        {  # HFO检测: 阈值法
            'hfo_amplitude_threshold': parameters['HFOdetectionOpt']['HFOAmplitudeThreshold'][0][0][0][0],
            'hfo_duration_threshold': parameters['HFOdetectionOpt']['HFODurationThreshold'][0][0][0][0],
            'hfo_oscillation_threshold': parameters['HFOdetectionOpt']['HFOOscillationThreshold'][0][0][0][0],
            'hfo_entropy_threshold': parameters['HFOdetectionOpt']['HFOEntropyThreshold'][0][0][0][0],
            'hfo_power_ratio_threshold': parameters['HFOdetectionOpt']['HFOPowerRatioThreshold'][0][0][0][0],
            # HFO检测: GMM聚类
            'ied_window': parameters['VirtualSensorOpt']['IED_Win'][0][0][0][0],
            'tf_hfo_window': parameters['HFOdetectionOpt']['TFWindows'][0][0][0][0],
            'n_segments_per_event': parameters['HFOdetectionOpt']['SegmentsEvents'][0][0][0][0],
            'n_component': parameters['HFOdetectionOpt']['GMMComponent'][0][0][0][0],
            'is_whitening': parameters['HFOdetectionOpt']['GMMWhitening'][0][0][0][0] == 1,
            'gmm_n_init': parameters['HFOdetectionOpt']['GMMNinit'][0][0][0][0],
            'gmm_covariance_type': parameters['HFOdetectionOpt']['GMMCovariance'][0][0][0]}

    return cuda_device, leadfields, bst_channel, file_dirs, preprocess_parameters, ied_detection_parameters, \
           vs_reconstruction_parameters, hfo_detection_parameters


def del_var(device, *arg):
    """
    Description:
        清除变量，释放缓存

    Input:
        :param device: torch.device
            torch的device
        :param arg: list, string
            需要清除的变量名称
    """
    if arg is not None:
        for key in list(globals().keys()):
            if key in arg:
                globals()[key] = []
    if device == torch.device("cuda"):
        torch.cuda.empty_cache()


def cal_filter(raw, freq_range=(40, 240), notch=None):
    """
    Description:
        对数据进行滤波

    Input:
        :param raw: mne.io.read_raw_fif返回值
            mne.io.read_raw_fif返回值
        :param freq_range: list/tuple, double, shape(2)
            滤波的频率范围
        :param notch: number, double
            陷波大小
    Return:
        :return raw: mne.io.read_raw_fif返回值
            滤波后数据
    """
    raw_filter = raw.copy()
    raw_filter.filter(l_freq=freq_range[0], h_freq=freq_range[1], fir_design='firwin', pad="reflect_limited",
                      verbose='error', n_jobs=10)
    if (notch is not None) and (freq_range[1] is not None) and (freq_range[1] >= notch):
        raw_filter.notch_filter(np.arange(notch, freq_range[1], notch), verbose='error', n_jobs=10)

    return raw_filter


def read_meg_data(fif_files=None, fif_files_ied=None, fif_files_hfo=None, ictal_event_dirs=None):
    """
    Description:
        读取MEG数据

    Input:
        :param fif_files: list, [path], shape(n_files)
            原始meg路径
        :param fif_files_ied: list, [path], shape(n_files)
            ied频段滤波meg路径
        :param fif_files_hfo: list, [path], shape(n_files)
            hfo频段滤波meg路径
        :param ictal_event_dirs: list, [path], shape(n_sz_files)
            发作事件的路径，需要和fif文件名称相同，并以_sz_events.eve结尾，加上fif的offset时间
    Return:
        :return raw_data_s: list, [ndarray, double, shape(channel, samples)], shape(n_files)
            MEG未滤波数据
        :return raw_data_s_ied: list, [ndarray, double, shape(channel, samples)], shape(n_files)
            MEG滤波后数据(IED)
        :return raw_data_s_hfo: list, [ndarray, double, shape(channel, samples)], shape(n_files)
            MEG滤波后数据(HFO)
        :return ictal_events: list, [ndarray, double, shape(1, sz_samples)], shape(n_sz_files)
            发作时间点
        :return ictal_fif_index: list, [double], shape(n_sz_files)
            发作时间点
        :return data_info_s: dict
            MEG数据的信息, MNE读取的raw.info
    """
    raw_data_s, raw_data_s_ied, raw_data_s_hfo, data_info_s, ictal_events, ictal_fif_index = [], [], [], [], [], []
    #  如果fif_files_ied，fif_files_hfo为空
    if len(fif_files_ied) != len(fif_files):
        fif_files_ied = [[] for _ in range(len(fif_files))]
    if len(fif_files_hfo) != len(fif_files):
        fif_files_hfo = [[] for _ in range(len(fif_files))]
    # 读取数据
    for file, file_ied, file_hfo, i in zip(fif_files, fif_files_ied, fif_files_hfo, range(len(fif_files))):
        subj_name = mne.io.read_info(file)['subject_info']
        subj_name = subj_name['last_name'] + subj_name['first_name']
        print(subj_name + ': Load_Data', str(i + 1) + '/' + str(len(fif_files)))
        # 读取原始数据
        raw = mne.io.read_raw_fif(file, verbose='error', preload=True)
        raw.info['bads'] = []
        raw_data_s.append(raw.get_data(picks=mne.pick_types(raw.info, meg=True)))
        info = mne.pick_info(raw.info, sel=mne.pick_types(raw.info, meg=True))
        data_info_s.append(info)
        # 读取IED频段滤波数据
        if len(file_ied) == 0:
            raw_data_s_ied.append(cal_filter(raw, freq_range=(3., 80.), notch=50.).get_data(
                picks=mne.pick_types(raw.info, meg=True)))
        else:
            raw_ied = mne.io.read_raw_fif(file_ied, verbose='error', preload=True)
            raw_data_s_ied.append(raw_ied.get_data(picks=mne.pick_types(raw.info, meg=True)))
        # 读取hfo频段滤波数据
        if len(file_ied) == 0:
            raw_data_s_hfo.append(cal_filter(raw, freq_range=(80., 200.), notch=None).get_data(
                picks=mne.pick_types(raw.info, meg=True)))
        else:
            raw_hfo = mne.io.read_raw_fif(file_hfo, verbose='error', preload=True)
            raw_data_s_hfo.append(raw_hfo.get_data(picks=mne.pick_types(raw.info, meg=True)))
        # 读取发作事件
        if ictal_event_dirs is not None:
            ictal_event_file = os.path.join(ictal_event_dirs, file.split('/')[-2],
                                            file.split('/')[-1].replace('_tsss.fif', '_sz_events.eve'))
            if os.path.isfile(ictal_event_file):
                temp_events = (mne.read_events(ictal_event_file)[:, 0] - raw.first_samp).reshape(-1, 2)
                ictal_events.append([np.arange(x[0], x[1]) for x in temp_events])
                ictal_fif_index.append(i)

    return raw_data_s, raw_data_s_ied, raw_data_s_hfo, data_info_s, ictal_events, ictal_fif_index


def cal_preprocess(fif_files, out_dir, do_tsss=False, tsss_auto_bad_lowpass=7, tsss_auto_bad_thre=200,
                   fine_cal_file=None, crosstalk_file=None, do_ica=False, cuda_device=-1):
    """
    Description:
        对原始数据进行除处理：TSSS、ICA、降采样等。

    Input:
        :param fif_files: list, [path], shape(n_files)
            原始meg路径
        :param out_dir: path
            预处理后fif文件的存储路径
        :param do_tsss: bool
            是否做TSSS
        :param tsss_auto_bad_lowpass: number, double
            自动检测bad channel时的数据低通滤波
        :param tsss_auto_bad_thre: number, double
            自动检测bad channel时的阈值
        :param fine_cal_file: str
            计算maxwell_filter时用到的fine_cal_file
        :param crosstalk_file: str
            计算maxwell_filter时用到的crosstalk_file
        :param do_ica: bool
            是否做ICA
        :param cuda_device:  number, int
            device<0 使用CPU, device>=0 使用对应GPU

    Return:
        :return fif_files_out: list, [path], shape(n_files)
            预处理后meg路径
    """

    # 处理数据
    subj_name = mne.io.read_info(fif_files[0])['subject_info']
    subj_name = subj_name['last_name'] + subj_name['first_name']
    fif_files_out = []
    for i, file in enumerate(fif_files):
        print(subj_name + ': Preprocess', str(i + 1) + '/' + str(len(fif_files)))
        meg = preprocess(fif_files=[file], out_dir=out_dir, is_log=False, device=cuda_device, n_jobs=10)
        # 确定预处理程序
        if '_tsss' not in os.path.split(file)[-1] or (do_tsss and '_tsss' not in os.path.split(file)[-1]):
            do_tsss = True
        else:
            do_tsss = False
        # 进行TSSS处理
        meg.run_multi_fifs(do_tsss=do_tsss,
                           tsss_auto_bad_lowpass=tsss_auto_bad_lowpass, tsss_auto_bad_thre=tsss_auto_bad_thre,
                           fine_cal_file=fine_cal_file, crosstalk_file=crosstalk_file,
                           do_ica=do_ica, do_resample=True, do_jump_detection=False, do_noisy_detection=False)
        del_var(meg.device)
        fif_files_out.append(os.path.join(out_dir,
                                          os.path.split(file)[-1].replace('_tsss', '').split('.')[-2] + '_tsss.fif'))

    return fif_files_out


def cal_bad_segments(fif_files, out_dir, jump_threshold=10, noisy_threshold=5, noisy_freq_range=(55, 240),
                     cuda_device=-1):
    """
    Description:
        使用阈值方法检测MEG数据中的bad segment

    Input:
        :param fif_files: list, [path], shape(n_files)
            原始meg路径
        :param out_dir: path
            bad segment的存储路径
        :param jump_threshold: number, double
            jump坏片段的阈值
        :param noisy_freq_range: list/tuple, double, shape(2)
            noisy的频率范围：肌电40-240Hz
        :param noisy_threshold: number, double
            坏片段的阈值
        :param cuda_device:  number, int
            device<0 使用CPU, device>=0 使用对应GPU

    Return:
        :return bad_segments: list, [ndarray, bool, shape(1, n_samples)], shape(n_files)
            每个MEG文件的bad segment
    """

    # 读取bad segment
    subj_name = mne.io.read_info(fif_files[0])['subject_info']
    subj_name = subj_name['last_name'] + subj_name['first_name']
    out_dir_segment = os.path.join(out_dir, 'MEG_BAD')
    bad_segments = []
    for i, file in enumerate(fif_files):
        print(subj_name + ': BadSegment_Detection', str(i + 1) + '/' + str(len(fif_files)))
        # 获取jump和noisy的文件路径
        fif_name = os.path.split(file)[-1]
        jump_file = os.path.join(out_dir_segment, fif_name[:-9] + '_jump_threshold_' + str(jump_threshold) + '.npy')
        noisy_file = os.path.join(out_dir_segment, fif_name[:-9] + '_noisy_threshold_' + str(noisy_threshold) + '.npy')
        # 如果文件不存在，则进行detection
        if not os.path.isfile(jump_file) or not os.path.isfile(noisy_file):
            # 检测bad segment
            meg = preprocess(fif_files=[file], out_dir=out_dir, is_log=False, device=cuda_device, n_jobs=10)
            meg.run_multi_fifs(do_tsss=False, do_ica=False, do_resample=False,
                               do_jump_detection=True, jump_threshold=jump_threshold,
                               do_noisy_detection=True, noisy_freq_range=noisy_freq_range,
                               noisy_threshold=noisy_threshold,
                               do_filter=False, filter_band=('IED', 'HFO'))
            del_var(meg.device)
        # 读取bad segment
        jump_artifact = np.load(jump_file)
        noisy_artifact = np.load(noisy_file)
        bad_segments.append(jump_artifact | noisy_artifact)

    return bad_segments


def ictal_ied_detection(raw_data_s_ied, data_info_s, bad_segments, cuda_device=-1, ictal_events=None,
                        # 阈值法 IED检测
                        smooth_windows=0.02, smooth_iterations=2,
                        z_win=30, z_region=True, z_mag_grad=False, exclude_duration=0.02,
                        peak_amplitude_threshold=(2, None), peak_slope_threshold=(None, None),
                        half_peak_slope_threshold=(2, None), peak_sharpness_threshold=(None, None),
                        chan_threshold=(5, 200), data_segment_window=0.1,
                        # 模版匹配法 IED检测
                        large_peak_amplitude_threshold=0.85, large_peak_half_slope_threshold=0.75,
                        large_peak_sharpness_threshold=0.75, large_peak_duration_threshold=100,
                        sim_template=0.85, dist_template=0.85, corr_template=0.85,
                        sim_threshold=0.8, dist_threshold=0.8, corr_threshold=0.8):
    # step1: 使用阈值法检测ied
    ictal_multi = ied_detection_threshold_multi(raw_data_s=raw_data_s_ied, data_info_s=data_info_s,
                                                bad_segments=bad_segments, device=cuda_device)
    ieds_peak_all, ieds_win_all, ieds_peak_closest_idx_all, data_segment_all, ied_amplitude_peaks_all, \
    ied_half_slope_peaks_all, ied_sharpness_peaks_all, ied_duration_peaks_all, _, _, _, _, _, _ = \
        ictal_multi.get_ieds_in_ictal_windows(
            candidate_Ictal_wins=ictal_events,
            smooth_windows=smooth_windows, smooth_iterations=smooth_iterations,
            z_win=z_win, z_region=z_region, z_mag_grad=z_mag_grad, exclude_duration=exclude_duration,
            peak_amplitude_threshold=peak_amplitude_threshold, peak_slope_threshold=peak_slope_threshold,
            half_peak_slope_threshold=half_peak_slope_threshold, peak_sharpness_threshold=peak_sharpness_threshold,
            chan_threshold=chan_threshold, data_segment_window=data_segment_window, ieds_window=0.15)
    del_var(ictal_multi.device)

    # step2: 使用template matching减少假阳性
    ied_cluster = temporal_cluster(device=cuda_device)
    ied_index, clusters_data, clusters_center_data, _, _ = ied_cluster.template_matching(
        ied_amplitude_peaks=torch.cat(ied_amplitude_peaks_all) if None not in ied_amplitude_peaks_all else None,
        large_peak_amplitude_threshold=large_peak_amplitude_threshold,
        ied_half_slope_peaks=torch.cat(ied_half_slope_peaks_all) if None not in ied_half_slope_peaks_all else None,
        large_peak_half_slope_threshold=large_peak_half_slope_threshold,
        ied_duration_peaks=torch.cat(ied_duration_peaks_all) if None not in ied_duration_peaks_all else None,
        large_peak_duration_threshold=int(large_peak_duration_threshold * data_info_s[0]['sfreq']),
        ied_sharpness_peaks=torch.cat(ied_sharpness_peaks_all) if None not in ied_sharpness_peaks_all else None,
        large_peak_sharpness_threshold=large_peak_sharpness_threshold,
        data_segment_all=torch.cat(data_segment_all),
        sim_template=sim_template, dist_template=dist_template, corr_template=corr_template,
        sim_threshold=sim_threshold, dist_threshold=dist_threshold, corr_threshold=corr_threshold,
        channel_threshold=chan_threshold[0])
    del_var(ied_cluster.device)

    # 获取最终的IED主peak位置
    ied_index = ied_index.split([x.shape[0] for x in ieds_peak_all])
    ieds_peak_samples = [x[y].cpu().numpy() for x, y in zip(ieds_peak_all, ied_index)]

    return ieds_peak_samples


def ied_detection(raw_data_s_ied, data_info_s, bad_segments, out_dirs, cuda_device=-1,
                  # 阈值法 IED检测
                  smooth_windows=0.02, smooth_iterations=2,
                  z_win=30, z_region=True, z_mag_grad=False, exclude_duration=0.02,
                  peak_amplitude_threshold=(2, None), peak_slope_threshold=(None, None),
                  half_peak_slope_threshold=(2, None), peak_sharpness_threshold=(None, None),
                  chan_threshold=(5, 200), data_segment_window=0.1,
                  # 模版匹配法 IED检测
                  large_peak_amplitude_threshold=0.85, large_peak_half_slope_threshold=0.75,
                  large_peak_sharpness_threshold=0.75, large_peak_duration_threshold=100,
                  sim_template=0.85, dist_template=0.85, corr_template=0.85,
                  sim_threshold=0.8, dist_threshold=0.8, corr_threshold=0.8, ):
    candidate_ied_wins = None
    # step1: 使用阈值法检测ied
    ied_multi = ied_detection_threshold_multi(raw_data_s=raw_data_s_ied, data_info_s=data_info_s,
                                              bad_segments=bad_segments, device=cuda_device)
    ieds_peak_all, ieds_win_all, _, data_segment_all, ied_amplitude_peaks_all, \
    ied_half_slope_peaks_all, ied_sharpness_peaks_all, ied_duration_peaks_all, _, _, _, _, _, emhapp_save = \
        ied_multi.get_ieds_in_whole_recording(
            smooth_windows=smooth_windows, smooth_iterations=smooth_iterations,
            z_win=z_win, z_region=z_region, z_mag_grad=z_mag_grad, exclude_duration=exclude_duration,
            peak_amplitude_threshold=peak_amplitude_threshold, peak_slope_threshold=peak_slope_threshold,
            half_peak_slope_threshold=half_peak_slope_threshold, peak_sharpness_threshold=peak_sharpness_threshold,
            chan_threshold=chan_threshold, data_segment_window=data_segment_window)
    del_var(ied_multi.device)

    # step2: 使用template matching减少假阳性
    ied_cluster = temporal_cluster(device=cuda_device)
    ied_index, clusters_data, clusters_center_data, corr, dist = ied_cluster.template_matching(
        ied_amplitude_peaks=torch.cat(ied_amplitude_peaks_all) if None not in ied_amplitude_peaks_all else None,
        large_peak_amplitude_threshold=large_peak_amplitude_threshold,
        ied_half_slope_peaks=torch.cat(ied_half_slope_peaks_all) if None not in ied_half_slope_peaks_all else None,
        large_peak_half_slope_threshold=large_peak_half_slope_threshold,
        ied_duration_peaks=torch.cat(ied_duration_peaks_all) if None not in ied_duration_peaks_all else None,
        large_peak_duration_threshold=int(large_peak_duration_threshold * data_info_s[0]['sfreq']),
        ied_sharpness_peaks=torch.cat(ied_sharpness_peaks_all) if None not in ied_sharpness_peaks_all else None,
        large_peak_sharpness_threshold=large_peak_sharpness_threshold,
        data_segment_all=torch.cat(data_segment_all),
        sim_template=sim_template, dist_template=dist_template, corr_template=corr_template,
        sim_threshold=sim_threshold, dist_threshold=dist_threshold, corr_threshold=corr_threshold,
        channel_threshold=chan_threshold[0])
    del_var(ied_cluster.device)

    # 将模版匹配结果加入
    emhapp_save_new = ied_cluster.export_emhapp(emhapp_save=emhapp_save, corr=corr, dist=dist,
                                                ied_num_fif=[x.shape[0] for x in ieds_peak_all],
                                                ied_index=ied_index, clusters_data=clusters_data)
    # 将结果保存为mat，输出到EMHapp中
    for x, y in zip(emhapp_save_new, out_dirs):
        if len(x) > 0:
            scipy.io.savemat(y, x)

    # 获取最终的IED主peak位置
    ied_index = ied_index.split([x.shape[0] for x in ieds_peak_all])
    ieds_peak_samples = [x[y].cpu().numpy() for x, y in zip(ieds_peak_all, ied_index)]

    return ieds_peak_samples, candidate_ied_wins


def vs_reconstruction(raw_data_s, raw_data_s_ied, raw_data_s_hfo, data_info_s, bad_segments,
                      out_dirs, bst_channel,
                      leadfields, ieds_peak_samples, cuda_device=-1,
                      ied_segment_time=2, ied_segment=0.15, ied_peak2peak_windows=0.03, ied_chunk_number=1500,
                      hfo_window=0.1, hfo_segment=0.3, mean_power_windows=0.05, hfo_chunk_number=250):
    ied_max_ori, inv_cov_whole_ied, hfo_max_ori, inv_cov_whole_hfo, ied_segment_samples = [], [], [], [], []
    hfo_window_samples, baseline_window_samples_hfo = [], []
    for raw_data, raw_data_ied, raw_data_hfo, data_info, bad_segment, leadfield, ied_peak_samples, out_dir, i in \
            zip(raw_data_s, raw_data_s_ied, raw_data_s_hfo, data_info_s, bad_segments, leadfields, ieds_peak_samples,
                out_dirs, range(len(raw_data_s))):
        print(data_info['subject_info']['last_name'] + data_info['subject_info']['first_name'] +
              ': VS_Reconstruction', str(i + 1) + '/' + str(len(raw_data_s)))
        if len(ied_peak_samples) > 0:
            # 重建VS信号，IED和HFO频段
            vs = vs_recon(raw_data_ied=raw_data_ied, raw_data_hfo=raw_data_hfo, data_info=data_info,
                          leadfield=leadfield, bad_segment=bad_segment, device=cuda_device)
            emhapp_save, ied_max_ori_temp, inv_cov_whole_ied_temp, hfo_max_ori_temp, inv_cov_whole_hfo_temp, \
            hfo_window_samples, baseline_window_samples_hfo, ied_segment_samples_temp = vs.cal_vs_reconstruction(
                ied_peak_samples, re_lambda=0.05,
                ied_segment_time=ied_segment_time, ied_segment=ied_segment, peak2peak_windows=ied_peak2peak_windows,
                ied_chunk_number=ied_chunk_number,
                hfo_window=hfo_window, hfo_segment=hfo_segment, mean_power_windows=mean_power_windows,
                hfo_chunk_number=hfo_chunk_number)
            hfo_window_samples = hfo_window_samples[0].cpu().numpy()
            baseline_window_samples_hfo = baseline_window_samples_hfo[0].cpu().numpy()
            ied_max_ori.append(ied_max_ori_temp.cpu().numpy())
            inv_cov_whole_ied.append(inv_cov_whole_ied_temp.cpu().numpy())
            hfo_max_ori.append(hfo_max_ori_temp.cpu().numpy())
            inv_cov_whole_hfo.append(inv_cov_whole_hfo_temp.cpu().numpy())
            ied_segment_samples.append(ied_segment_samples_temp.cpu().numpy())
            # 将结果保存为mat，输出到EMHapp中
            emhapp_save.update(bst_channel)
            scipy.io.savemat(out_dir, emhapp_save)
        else:
            ied_max_ori.append([])
            inv_cov_whole_ied.append([])
            hfo_max_ori.append([])
            inv_cov_whole_hfo.append([])
            ied_segment_samples.append([])

    del_var(vs.device)
    return ied_max_ori, inv_cov_whole_ied, hfo_max_ori, inv_cov_whole_hfo, \
           hfo_window_samples, baseline_window_samples_hfo, ied_segment_samples


def hfo_detection(out_dirs, raw_data_s, raw_data_s_ied, raw_data_s_hfo, data_info_s, leadfields, ied_segment_samples,
                  ied_max_ori, inv_cov_whole_ied, hfo_max_ori, inv_cov_whole_hfo,
                  hfo_window_samples, baseline_window_samples_hfo,
                  # clustering-based detector
                  hfo_amplitude_threshold=2, hfo_duration_threshold=0.02, hfo_oscillation_threshold=4,
                  hfo_entropy_threshold=1.25, hfo_power_ratio_threshold=1.25,
                  # clustering-based detector
                  hfo_frequency_range=(80, 200), ied_frequency_range=(3, 80),
                  ied_window=0.1, hfo_window=0.25, tf_bw_threshold=0.5,
                  n_segments_per_event=5, n_component=1, is_whitening=False,
                  gmm_n_init=10, gmm_covariance_type='diag', cuda_device=-1):
    # 使用阈值法和聚类法检测HFO
    hfo = hfo_cluster(raw_data_s=raw_data_s, raw_data_s_ied=raw_data_s_ied, raw_data_s_hfo=raw_data_s_hfo,
                      data_info_s=data_info_s, leadfield_s=leadfields, device=cuda_device)
    emhapp_save, best_cls, cluster, picked_channel_index, cluster_fit_value, fif_index = \
        hfo.cal_hfo_clustering(ied_time_s=ied_segment_samples, max_oris_hfo_s=hfo_max_ori, max_oris_ied_s=ied_max_ori,
                               inv_covariance_hfo_s=inv_cov_whole_hfo, inv_covariance_ied_s=inv_cov_whole_ied,
                               hfo_window_samples=hfo_window_samples,
                               baseline_window_samples_hfo=baseline_window_samples_hfo,
                               hfo_amplitude_threshold=hfo_amplitude_threshold,
                               hfo_duration_threshold=hfo_duration_threshold,
                               hfo_oscillation_threshold=hfo_oscillation_threshold,
                               hfo_entropy_threshold=hfo_entropy_threshold,
                               hfo_power_ratio_threshold=hfo_power_ratio_threshold,
                               hfo_frequency_range=hfo_frequency_range, ied_frequency_range=ied_frequency_range,
                               ied_window=ied_window, hfo_window=hfo_window, tf_bw_threshold=tf_bw_threshold,
                               n_segments_per_event=n_segments_per_event, n_component=n_component,
                               is_whitening=is_whitening, gmm_n_init=gmm_n_init,
                               gmm_covariance_type=gmm_covariance_type)
    if len(emhapp_save):
        emhapp_save = hfo.export_emhapp(emhapp_save=emhapp_save, best_cls=best_cls, cluster=cluster,
                                        picked_channel_index=picked_channel_index, cluster_fit_value=cluster_fit_value)

        # 将结果保存为mat，输出到EMHapp中
        for x, y in zip(emhapp_save, fif_index):
            scipy.io.savemat(out_dirs[y], x)


def run_emhapp_pipeline(fif_files, fif_files_ied, fif_files_hfo, bad_segment_out_dir,
                        cuda_device=1, log_dirs=None,
                        # 预处理
                        do_tsss=False,
                        tsss_auto_bad_lowpass=7., tsss_auto_bad_thre=200.,
                        fine_cal_file=None, crosstalk_file=None, do_ica=False,
                        # 坏段检测
                        jump_threshold=10, noisy_threshold=5, noisy_freq_range=(55, 240),
                        # IED检测: 阈值法
                        mat_files_ied_detection=None,
                        smooth_windows=0.02, smooth_iterations=2,
                        z_win=30, z_region=True, z_mag_grad=False, exclude_duration=0.02,
                        peak_amplitude_threshold=(2, None), peak_slope_threshold=(None, None),
                        half_peak_slope_threshold=(2, None), peak_sharpness_threshold=(None, None),
                        chan_threshold=(5, 200),
                        # IED检测: 模版匹配法
                        data_segment_window=0.1,
                        large_peak_amplitude_threshold=0.85, large_peak_half_slope_threshold=0.75,
                        large_peak_sharpness_threshold=0.75, large_peak_duration_threshold=100,
                        sim_template=0.85, dist_template=0.85, corr_template=0.85,
                        sim_threshold=0.8, dist_threshold=0.8, corr_threshold=0.8,
                        # 虚拟电极重建
                        mat_files_vs_recon=None, bst_channel=None,
                        leadfields=None, ied_segment_time=2,
                        ied_segment=0.15, ied_peak2peak_windows=0.03, ied_chunk_number=1500,
                        hfo_window=0.1, hfo_segment=0.3, mean_power_windows=0.05, hfo_chunk_number=250,
                        # HFO检测: 阈值法
                        mat_files_hfo=None,
                        hfo_amplitude_threshold=2, hfo_duration_threshold=0.02, hfo_oscillation_threshold=4,
                        hfo_entropy_threshold=1.25, hfo_power_ratio_threshold=1.25,
                        # HFO检测: GMM聚类
                        hfo_frequency_range=(80, 200), ied_frequency_range=(3, 80),
                        ied_window=0.1, tf_hfo_window=0.25, tf_bw_threshold=0.5,
                        n_segments_per_event=5, n_component=1, is_whitening=False,
                        gmm_n_init=10, gmm_covariance_type='diag'):
    # 记录开始时间
    if (log_dirs is not None) and (os.path.exists(os.path.split(log_dirs)[0])):
        set_log_file(fname=log_dirs, overwrite=True,
                     output_format='[BEGIN][EMHapp][%(asctime)s ][' +
                                   log_dirs.split('/')[-1].split('.txt')[0] + ']: %(message)s')
        logger.info('Pipeline begin')
    try:
        # step0: 预处理
        fif_out_dir = os.path.split(fif_files[0])[0]
        fif_files = cal_preprocess(fif_files=fif_files, out_dir=fif_out_dir, do_tsss=do_tsss,
                                   tsss_auto_bad_lowpass=tsss_auto_bad_lowpass, tsss_auto_bad_thre=tsss_auto_bad_thre,
                                   fine_cal_file=fine_cal_file, crosstalk_file=crosstalk_file,
                                   do_ica=do_ica,
                                   cuda_device=cuda_device)

        # step1: 坏段检测
        bad_segments = cal_bad_segments(fif_files=fif_files, out_dir=bad_segment_out_dir, jump_threshold=jump_threshold,
                                        noisy_threshold=noisy_threshold, noisy_freq_range=noisy_freq_range,
                                        cuda_device=cuda_device)

        # step2: 读取数据
        raw_data_s, raw_data_s_ied, raw_data_s_hfo, data_info_s, ictal_events, ictal_fif_index = \
            read_meg_data(fif_files=fif_files, fif_files_ied=fif_files_ied, fif_files_hfo=fif_files_hfo)

        # step3: IED检测
        ieds_peak_samples, candidate_ied_wins = ied_detection(
            cuda_device=cuda_device,
            out_dirs=mat_files_ied_detection,
            raw_data_s_ied=raw_data_s_ied, data_info_s=data_info_s, bad_segments=bad_segments,
            smooth_windows=smooth_windows, smooth_iterations=smooth_iterations,
            z_win=z_win, z_region=z_region, z_mag_grad=z_mag_grad, exclude_duration=exclude_duration,
            peak_amplitude_threshold=peak_amplitude_threshold, peak_slope_threshold=peak_slope_threshold,
            half_peak_slope_threshold=half_peak_slope_threshold, peak_sharpness_threshold=peak_sharpness_threshold,
            chan_threshold=chan_threshold, data_segment_window=data_segment_window,
            large_peak_amplitude_threshold=large_peak_amplitude_threshold,
            large_peak_half_slope_threshold=large_peak_half_slope_threshold,
            large_peak_sharpness_threshold=large_peak_sharpness_threshold,
            large_peak_duration_threshold=large_peak_duration_threshold,
            sim_template=sim_template, dist_template=dist_template, corr_template=corr_template,
            sim_threshold=sim_threshold, dist_threshold=dist_threshold, corr_threshold=corr_threshold)

        # step4: 虚拟电极重建
        ied_max_ori, inv_cov_whole_ied, hfo_max_ori, inv_cov_whole_hfo, \
        hfo_window_samples, baseline_window_samples_hfo, ied_segment_samples = vs_reconstruction(
            out_dirs=mat_files_vs_recon, bst_channel=bst_channel, cuda_device=cuda_device,
            raw_data_s=raw_data_s, raw_data_s_ied=raw_data_s_ied, raw_data_s_hfo=raw_data_s_hfo,
            data_info_s=data_info_s,
            bad_segments=bad_segments, leadfields=leadfields, ieds_peak_samples=ieds_peak_samples,
            ied_segment_time=ied_segment_time, ied_segment=ied_segment, ied_peak2peak_windows=ied_peak2peak_windows,
            ied_chunk_number=ied_chunk_number, hfo_window=hfo_window, hfo_segment=hfo_segment,
            mean_power_windows=mean_power_windows, hfo_chunk_number=hfo_chunk_number)

        # step5: HFO检测
        hfo_detection(
            cuda_device=cuda_device,
            out_dirs=mat_files_hfo, raw_data_s=raw_data_s, raw_data_s_ied=raw_data_s_ied, raw_data_s_hfo=raw_data_s_hfo,
            data_info_s=data_info_s, leadfields=leadfields, ied_segment_samples=ied_segment_samples,
            ied_max_ori=ied_max_ori, inv_cov_whole_ied=inv_cov_whole_ied, hfo_max_ori=hfo_max_ori,
            inv_cov_whole_hfo=inv_cov_whole_hfo, hfo_window_samples=hfo_window_samples,
            baseline_window_samples_hfo=baseline_window_samples_hfo, hfo_amplitude_threshold=hfo_amplitude_threshold,
            hfo_duration_threshold=hfo_duration_threshold, hfo_oscillation_threshold=hfo_oscillation_threshold,
            hfo_entropy_threshold=hfo_entropy_threshold, hfo_power_ratio_threshold=hfo_power_ratio_threshold,
            hfo_frequency_range=hfo_frequency_range, ied_frequency_range=ied_frequency_range,
            ied_window=ied_window, hfo_window=tf_hfo_window, tf_bw_threshold=tf_bw_threshold,
            n_segments_per_event=n_segments_per_event, n_component=n_component, is_whitening=is_whitening,
            gmm_n_init=gmm_n_init, gmm_covariance_type=gmm_covariance_type)

        # step6: 保存完成log
        if (log_dirs is not None) and (os.path.exists(os.path.split(log_dirs)[0])):
            set_log_file(fname=log_dirs, overwrite=False,
                         output_format='[DONE][EMHapp][%(asctime)s ][' +
                                       log_dirs.split('/')[-1].split('.txt')[0] + ']: %(message)s')
            logger.info('Pipeline finish')
    except Exception as e:
        # 保存错误
        if (log_dirs is not None) and (os.path.exists(os.path.split(log_dirs)[0])):
            set_log_file(fname=log_dirs, overwrite=False,
                         output_format='[%(levelname)s][EMHapp][%(asctime)s ][' +
                                       log_dirs.split('/')[-1].split('.txt')[0] + ']: <line:%(lineno)d> %(message)s')
            logger.error(e)


# 获取输入
parser = argparse.ArgumentParser(description='运行EMHapp pipeline')
parser.add_argument('--mat', type=str, default=None)
parameters_mat = parser.parse_args().mat
# parameters_mat = '/data2/cuiwei/EMHapp_test/Device4/LNK/Parameters.mat'
# 根据mat文件读取参数
Cuda_device, Leadfields, Bst_channel, File_dirs, Prep_Param, IED_Param, VS_Param, HFO_Param = \
    load_parameters(parameter_dir=parameters_mat)
# 运行
run_emhapp_pipeline(fif_files=File_dirs['fif_files_raw'], fif_files_ied=File_dirs['fif_files_ied'],
                    fif_files_hfo=File_dirs['fif_files_hfo'],
                    bad_segment_out_dir=os.path.split(os.path.split(File_dirs['fif_files_raw'][0])[0])[0],
                    cuda_device=Cuda_device, log_dirs=File_dirs['log_file'],
                    # 预处理
                    do_tsss=Prep_Param['do_tsss'],
                    tsss_auto_bad_lowpass=Prep_Param['tsss_auto_bad_lowpass'],
                    tsss_auto_bad_thre=Prep_Param['tsss_auto_bad_thre'],
                    fine_cal_file=Prep_Param['fine_cal_file'], crosstalk_file=Prep_Param['crosstalk_file'],
                    do_ica=Prep_Param['do_tsss'],
                    # 坏段检测
                    jump_threshold=Prep_Param['jump_threshold'],
                    noisy_threshold=Prep_Param['noisy_threshold'], noisy_freq_range=Prep_Param['noisy_freq_range'],
                    # IED检测: 阈值法
                    mat_files_ied_detection=File_dirs['mat_files_ied'],
                    smooth_windows=0.02, smooth_iterations=2,
                    z_win=IED_Param['z_win'], z_region=True, z_mag_grad=False,
                    exclude_duration=IED_Param['exclude_duration'],
                    peak_amplitude_threshold=IED_Param['peak_amplitude_threshold'],
                    peak_slope_threshold=(None, None),
                    half_peak_slope_threshold=IED_Param['half_peak_slope_threshold'],
                    peak_sharpness_threshold=[0.5, None] if IED_Param['peak_sharpness_threshold'] == [None, None] else
                    IED_Param['peak_sharpness_threshold'],
                    chan_threshold=IED_Param['chan_threshold'],
                    # IED检测: 模版匹配法
                    data_segment_window=IED_Param['data_segment_window'],
                    large_peak_amplitude_threshold=IED_Param['large_peak_amplitude_threshold'],
                    large_peak_half_slope_threshold=IED_Param['large_peak_half_slope_threshold'],
                    large_peak_sharpness_threshold=IED_Param['large_peak_sharpness_threshold'],
                    large_peak_duration_threshold=IED_Param['large_peak_duration_threshold'],
                    sim_template=IED_Param['sim_template'], dist_template=IED_Param['dist_template'],
                    corr_template=IED_Param['corr_template'],
                    sim_threshold=IED_Param['sim_threshold'], dist_threshold=IED_Param['dist_threshold'],
                    corr_threshold=IED_Param['corr_threshold'],
                    # 虚拟电极重建
                    leadfields=Leadfields,
                    bst_channel=Bst_channel, mat_files_vs_recon=File_dirs['mat_files_vs_recon'],
                    ied_segment_time=VS_Param['ied_segment_time'], ied_segment=VS_Param['ied_segment'],
                    ied_peak2peak_windows=VS_Param['ied_peak2peak_windows'], ied_chunk_number=200,
                    hfo_window=VS_Param['hfo_window'], hfo_segment=VS_Param['hfo_segment'],
                    mean_power_windows=VS_Param['mean_power_windows'], hfo_chunk_number=40,
                    # HFO检测: 阈值法
                    mat_files_hfo=File_dirs['mat_files_hfo'],
                    hfo_amplitude_threshold=HFO_Param['hfo_amplitude_threshold'],
                    hfo_duration_threshold=HFO_Param['hfo_duration_threshold'],
                    hfo_oscillation_threshold=HFO_Param['hfo_oscillation_threshold'],
                    hfo_entropy_threshold=HFO_Param['hfo_entropy_threshold'],
                    hfo_power_ratio_threshold=HFO_Param['hfo_power_ratio_threshold'],
                    # HFO检测: GMM聚类
                    hfo_frequency_range=Prep_Param['hfo_frequency_range'],
                    ied_frequency_range=Prep_Param['ied_frequency_range'],
                    ied_window=HFO_Param['ied_window'],
                    tf_hfo_window=HFO_Param['tf_hfo_window'], tf_bw_threshold=0.5,
                    n_segments_per_event=HFO_Param['n_segments_per_event'],
                    n_component=HFO_Param['n_component'], is_whitening=HFO_Param['is_whitening'],
                    gmm_n_init=HFO_Param['gmm_n_init'], gmm_covariance_type=HFO_Param['gmm_covariance_type'])
