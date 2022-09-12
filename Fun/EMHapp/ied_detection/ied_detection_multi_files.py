# -*- coding:utf-8 -*-
# @Time    : 2022/2/4
# @Author  : cuiwei
# @File    : ied_detection_multi_files.py
# @Software: PyCharm
# @Script to:
#   - 对于多个MEG文件，使用阈值法检测IED

import os
import mne
import numpy as np
import torch
from ied_detection import ied_detection_threshold
from ied_detection import ied_peak_feature
import matplotlib.pyplot as plt


class ied_detection_threshold_multi:
    def __init__(self, raw_data_s=None, data_info_s=None, bad_segments=None, device=-1, n_jobs=10):
        """
        Description:
            初始化，使用GPU或者CPU。

        Input:
            :param raw_data_s: ndarray, double, shape(n_files, channel, samples)
                MEG滤波后数据
            :param data_info_s: list, dict, shape(n_files)
                MEG数据的信息, MNE读取的raw.info
            :param bad_segments: list, [ndarray, bool, shape(n_samples)], shape(n_files)
                每个MEG文件的bad segment
            :param device: number, int
                device<0 使用CPU, device>=0 使用对应GPU
            :param n_jobs: number, int
                MEN函数中，使用到的并行个数
        """
        # 使用GPU或者CPU
        self.device_number = device
        self.n_jobs = n_jobs
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        self._check_cuda(device)
        # 变量初始化
        self.raw_data_s, self.data_info_s, self.bad_segments = raw_data_s, data_info_s, bad_segments
        self.set_raw_data(raw_data_s)
        self.set_data_info(data_info_s)
        self.set_bad_segment(bad_segments)

    def _check_cuda(self, device):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
        if device > -1:
            # Init MEN cuda
            try:
                mne.cuda.set_cuda_device(device, verbose=False)
                mne.utils.set_config('MNE_USE_CUDA', 'true', verbose=False)
                mne.cuda.init_cuda(verbose=False)
                self.is_cuda = 1
            except:
                self.is_cuda = 0
            # Init torch cuda
            if torch.cuda.is_available():
                # Init torch
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")
            self.is_cuda = 0

    def del_var(self, *arg):
        """
        Description:
            清除变量，释放缓存

        Input:
            :param arg: list, string
                需要清除的变量名称
        """
        if arg is not None:
            for key in list(globals().keys()):
                if key in arg:
                    globals()[key] = []
        if self.device == torch.device("cuda"):
            torch.cuda.empty_cache()

    def set_raw_data(self, raw_data_s=None):
        assert raw_data_s is not None
        if type(raw_data_s) == type([]):
            self.raw_data_s = raw_data_s
        else:
            self.raw_data_s = [raw_data_s]

    def get_raw_data(self):
        return self.raw_data_s

    def set_data_info(self, data_info_s=None):
        assert data_info_s is not None
        if type(data_info_s) == type([]):
            self.data_info_s = data_info_s
        else:
            self.data_info_s = [data_info_s]

    def get_data_info(self):
        return self.data_info_s

    def set_bad_segment(self, bad_segments=None):
        if bad_segments is None:
            self.bad_segments = [None for _ in range(len(self.get_raw_data()))]
        else:
            self.bad_segments = bad_segments

    def get_bad_segment(self):
        return self.bad_segments

    def get_ieds_in_whole_recording(self, smooth_windows=0.02, smooth_iterations=2, z_win=30,
                                    z_region=True, z_mag_grad=False, exclude_duration=0.02,
                                    peak_amplitude_threshold=(2, None), peak_slope_threshold=(None, None),
                                    half_peak_slope_threshold=(2, None), peak_sharpness_threshold=(None, None),
                                    mag_grad_ratio=0.056, chan_threshold=(5, 200), ieds_window=0.3,
                                    data_segment_window=0.1):
        """
        Description:
            对于多个文件，计算整段数据中所有的ieds，并返回ieds的peak和windows，以及的特征值。

        Input:
            ---------------对于输入MEG的每一个通道信号，重建出波峰波谷的采样点位置以及对应信号强度---------------
            :param smooth_windows: number, float
                平滑窗长度(毫秒)。略高于需要平滑掉的噪声峰持续时间。
            :param smooth_iterations: number, int
                平滑次数。
            ---------------------------------对所有features进行z-score--------------------------------
            :param exclude_duration: number, double
                在计算z-score中的均值和方差之前，去除持续时间小于该阈值的峰
            :param z_win: number, double
                选取时间上计算z-score中的均值和方差
                滑动窗长度：z_win=大于1，时间窗为z_win；z_ win=0到1之间，时间窗为z_win*数据总长；z_win=None，时间窗为数据总长。
            :param z_region: bool
                是否通过脑区选择多通道。
            :param z_mag_grad: bool
                是否通过MAG/GRAD选择多通道。
            -----------------------对所有feature进行阈值处理，获得满足阈值要求的peaks----------------------
            :param peak_amplitude_threshold: list, double, shape(2 * 1)
                峰峰值阈值：[下阈值， 上阈值]
            :param peak_slope_threshold: list, double, shape(2 * 1)
                斜率阈值：[下阈值， 上阈值]
            :param half_peak_slope_threshold: list, double, shape(2 * 1)
                半高宽斜率阈值：[下阈值， 上阈值]
            :param peak_sharpness_threshold: list, double, shape(2 * 1)
                锐度阈值：[下阈值， 上阈值]
            -----------------------根据过阈值的peaks，找出IED event，以及对应的峰值-----------------------
            :param mag_grad_ratio: number, double
                mag和grad的比值，用于rescale mag和grad到同一scale
            :param chan_threshold: list, long, shape(2); or number, long
                需要满足的channel个数，每个时间内检测为IED的channel个数。
                为list的时候，channel的上限和下限；为number的时候，channel下限。
            :param ieds_window: number, double
                输出ieds的时间窗长度，单位s
            :param data_segment_window: number, double
                截取ied数据片段的长度，用于后续的时间聚类，单位s
        Return:
            :return ieds_peak_all: list, shape(n_fifs: list, long, shape(n_ieds))
                每个fif文件，检测到的ied events的peak位置
            :return ieds_win_all: list, shape(n_fifs: list, long, shape(n_ieds))
                每个fif文件，检测到的ieds events的duration
            :return ieds_peak_closest_idx_all: list, shape(n_fifs: torch.tensor, double, shape(n*n_channel*shift个数))
                每个fif文件的每个通道中，ied peak两侧最近的峰，在raw meg中的采样点位置
            :return data_segment_all: list, shape(n_fifs: torch.tensor, double,
                                                          shape(n*n_channel*shift个数*data_segment_window))
                每个fif文件，截取每个ied片段的数据。
            :return ied_amplitude_peaks_all: list, shape(n_fifs: torch.tensor, double, shape(n_ieds*n_channel*2))
                每个fif文件, 每个通道，ieds主峰值两侧peaks的幅度
            :return ied_half_slope_peaks: list, shape(n_fifs: torch.tensor, double, shape(n_ieds*n_channel*2))
                每个fif文件, 每个通道，ieds主峰值两侧peaks的斜率
            :return ied_sharpness_peaks: list, shape(n_fifs: torch.tensor, double, shape(n_ieds*n_channel*2))
                每个fif文件, 每个通道，ieds主峰值两侧peaks的锐度
            :return ied_duration_peaks: list, shape(n_fifs: torch.tensor, double, shape(n_ieds*n_channel*2))
                每个fif文件, 每个通道，ieds主峰值两侧peaks的峰持续时间
            :return ied_amplitude: list, shape(n_fifs: torch.tensor, double, shape(n_ieds))
                每个fif文件, 每个ieds的幅度
            :return ied_half_slope: list, shape(n_fifs: torch.tensor, double, shape(n_ieds))
                每个fif文件, 每个ieds的斜率
            :return ied_sharpness: list, shape(n_fifs: torch.tensor, double, shape(n_ieds))
                每个fif文件, 每个ieds的锐度
            :return ied_duration: list, shape(n_fifs: torch.tensor, double, shape(n_ieds))
                每个fif文件, 每个ieds的峰持续时间
            :return ied_gfp_amp: list, shape(n_fifs: torch.tensor, double, shape(n_ieds))
                每个fif文件, 每个ieds的gfp值
            :return emhapp_save: list, shape(n_fifs: dict)
                用于保存emhapp mat文件的字典

        """
        ieds_peak_all, ieds_win_all, ieds_peak_closest_idx_all, data_segment_all, data_segment_large_all, \
        ied_amplitude_peaks_all, ied_half_slope_peaks_all, ied_sharpness_peaks_all, ied_duration_peaks_all, \
        ied_amplitude_all, ied_half_slope_all, ied_sharpness_all, ied_duration_all, ied_gfp_amp_all, emhapp_save = \
            [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

        for data_raw, data_info, bad_segment, i in zip(self.get_raw_data(), self.get_data_info(),
                                                       self.get_bad_segment(), range(len(self.get_bad_segment()))):
            print(data_info['subject_info']['last_name'] + data_info['subject_info']['first_name']
                  + ': IED_Detection(Threshold)', str(i + 1) + '/' + str(len(self.get_bad_segment())))
            find_peaks = ied_peak_feature.cal_peak_feature(raw_data=data_raw, data_info=data_info,
                                                           bad_segment=bad_segment, device=self.device_number,
                                                           n_jobs=self.n_jobs)
            # 计算每个fif文件的peaks，以及对应特征值
            peaks = find_peaks.get_peaks(smooth_windows=smooth_windows, smooth_iterations=smooth_iterations,
                                         z_win=z_win, z_region=z_region, z_mag_grad=z_mag_grad,
                                         exclude_duration=exclude_duration,
                                         peak_amplitude_threshold=peak_amplitude_threshold,
                                         peak_slope_threshold=peak_slope_threshold,
                                         half_peak_slope_threshold=half_peak_slope_threshold,
                                         peak_sharpness_threshold=peak_sharpness_threshold)
            # 根据特征值以及阈值，获取每个IED event的peak位置
            ieds = ied_detection_threshold.ied_detection_threshold(raw_data=data_raw, data_info=data_info,
                                                                   device=self.device_number, n_jobs=self.n_jobs)
            ieds_peaks_temp, ieds_win_temp = \
                ieds.get_ieds_in_all_meg(mag_grad_ratio=mag_grad_ratio, chan_threshold=chan_threshold,
                                         ieds_window=ieds_window,
                                         peak_amplitude=peaks['peak_amplitude'],
                                         half_peak_slope=peaks['half_peak_slope'],
                                         win_samples_of_picked_peak=peaks['win_samples_of_picked_peak'],
                                         all_samples_of_picked_peak=peaks['all_samples_of_picked_peak'])
            ieds_peak_closest_idx_temp, data_segment_temp = \
                ieds.cal_ied_segments(ieds_peaks=ieds_peaks_temp, ieds_win=ieds_win_temp,
                                      samples_of_picked_peak=torch.tensor(peaks['samples_of_picked_peak']),
                                      data_segment_window=data_segment_window)
            # 计算IED event的feature
            ied_amplitude_peaks_temp, ied_half_slope_peaks_temp, ied_sharpness_peaks_temp, ied_duration_peaks_temp, \
            ied_amplitude_temp, ied_half_slope_temp, ied_sharpness_temp, ied_duration_temp, ied_gfp_amp_temp = \
                ieds.cal_ieds_features(samples_of_picked_peak=torch.tensor(peaks['samples_of_picked_peak']),
                                       ieds_peak_closest_idx=ieds_peak_closest_idx_temp,
                                       peak_amplitude=peaks['peak_amplitude'], half_peak_slope=peaks['half_peak_slope'],
                                       peak_duration=peaks['peak_duration'], peak_sharpness=peaks['peak_sharpness'],
                                       data_gfp=ieds.get_data_gfp()[0], ieds_peaks=ieds_peaks_temp,
                                       channel_used_for_average=chan_threshold[0])

            # 如果没有检测到ied，将输出的None变为torch.tensor([])
            if ieds_peaks_temp is None:
                ieds_peaks_temp, ieds_win_temp, ied_amplitude_peaks_temp, ied_half_slope_peaks_temp, \
                ied_sharpness_peaks_temp, ied_duration_peaks_temp, ied_amplitude_temp, ied_half_slope_temp, \
                ied_sharpness_temp, ied_duration_temp, ied_gfp_amp_temp, ieds_peak_closest_idx_temp, \
                data_segment_temp, emhapp_save_temp = \
                    torch.tensor([]), torch.tensor([]), torch.tensor([]).to(self.device), \
                    torch.tensor([]).to(self.device), torch.tensor([]).to(self.device), \
                    torch.tensor([]).to(self.device), torch.tensor([]).to(self.device), \
                    torch.tensor([]).to(self.device), torch.tensor([]).to(self.device), \
                    torch.tensor([]).to(self.device), torch.tensor([]).to(self.device), \
                    torch.tensor([]).to(self.device), torch.tensor([]).to(self.device), []
            else:
                # 生成用于GUI的mat文件
                emhapp_save_temp = self.export_emhapp(info=data_info, meg_size=data_raw.shape[1],
                                                      ieds_peak=ieds_peaks_temp, ied_amplitude=ied_amplitude_peaks_temp,
                                                      ied_sharpness=ied_sharpness_peaks_temp,
                                                      ied_half_slope=ied_half_slope_peaks_temp)

            # 输出
            ieds_peak_all.append(ieds_peaks_temp)
            ieds_win_all.append(ieds_win_temp)
            ieds_peak_closest_idx_all.append(ieds_peak_closest_idx_temp)
            data_segment_all.append(data_segment_temp)
            ied_amplitude_peaks_all.append(ied_amplitude_peaks_temp)
            ied_half_slope_peaks_all.append(ied_half_slope_peaks_temp)
            ied_sharpness_peaks_all.append(ied_sharpness_peaks_temp)
            ied_duration_peaks_all.append(ied_duration_peaks_temp)
            ied_amplitude_all.append(ied_amplitude_temp)
            ied_half_slope_all.append(ied_half_slope_temp)
            ied_sharpness_all.append(ied_sharpness_temp)
            ied_duration_all.append(ied_duration_temp)
            ied_gfp_amp_all.append(ied_gfp_amp_temp)
            emhapp_save.append(emhapp_save_temp)

        return ieds_peak_all, ieds_win_all, ieds_peak_closest_idx_all, data_segment_all, \
               ied_amplitude_peaks_all, ied_half_slope_peaks_all, ied_sharpness_peaks_all, ied_duration_peaks_all, \
               ied_amplitude_all, ied_half_slope_all, ied_sharpness_all, ied_duration_all, ied_gfp_amp_all, emhapp_save

    def get_ieds_in_ictal_windows(self, candidate_Ictal_wins=None,
                                  smooth_windows=0.02, smooth_iterations=2, z_win=30,
                                  z_region=True, z_mag_grad=False, exclude_duration=0.02,
                                  peak_amplitude_threshold=(2, None), peak_slope_threshold=(None, None),
                                  half_peak_slope_threshold=(2, None), peak_sharpness_threshold=(None, None),
                                  mag_grad_ratio=0.056, chan_threshold=(5, 200), ieds_window=0.15,
                                  data_segment_window=0.1):
        """
        Description:
            对于多个文件，计算发作时间段内所有的ieds，并返回ieds的peak和windows，以及的特征值。

        Input:
            ---------------对于输入MEG的每一个通道信号，重建出波峰波谷的采样点位置以及对应信号强度---------------
            :param smooth_windows: number, float
                平滑窗长度(毫秒)。略高于需要平滑掉的噪声峰持续时间。
            :param smooth_iterations: number, int
                平滑次数。
            ---------------------------------对所有features进行z-score--------------------------------
            :param exclude_duration: number, double
                在计算z-score中的均值和方差之前，去除持续时间小于该阈值的峰
            :param z_win: number, double
                选取时间上计算z-score中的均值和方差
                滑动窗长度：z_win=大于1，时间窗为z_win；z_ win=0到1之间，时间窗为z_win*数据总长；z_win=None，时间窗为数据总长。
            :param z_region: bool
                是否通过脑区选择多通道。
            :param z_mag_grad: bool
                是否通过MAG/GRAD选择多通道。
            -----------------------对所有feature进行阈值处理，获得满足阈值要求的peaks----------------------
            :param peak_amplitude_threshold: list, double, shape(2 * 1)
                峰峰值阈值：[下阈值， 上阈值]
            :param peak_slope_threshold: list, double, shape(2 * 1)
                斜率阈值：[下阈值， 上阈值]
            :param half_peak_slope_threshold: list, double, shape(2 * 1)
                半高宽斜率阈值：[下阈值， 上阈值]
            :param peak_sharpness_threshold: list, double, shape(2 * 1)
                锐度阈值：[下阈值， 上阈值]
            -----------------------根据过阈值的peaks，找出IED event，以及对应的峰值-----------------------
            :param candidate_Ictal_wins: list, shape(n_fifs: np.array, long, shape(n_ictal_segments)）
                每个fif文件，发作的ied时间段在原始MEG数据中的采样点位置。
            :param mag_grad_ratio: number, double
                mag和grad的比值，用于rescale mag和grad到同一scale
            :param chan_threshold: list, long, shape(2); or number, long
                需要满足的channel个数，每个时间内检测为IED的channel个数。
                为list的时候，channel的上限和下限；为number的时候，channel下限。
            :param ieds_window: number, double
                输出ieds的时间窗长度，单位s
            :param data_segment_window: number, double
                截取ied数据片段的长度，用于后续的时间聚类，单位s
        Return:
            :return ieds_peak_all: list, shape(n_fifs: list, long, shape(n_ieds))
                每个fif文件，检测到的ied events的peak位置
            :return ieds_win_all: list, shape(n_fifs: list, long, shape(n_ieds))
                每个fif文件，检测到的ieds events的duration
            :return ieds_peak_closest_idx_all: list, shape(n_fifs: torch.tensor, double, shape(n*n_channel*shift个数))
                每个fif文件的每个通道中，ied peak两侧最近的峰，在raw meg中的采样点位置
            :return data_segment_all: list, shape(n_fifs: torch.tensor, double,
                                                          shape(n*n_channel*shift个数*data_segment_window))
                每个fif文件，截取每个ied片段的数据。
            :return ied_amplitude_peaks_all: list, shape(n_fifs: torch.tensor, double, shape(n_ieds*n_channel*2))
                每个fif文件, 每个通道，ieds主峰值两侧peaks的幅度
            :return ied_half_slope_peaks: list, shape(n_fifs: torch.tensor, double, shape(n_ieds*n_channel*2))
                每个fif文件, 每个通道，ieds主峰值两侧peaks的斜率
            :return ied_sharpness_peaks: list, shape(n_fifs: torch.tensor, double, shape(n_ieds*n_channel*2))
                每个fif文件, 每个通道，ieds主峰值两侧peaks的锐度
            :return ied_duration_peaks: list, shape(n_fifs: torch.tensor, double, shape(n_ieds*n_channel*2))
                每个fif文件, 每个通道，ieds主峰值两侧peaks的峰持续时间
            :return ied_amplitude: list, shape(n_fifs: torch.tensor, double, shape(n_ieds))
                每个fif文件, 每个ieds的幅度
            :return ied_half_slope: list, shape(n_fifs: torch.tensor, double, shape(n_ieds))
                每个fif文件, 每个ieds的斜率
            :return ied_sharpness: list, shape(n_fifs: torch.tensor, double, shape(n_ieds))
                每个fif文件, 每个ieds的锐度
            :return ied_duration: list, shape(n_fifs: torch.tensor, double, shape(n_ieds))
                每个fif文件, 每个ieds的峰持续时间
            :return ied_gfp_amp: list, shape(n_fifs: torch.tensor, double, shape(n_ieds))
                每个fif文件, 每个ieds的gfp值
            :return emhapp_save: list, shape(n_fifs: dict)
                用于保存emhapp mat文件的字典

        """
        # 判断输入是否存在
        if candidate_Ictal_wins is None or len(candidate_Ictal_wins) != len(self.get_data_info()):
            return None, None, None, None, None, None, None, None, None, None, None, None, None

        ieds_peak_all, ieds_win_all, ieds_peak_closest_idx_all, data_segment_all, data_segment_large_all, \
        ied_amplitude_peaks_all, ied_half_slope_peaks_all, ied_sharpness_peaks_all, ied_duration_peaks_all, \
        ied_amplitude_all, ied_half_slope_all, ied_sharpness_all, ied_duration_all, ied_gfp_amp_all, emhapp_save = \
            [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        for data_raw, data_info, bad_segment, candidate_Ictal_win, i in \
                zip(self.get_raw_data(), self.get_data_info(), self.get_bad_segment(), candidate_Ictal_wins,
                    range(len(candidate_Ictal_wins))):
            print(data_info['subject_info']['last_name'] + ': Ictal_IED_Detection(Threshold)',
                  str(i + 1) + '/' + str(len(candidate_Ictal_wins)))
            find_peaks = ied_peak_feature.cal_peak_feature(raw_data=data_raw, data_info=data_info,
                                                           bad_segment=bad_segment, device=self.device_number,
                                                           n_jobs=self.n_jobs)
            # 计算每个fif文件的peaks，以及对应特征值
            peaks = find_peaks.get_peaks(smooth_windows=smooth_windows, smooth_iterations=smooth_iterations,
                                         z_win=z_win, z_region=z_region, z_mag_grad=z_mag_grad,
                                         exclude_duration=exclude_duration,
                                         peak_amplitude_threshold=peak_amplitude_threshold,
                                         peak_slope_threshold=peak_slope_threshold,
                                         half_peak_slope_threshold=half_peak_slope_threshold,
                                         peak_sharpness_threshold=peak_sharpness_threshold)
            # 根据特征值以及阈值，获取每个IED event的peak位置
            ieds = ied_detection_threshold.ied_detection_threshold(raw_data=data_raw, data_info=data_info,
                                                                   device=self.device_number, n_jobs=self.n_jobs)
            ieds_peaks_temp, ieds_win_temp = \
                ieds.get_ieds_in_ictal_windows(mag_grad_ratio=mag_grad_ratio, chan_threshold=chan_threshold,
                                               candidate_ictal_wins=candidate_Ictal_win, ieds_window=ieds_window,
                                               peak_amplitude=peaks['peak_amplitude'],
                                               half_peak_slope=peaks['half_peak_slope'],
                                               win_samples_of_picked_peak=peaks['win_samples_of_picked_peak'],
                                               all_samples_of_picked_peak=peaks['all_samples_of_picked_peak'])
            ieds_peak_closest_idx_temp, data_segment_temp = \
                ieds.cal_ied_segments(ieds_peaks=ieds_peaks_temp, ieds_win=ieds_win_temp,
                                      samples_of_picked_peak=torch.tensor(peaks['samples_of_picked_peak']),
                                      data_segment_window=data_segment_window)
            # 计算IED event的feature
            ied_amplitude_peaks_temp, ied_half_slope_peaks_temp, ied_sharpness_peaks_temp, ied_duration_peaks_temp, \
            ied_amplitude_temp, ied_half_slope_temp, ied_sharpness_temp, ied_duration_temp, ied_gfp_amp_temp = \
                ieds.cal_ieds_features(samples_of_picked_peak=torch.tensor(peaks['samples_of_picked_peak']),
                                       ieds_peak_closest_idx=ieds_peak_closest_idx_temp,
                                       peak_amplitude=peaks['peak_amplitude'], half_peak_slope=peaks['half_peak_slope'],
                                       peak_duration=peaks['peak_duration'], peak_sharpness=peaks['peak_sharpness'],
                                       data_gfp=ieds.get_data_gfp()[0], ieds_peaks=ieds_peaks_temp,
                                       channel_used_for_average=chan_threshold[0])

            # 如果没有检测到ied，将输出的None变为torch.tensor([])
            if ieds_peaks_temp is None:
                ieds_peaks_temp, ieds_win_temp, ied_amplitude_peaks_temp, ied_half_slope_peaks_temp, \
                ied_sharpness_peaks_temp, ied_duration_peaks_temp, ied_amplitude_temp, ied_half_slope_temp, \
                ied_sharpness_temp, ied_duration_temp, ied_gfp_amp_temp, ieds_peak_closest_idx_temp, \
                data_segment_temp, emhapp_save_temp = \
                    torch.tensor([]), torch.tensor([]), torch.tensor([]).to(self.device), \
                    torch.tensor([]).to(self.device), torch.tensor([]).to(self.device), \
                    torch.tensor([]).to(self.device), torch.tensor([]).to(self.device), \
                    torch.tensor([]).to(self.device), torch.tensor([]).to(self.device), \
                    torch.tensor([]).to(self.device), torch.tensor([]).to(self.device), \
                    torch.tensor([]).to(self.device), torch.tensor([]).to(self.device), []
            else:
                # 生成用于GUI的mat文件
                emhapp_save_temp = self.export_emhapp(info=data_info, meg_size=data_raw.shape[1],
                                                      ieds_peak=ieds_peaks_temp, ied_amplitude=ied_amplitude_peaks_temp,
                                                      ied_sharpness=ied_sharpness_peaks_temp,
                                                      ied_half_slope=ied_half_slope_peaks_temp)

            # 输出
            ieds_peak_all.append(ieds_peaks_temp)
            ieds_win_all.append(ieds_win_temp)
            ieds_peak_closest_idx_all.append(ieds_peak_closest_idx_temp)
            data_segment_all.append(data_segment_temp)
            ied_amplitude_peaks_all.append(ied_amplitude_peaks_temp)
            ied_half_slope_peaks_all.append(ied_half_slope_peaks_temp)
            ied_sharpness_peaks_all.append(ied_sharpness_peaks_temp)
            ied_duration_peaks_all.append(ied_duration_peaks_temp)
            ied_amplitude_all.append(ied_amplitude_temp)
            ied_half_slope_all.append(ied_half_slope_temp)
            ied_sharpness_all.append(ied_sharpness_temp)
            ied_duration_all.append(ied_duration_temp)
            ied_gfp_amp_all.append(ied_gfp_amp_temp)
            emhapp_save.append(emhapp_save_temp)

        return ieds_peak_all, ieds_win_all, ieds_peak_closest_idx_all, data_segment_all, \
               ied_amplitude_peaks_all, ied_half_slope_peaks_all, ied_sharpness_peaks_all, ied_duration_peaks_all, \
               ied_amplitude_all, ied_half_slope_all, ied_sharpness_all, ied_duration_all, ied_gfp_amp_all, emhapp_save

    def get_ieds_in_candidate_ieds_windows(self, candidate_ied_wins=None, is_unique=True,
                                           smooth_windows=0.02, smooth_iterations=2, z_win=30,
                                           z_region=True, z_mag_grad=False, exclude_duration=0.02,
                                           peak_amplitude_threshold=(2, None), peak_slope_threshold=(None, None),
                                           half_peak_slope_threshold=(2, None), peak_sharpness_threshold=(None, None),
                                           mag_grad_ratio=0.056, chan_threshold=(5, 200),
                                           data_segment_window=0.1):
        """
        Description:
            对于多个文件，计算发作时间段内所有的ieds，并返回ieds的peak和windows，以及的特征值。

        Input:
            :param is_unique: number, bool
                对peaks是否取唯一
            ---------------对于输入MEG的每一个通道信号，重建出波峰波谷的采样点位置以及对应信号强度---------------
            :param smooth_windows: number, float
                平滑窗长度(毫秒)。略高于需要平滑掉的噪声峰持续时间。
            :param smooth_iterations: number, int
                平滑次数。
            ---------------------------------对所有features进行z-score--------------------------------
            :param exclude_duration: number, double
                在计算z-score中的均值和方差之前，去除持续时间小于该阈值的峰
            :param z_win: number, double
                选取时间上计算z-score中的均值和方差
                滑动窗长度：z_win=大于1，时间窗为z_win；z_ win=0到1之间，时间窗为z_win*数据总长；z_win=None，时间窗为数据总长。
            :param z_region: bool
                是否通过脑区选择多通道。
            :param z_mag_grad: bool
                是否通过MAG/GRAD选择多通道。
            -----------------------对所有feature进行阈值处理，获得满足阈值要求的peaks----------------------
            :param peak_amplitude_threshold: list, double, shape(2 * 1)
                峰峰值阈值：[下阈值， 上阈值]
            :param peak_slope_threshold: list, double, shape(2 * 1)
                斜率阈值：[下阈值， 上阈值]
            :param half_peak_slope_threshold: list, double, shape(2 * 1)
                半高宽斜率阈值：[下阈值， 上阈值]
            :param peak_sharpness_threshold: list, double, shape(2 * 1)
                锐度阈值：[下阈值， 上阈值]
            -----------------------根据过阈值的peaks，找出IED event，以及对应的峰值-----------------------
            :param candidate_ied_wins: list, [np.array, long, shape(n_ieds*n_ied_samples)], shape(n_fifs)
                每个fif文件，每个ied时间窗在原始MEG数据中的采样点位置。
            :param mag_grad_ratio: number, double
                mag和grad的比值，用于rescale mag和grad到同一scale
            :param chan_threshold: list, long, shape(2); or number, long
                需要满足的channel个数，每个时间内检测为IED的channel个数。
                为list的时候，channel的上限和下限；为number的时候，channel下限。
            :param data_segment_window: number, double
                截取ied数据片段的长度，用于后续的时间聚类，单位s
        Return:
            :return is_ieds_all: list, shape(n_fifs: list, bool, shape(n_ieds)
                每个fif文件，输入的ied events是否被检测为ied
            :return ieds_peak_all: list, shape(n_fifs: list, long, shape(n_ieds))
                每个fif文件，检测到的ied events的peak位置
            :return ieds_win_all: list, shape(n_fifs: list, long, shape(n_ieds))
                每个fif文件，检测到的ieds events的duration
            :return ieds_peak_closest_idx_all: list, shape(n_fifs: torch.tensor, double, shape(n*n_channel*shift个数))
                每个fif文件的每个通道中，ied peak两侧最近的峰，在raw meg中的采样点位置
            :return data_segment_all: list, shape(n_fifs: torch.tensor, double,
                                                          shape(n*n_channel*shift个数*data_segment_window))
                每个fif文件，截取每个ied片段的数据。
            :return ied_amplitude_peaks_all: list, shape(n_fifs: torch.tensor, double, shape(n_ieds*n_channel*2))
                每个fif文件, 每个通道，ieds主峰值两侧peaks的幅度
            :return ied_half_slope_peaks_all: list, shape(n_fifs: torch.tensor, double, shape(n_ieds*n_channel*2))
                每个fif文件, 每个通道，ieds主峰值两侧peaks的斜率
            :return ied_sharpness_peaks_all: list, shape(n_fifs: torch.tensor, double, shape(n_ieds*n_channel*2))
                每个fif文件, 每个通道，ieds主峰值两侧peaks的锐度
            :return ied_duration_peaks_all: list, shape(n_fifs: torch.tensor, double, shape(n_ieds*n_channel*2))
                每个fif文件, 每个通道，ieds主峰值两侧peaks的峰持续时间
            :return ied_amplitude: list, shape(n_fifs: torch.tensor, double, shape(n_ieds))
                每个fif文件, 每个ieds的幅度
            :return ied_half_slope: list, shape(n_fifs: torch.tensor, double, shape(n_ieds))
                每个fif文件, 每个ieds的斜率
            :return ied_sharpness: list, shape(n_fifs: torch.tensor, double, shape(n_ieds))
                每个fif文件, 每个ieds的锐度
            :return ied_duration: list, shape(n_fifs: torch.tensor, double, shape(n_ieds))
                每个fif文件, 每个ieds的峰持续时间
            :return ied_gfp_amp: list, shape(n_fifs: torch.tensor, double, shape(n_ieds))
                每个fif文件, 每个ieds的gfp值
            :return emhapp_save: list, shape(n_fifs: dict)
                用于保存emhapp mat文件的字典

        """
        # 判断输入是否存在
        if candidate_ied_wins is None or len(candidate_ied_wins) != len(self.get_data_info()):
            return None, None, None, None, None, None, None, None, None, None, None, None, None, None

        is_ieds_all, ieds_peak_all, ieds_win_all, ieds_peak_closest_idx_all, data_segment_all, data_segment_large_all, \
        ied_amplitude_peaks_all, ied_half_slope_peaks_all, ied_sharpness_peaks_all, ied_duration_peaks_all, \
        ied_amplitude_all, ied_half_slope_all, ied_sharpness_all, ied_duration_all, ied_gfp_amp_all, emhapp_save = \
            [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        for data_raw, data_info, bad_segment, candidate_ied_win, i in \
                zip(self.get_raw_data(), self.get_data_info(), self.get_bad_segment(), candidate_ied_wins,
                    range(len(candidate_ied_wins))):
            print(data_info['subject_info']['last_name'] + data_info['subject_info']['first_name'] +
                  ': IED_Detection(Threshold)', str(i + 1) + '/' + str(len(candidate_ied_wins)))
            find_peaks = ied_peak_feature.cal_peak_feature(raw_data=data_raw, data_info=data_info,
                                                           bad_segment=bad_segment, device=self.device_number,
                                                           n_jobs=self.n_jobs)
            # 计算每个fif文件的peaks，以及对应特征值
            peaks = find_peaks.get_peaks(smooth_windows=smooth_windows, smooth_iterations=smooth_iterations,
                                         z_win=z_win, z_region=z_region, z_mag_grad=z_mag_grad,
                                         exclude_duration=exclude_duration,
                                         peak_amplitude_threshold=peak_amplitude_threshold,
                                         peak_slope_threshold=peak_slope_threshold,
                                         half_peak_slope_threshold=half_peak_slope_threshold,
                                         peak_sharpness_threshold=peak_sharpness_threshold)
            # 根据特征值以及阈值，获取每个IED event的peak位置
            ieds = ied_detection_threshold.ied_detection_threshold(raw_data=data_raw, data_info=data_info,
                                                                   device=self.device_number, n_jobs=self.n_jobs)
            is_ieds_temp, ieds_peaks_temp, ieds_win_temp = \
                ieds.get_ieds_in_candidate_ieds_windows(mag_grad_ratio=mag_grad_ratio, chan_threshold=chan_threshold,
                                                        candidate_ieds_win=candidate_ied_win,
                                                        peak_amplitude=peaks['peak_amplitude'],
                                                        half_peak_slope=peaks['half_peak_slope'],
                                                        win_samples_of_picked_peak=peaks['win_samples_of_picked_peak'],
                                                        all_samples_of_picked_peak=peaks['all_samples_of_picked_peak'])
            # 去除重复的peaks
            if is_unique and ieds_peaks_temp is not None:
                temp = torch.tensor([torch.where(ieds_peaks_temp == x)[0][0] for x in ieds_peaks_temp.unique()])
                ieds_peaks_temp, ieds_win_temp = ieds_peaks_temp[temp], ieds_win_temp[temp]

            ieds_peak_closest_idx_temp, data_segment_temp = \
                ieds.cal_ied_segments(ieds_peaks=ieds_peaks_temp, ieds_win=ieds_win_temp,
                                      samples_of_picked_peak=torch.tensor(peaks['samples_of_picked_peak']),
                                      data_segment_window=data_segment_window)
            # 计算IED event的feature
            ied_amplitude_peaks_temp, ied_half_slope_peaks_temp, ied_sharpness_peaks_temp, ied_duration_peaks_temp, \
            ied_amplitude_temp, ied_half_slope_temp, ied_sharpness_temp, ied_duration_temp, ied_gfp_amp_temp = \
                ieds.cal_ieds_features(samples_of_picked_peak=torch.tensor(peaks['samples_of_picked_peak']),
                                       ieds_peak_closest_idx=ieds_peak_closest_idx_temp,
                                       peak_amplitude=peaks['peak_amplitude'], half_peak_slope=peaks['half_peak_slope'],
                                       peak_duration=peaks['peak_duration'], peak_sharpness=peaks['peak_sharpness'],
                                       data_gfp=ieds.get_data_gfp()[0], ieds_peaks=ieds_peaks_temp,
                                       channel_used_for_average=chan_threshold[0])

            # 如果没有检测到ied，将输出的None变为torch.tensor([])
            if ieds_peaks_temp is None:
                ieds_peaks_temp, ieds_win_temp, ied_amplitude_peaks_temp, ied_half_slope_peaks_temp, \
                ied_sharpness_peaks_temp, ied_duration_peaks_temp, ied_amplitude_temp, ied_half_slope_temp, \
                ied_sharpness_temp, ied_duration_temp, ied_gfp_amp_temp, ieds_peak_closest_idx_temp, \
                data_segment_temp, emhapp_save_temp = \
                    torch.tensor([]), torch.tensor([]), torch.tensor([]).to(self.device), \
                    torch.tensor([]).to(self.device), torch.tensor([]).to(self.device), \
                    torch.tensor([]).to(self.device), torch.tensor([]).to(self.device), \
                    torch.tensor([]).to(self.device), torch.tensor([]).to(self.device), \
                    torch.tensor([]).to(self.device), torch.tensor([]).to(self.device), \
                    torch.tensor([]).to(self.device), torch.tensor([]).to(self.device), []
            else:
                # 生成用于GUI的mat文件
                emhapp_save_temp = self.export_emhapp(info=data_info, meg_size=data_raw.shape[1],
                                                      ieds_peak=ieds_peaks_temp, ied_amplitude=ied_amplitude_peaks_temp,
                                                      ied_sharpness=ied_sharpness_peaks_temp,
                                                      ied_half_slope=ied_half_slope_peaks_temp)

            # 输出
            is_ieds_all.append(is_ieds_temp)
            ieds_peak_all.append(ieds_peaks_temp)
            ieds_win_all.append(ieds_win_temp)
            ieds_peak_closest_idx_all.append(ieds_peak_closest_idx_temp)
            data_segment_all.append(data_segment_temp)
            ied_amplitude_peaks_all.append(ied_amplitude_peaks_temp)
            ied_half_slope_peaks_all.append(ied_half_slope_peaks_temp)
            ied_sharpness_peaks_all.append(ied_sharpness_peaks_temp)
            ied_duration_peaks_all.append(ied_duration_peaks_temp)
            ied_amplitude_all.append(ied_amplitude_temp)
            ied_half_slope_all.append(ied_half_slope_temp)
            ied_sharpness_all.append(ied_sharpness_temp)
            ied_duration_all.append(ied_duration_temp)
            ied_gfp_amp_all.append(ied_gfp_amp_temp)
            emhapp_save.append(emhapp_save_temp)

        return is_ieds_all, ieds_peak_all, ieds_win_all, ieds_peak_closest_idx_all, data_segment_all, \
               ied_amplitude_peaks_all, ied_half_slope_peaks_all, ied_sharpness_peaks_all, ied_duration_peaks_all, \
               ied_amplitude_all, ied_half_slope_all, ied_sharpness_all, ied_duration_all, ied_gfp_amp_all, emhapp_save

    @staticmethod
    def export_emhapp(info, meg_size, ieds_peak, ied_amplitude, ied_sharpness, ied_half_slope):
        """
        Description:
            将结果添加到emhapp_save中

        Input:
            :param info: dict
                meg数据的头文件
            :param meg_size: number, long
                meg数据采样点个数
            :param ieds_peak: torch, long, shape(n_ieds)
                检测到的ied events的peak位置
            :param ied_amplitude: torch.tensor, double, shape(n_ieds*n_channel*2)
                每个通道，ieds主峰值两侧peaks的幅度
            :param ied_sharpness: torch.tensor, double, shape(n_ieds*n_channel*2)
                每个通道，ieds主峰值两侧peaks的锐度
            :param ied_half_slope: torch.tensor, double, shape(n_ieds*n_channel*2)
                每个通道，ieds主峰值两侧peaks的斜率

        Return:
            :return emhapp_save: dict
                用于保存emhapp mat文件的字典
        """

        # 获取SpikeChan: 每个IED事件中被检测为IED peak的通道索引
        temp = ied_amplitude.any(dim=-1).cpu()
        temp[~(temp != 0).any(dim=-1), 0] = 1
        temp = torch.where(temp)
        spike_chan = torch.split(temp[1], temp[0].unique(return_counts=True)[1].tolist())

        # 获取SpikeRegion: 每个IED时间中，根据每个脑区检测到的IED个数的脑区索引排序
        chan_region = [torch.tensor([info.ch_names.index(y) for y in mne.read_vectorview_selection(x, info=info)])
                       for x in ['Left-temporal', 'Right-temporal', 'Left-parietal', 'Right-parietal', 'Left-occipital',
                                 'Right-occipital', 'Left-frontal', 'Right-frontal']]
        chan_region = torch.cat([(torch.ones(x.shape[0]) * i).long()
                                 for i, x in enumerate(chan_region)])[torch.cat(chan_region).argsort()]
        spike_region = [torch.cat([chan_region[x], torch.arange(8)]).unique(return_counts=True) for x in spike_chan]
        spike_region = torch.stack([x[0][x[1].argsort(descending=True)] for x in spike_region])

        # 获取Amp: 每个IED事件中被检测为IED peak的通道，对应的peak幅值
        amp = [x[y].amax(dim=-1).cpu().numpy() for x, y in zip(ied_amplitude, spike_chan)]

        # 获取Sharp: 每个IED事件中被检测为IED peak的通道，对应的peak锐度
        sharp = [x[y].amax(dim=-1).cpu().numpy() for x, y in zip(ied_sharpness, spike_chan)]

        # 获取Slope: 每个IED事件中被检测为IED peak的通道，对应的peak值
        slope = [x[y].amax(dim=-1).cpu().numpy() for x, y in zip(ied_half_slope, spike_chan)]

        # 获取SpikeBsTime: 每个IED事件-1秒到1秒的采样点
        # 获取EventSpikeTime: 每个IED的peak在SpikeBsTime中的位置
        spike_bs_time = ieds_peak.reshape(-1, 1) + torch.arange(-1 * info['sfreq'], 1 * info['sfreq']).long()
        # 计算shift，确保SpikeBsTime在[0，meg_size]之间
        time_shift = torch.zeros(spike_bs_time.shape[0]).long()
        time_shift[spike_bs_time[:, 0] < 0] = -spike_bs_time[:, 0][spike_bs_time[:, 0] < 0]
        time_shift[spike_bs_time[:, -1] >= meg_size] = meg_size - spike_bs_time[:, -1][spike_bs_time[:, -1] >= meg_size]
        spike_bs_time = spike_bs_time + time_shift.reshape(-1, 1)
        spike_time = (1 * info['sfreq'] * torch.ones(spike_bs_time.shape[0])).long() - time_shift

        # 获取emhapp_save
        emhapp_save = \
            {'SampleFreq': info['sfreq'],
             'SpikeChan': np.array([x.cpu().numpy() + 1 for x in spike_chan] + [0], dtype=object)[:-1],
             'SpikeRegion': spike_region.cpu().numpy() + 1,
             'BsTime': spike_bs_time.cpu().numpy() + 1, 'EventSpikeTime': spike_time.cpu().numpy() + 1,
             'Amp': np.array(amp + [0], dtype=object)[:-1],
             'Slope': np.array(slope + [0], dtype=object)[:-1],
             'Sharp': np.array(sharp + [0], dtype=object)[:-1]}

        return emhapp_save


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(2)

    fif_list = ['/sanbo_dataset/4k_Project/Data_Full/MEG_IED/MEG1781/MEG1781_EP_1_tsss.fif',
                '/sanbo_dataset/4k_Project/Data_Full/MEG_IED/MEG1781/MEG1781_EP_2_tsss.fif']
    Raw_data, Info = [], []
    for fif in fif_list:
        raw = mne.io.read_raw_fif(fif, verbose='error', preload=True)
        raw.pick(mne.pick_types(raw.info, meg=True, ref_meg=False))
        Info.append(raw.info)
        Raw_data.append(raw.get_data())

    # 计算发作间期IEDs
    IED_Multi = ied_detection_threshold_multi(raw_data_s=Raw_data, data_info_s=Info, device=2, n_jobs=10)
    IEDs_peak_all, IEDs_win_all, IEDs_peak_closest_idx_all, Data_segment_all, \
    IEDs_amplitude_peaks_all, IEDs_half_slope_peaks_all, IEDs_sharpness_peaks_all, IEDs_duration_peaks_all, \
    IEDs_amplitude_all, IEDs_half_slope_all, IEDs_sharpness_all, IEDs_duration_all, IEDs_gfp_amp_all, _ = \
        IED_Multi.get_ieds_in_whole_recording(smooth_windows=0.02, smooth_iterations=2, z_win=100,
                                              z_region=True, z_mag_grad=False, exclude_duration=0.02,
                                              peak_amplitude_threshold=(2, None), half_peak_slope_threshold=(2, None),
                                              mag_grad_ratio=0.056, chan_threshold=(5, 200), ieds_window=0.3,
                                              data_segment_window=0.1)

    # 计算发作间期IEDs（时间窗）
    Candidate_ied_wins = [(np.random.randint(200, 10000, 100) + np.arange(-150, 150).reshape(300, -1)).T,
                          (np.random.randint(200, 300, 1) + np.arange(-150, 150).reshape(300, -1)).T]
    IED_Multi = ied_detection_threshold_multi(raw_data_s=Raw_data, data_info_s=Info, device=2, n_jobs=10)
    Is_IEDs, IEDs_peak_all, IEDs_win_all, IEDs_peak_closest_idx_all, Data_segment_all, \
    IEDs_amplitude_peaks_all, IEDs_half_slope_peaks_all, IEDs_sharpness_peaks_all, IEDs_duration_peaks_all, \
    IEDs_amplitude_all, IEDs_half_slope_all, IEDs_sharpness_all, IEDs_duration_all, IEDs_gfp_amp_all = \
        IED_Multi.get_ieds_in_candidate_ieds_windows(
            candidate_ied_wins=Candidate_ied_wins, smooth_windows=0.02, smooth_iterations=2,
            z_win=30, z_region=True, z_mag_grad=False, exclude_duration=0.02,
            peak_amplitude_threshold=(2, None), peak_slope_threshold=(None, None),
            half_peak_slope_threshold=(2, None), peak_sharpness_threshold=(None, None),
            mag_grad_ratio=0.056, chan_threshold=(5, 200), data_segment_window=0.1)
