# -*- coding:utf-8 -*-
# @Time    : 2022/2/4
# @Author  : cuiwei
# @File    : hfo_clustering.py
# @Software: PyCharm
# @Script to:
#   - 对于多个MEG文件，使用阈值法检测HFO，并使用GMM对所有的HFO片段进行聚类

import mne
import numpy as np
import torch
import scipy.io
from torch.nn.utils.rnn import pad_sequence
import os
import matplotlib.pyplot as plt
import copy
from hfo_detection import hfo_detection_threshold
from sklearn.mixture import GaussianMixture

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class hfo_cluster:
    def __init__(self, raw_data_s=None, raw_data_s_ied=None, raw_data_s_hfo=None, data_info_s=None, leadfield_s=None,
                 device=-1, n_jobs=10):
        """
        Description:
            初始化，使用GPU或者CPU。

        Input:
            :param raw_data_s: list, [ndarray, double, shape(channel, samples)], shape(n_files)
                MEG未滤波数据
            :param raw_data_s_ied: list, [ndarray, double, shape(channel, samples)], shape(n_files)
                MEG滤波后数据(IED)
            :param raw_data_s_hfo: list, [ndarray, double, shape(channel, samples)], shape(n_files)
                MEG滤波后数据(HFO)
            :param data_info_s: dict
                MEG数据的信息, MNE读取的raw.info
            :param leadfield_s: list, [ndarray, double, shape(n_dipoles, 3, n_channels)], shape(n_files)
                leadfield矩阵
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
        self.raw_data_s, self.raw_data_s_ied, self.raw_data_s_hfo, self.data_info_s, self.leadfield_s = \
            raw_data_s, raw_data_s_ied, raw_data_s_hfo, data_info_s, leadfield_s
        self.set_raw_data_s(raw_data_s=raw_data_s)
        self.set_raw_data_s_ied(raw_data_s_ied=raw_data_s_ied)
        self.set_raw_data_s_hfo(raw_data_s_hfo=raw_data_s_hfo)
        self.set_data_info_s(data_info_s=data_info_s)
        self.set_leadfield_s(leadfield_s=leadfield_s)

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

    def set_raw_data_s(self, raw_data_s=None):
        assert raw_data_s is not None
        self.raw_data_s = raw_data_s

    def get_raw_data_s(self):
        return self.raw_data_s

    def set_raw_data_s_ied(self, raw_data_s_ied=None):
        assert raw_data_s_ied is not None
        self.raw_data_s_ied = raw_data_s_ied

    def get_raw_data_s_ied(self):
        return self.raw_data_s_ied

    def set_raw_data_s_hfo(self, raw_data_s_hfo=None):
        assert raw_data_s_hfo is not None
        self.raw_data_s_hfo = raw_data_s_hfo

    def get_raw_data_s_hfo(self):
        return self.raw_data_s_hfo

    def set_data_info_s(self, data_info_s=None):
        assert data_info_s is not None
        self.data_info_s = data_info_s

    def get_data_info_s(self):
        return self.data_info_s

    def set_leadfield_s(self, leadfield_s=None):
        assert leadfield_s is not None
        self.leadfield_s = leadfield_s

    def get_leadfield_s(self):
        return self.leadfield_s

    @staticmethod
    def zscore(data, dim=-1):
        data_zscore = (data - data.mean(dim=dim, keepdim=True)) / data.std(dim=dim, keepdim=True)
        data_zscore = torch.nan_to_num(data_zscore, nan=0)
        return data_zscore

    def get_hfo_from_multi_fifs(self, ied_time_s, max_oris_hfo_s, max_oris_ied_s,
                                inv_covariance_hfo_s, inv_covariance_ied_s,
                                hfo_window_samples, baseline_window_samples_hfo,
                                hfo_amplitude_threshold=2, hfo_duration_threshold=0.02, hfo_oscillation_threshold=4,
                                hfo_entropy_threshold=1.25, hfo_power_ratio_threshold=1.25,
                                hfo_frequency_range=(80, 200), ied_frequency_range=(3, 80),
                                ied_window=0.1, hfo_window=0.25, tf_bw_threshold=0.5):
        """
        Description:
            对多个fif文件，使用阈值法间的hfo event，并计算特征值。使用阈值的方法，计算hfo segments和hfo events

        Input:
            :param ied_time_s: list, [np.array, double, shape(n_ieds*n_samples)], shape(n_fifs)
                ied片段在原始数据中的采样点位置
            :param max_oris_hfo_s: list, [np.array, double, shape(n_ieds*n_segments*n_vs_channels*3)], shape(n_fifs)
                hfo最优源空间信号方向
            :param max_oris_ied_s: list, [np.array, double, shape(n_ieds*n_segments*n_vs_channels*3)], shape(n_fifs)
                ied最优源空间信号方向
            :param inv_covariance_hfo_s: list, [np.array, double, shape(n_channels*n_channels)], shape(n_fifs)
                hfo频段的协方差矩阵逆矩阵
            :param inv_covariance_ied_s: list, [np.array, double, shape(n_channels*n_channels)], shape(n_fifs)
                ied频段的协方差矩阵逆矩阵
            :param hfo_window_samples: np.array, double, shape(n_segments*n_hfo_samples)
                hfo segments在ieds_segment_samples中的采样点位置
            :param baseline_window_samples_hfo: np.array, double, shape(n_segments*n_baseline_samples)
                hfo segments对应的baseline在ieds_segment_samples中的采样点位置
            :param hfo_amplitude_threshold: number, double
                hfo最小幅度阈值
            :param hfo_duration_threshold: number, double
                hfo小持续时间阈值，单位s
            :param hfo_oscillation_threshold: number, double
                hfo最小振荡个数阈值
            :param hfo_frequency_range: tuple, double
                hfo的[频率下限， 频率上限]
            :param hfo_entropy_threshold: number, double
                hfo最大时频熵的阈值
            :param hfo_power_ratio_threshold: number, double
                hfo最小最小能量比阈值
            :param ied_window: number, double
                ide的感兴趣时间窗长度，单位S
            :param hfo_window: number, double
                hfo的感兴趣时间窗长度，用于计算时频图，单位S
            :param hfo_frequency_range: list, double, shape(2)
                hfo的起始和结束频率
            :param ied_frequency_range: list, double, shape(2)
                ied的起始和结束频率
            :param tf_bw_threshold: number, double
                用于二值化时频图的阈值，最大值的倍数
        Return:
            :return emhapp_save: dict
                用于保存mat文件的字典
            :return hfo_segments:torch.tensor, double, shape(n_segments*9), in_device
                每个fo segments，[ied index, hfo segments index, vs channel,
                                 sample index in 100 ms hfo segments(begin),
                                 sample index in 100 ms hfo segments(end),
                                 sample index in hfo segments(begin),
                                 sample index in hfo segments(end),
                                 sample index in raw data(begin),
                                 sample index in raw data(end),]
            :return fif_index: torch.tensor, long, shape(n_events*9)
                每个event属于的fif文件的index
            :return hfo_hilbert_amplitude: list, [torch.tensor, double, shape(n_segments)], shape(n_events)
                每个hfo event的hfo segments，hfo平均包络幅度
            :return hfo_line_length: list, [torch.tensor, double, shape(n_segments)], shape(n_events)
                每个hfo event的hfo segments，hfo的line length
            :return hfo_mean_power: list, [torch.tensor, double, shape(n_segments)], shape(n_events)
                每个hfo event的hfo segments，hfo的平均能量
            :return hfo_std: list, [torch.tensor, double, shape(n_segments)], shape(n_events)
                每个hfo event的hfo segments，hfo的标准差
            :return hfo_center_frequency: list, [torch.tensor, double, shape(n_segments)], shape(n_events)
                每个hfo event的hfo segments，hfo的中心频率
            :return hfo_tf_area: list, [torch.tensor, double, shape(n_segments)], shape(n_events)
                每个hfo event的hfo segments，hfo的时频图"孤岛"面积
            :return hfo_tf_frequency_range: list, [torch.tensor, double, shape(n_segments)], shape(n_events)
                每个hfo event的hfo segments，hfo的时频图"孤岛"频率范围
            :return hfo_tf_entropy: list, [torch.tensor, double, shape(n_segments)], shape(n_events)
                每个hfo event的hfo segments，hfo的时频图熵
            :return ied_line_length: list, [torch.tensor, double, shape(n_segments)], shape(n_events)
                每个hfo event的hfo segments，ied的line length
            :return ied_mean_power: list, [torch.tensor, double, shape(n_segments)], shape(n_events)
                每个hfo event的hfo segments，ied的平均能量
            :return ied_std: list, [torch.tensor, double, shape(n_segments)], shape(n_events)
                每个hfo event的hfo segments，ied的标准差
            :return ied_peak2peak_amplitude: list, [torch.tensor, double, shape(n_segments)], shape(n_events)
                每个hfo event的hfo segments，ied的最大峰峰值
            :return ied_peak_amplitude: list, [torch.tensor, double, shape(n_segments)], shape(n_events)
                每个hfo event的hfo segments，ied的最大peak幅度
            :return ied_half_peak_slope: list, [torch.tensor, double, shape(n_segments)], shape(n_events)
                每个hfo event的hfo segments，ied的最大peak半高宽斜率
            :return ied_bad_power_middle_to_low: list, [torch.tensor, double, shape(n_segments)], shape(n_events)
                每个hfo event的hfo segments，ied的中间频带和低频带能量比
            :return ied_bad_power_high_to_low: list, [torch.tensor, double, shape(n_segments)], shape(n_events)
                每个hfo event的hfo segments，ied的高频带和低频带能量比
            :return ied_tf_entropy: list, [torch.tensor, double, shape(n_segments)], shape(n_events)
                每个hfo event的hfo segments，ied的时频图熵
        """

        # 初始化所有fif文件的hfo特征和hfo events时间
        hfo_hilbert_amplitude, hfo_line_length, hfo_mean_power, hfo_std, hfo_center_frequency, hfo_tf_area, \
        hfo_tf_frequency_range, hfo_tf_entropy, \
        ied_line_length, ied_mean_power, ied_std, ied_peak2peak_amplitude, ied_peak_amplitude, \
        ied_half_peak_slope, ied_bad_power_middle_to_low, ied_bad_power_high_to_low, ied_tf_entropy, \
        hfo_events, fif_index = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], torch.tensor([])
        emhapp_save = []

        # 对每个fif文件分依次进行处理
        for index, raw_data, raw_data_ied, raw_data_hfo, info, leadfield, \
            ied_time, max_oris_hfo, max_oris_ied, inv_covariance_hfo, inv_covariance_ied in \
                zip(range(len(ied_time_s)),
                    self.get_raw_data_s(), self.get_raw_data_s_ied(), self.get_raw_data_s_hfo(),
                    self.get_data_info_s(), self.get_leadfield_s(),
                    ied_time_s, max_oris_hfo_s, max_oris_ied_s, inv_covariance_hfo_s, inv_covariance_ied_s):
            print(info['subject_info']['last_name'] + info['subject_info']['first_name'] +
                  ': HFO_Detection', str(index + 1) + '/' + str(len(ied_time_s)))
            if len(ied_time) == 0:
                continue
            # 构建每个fif文件的hfo detection
            hfo = hfo_detection_threshold.hfo_detection_threshold(raw_data=raw_data, raw_data_ied=raw_data_ied,
                                                                  raw_data_hfo=raw_data_hfo, data_info=info,
                                                                  leadfield=leadfield, device=self.device_number)
            # 使用阈值法检测hfo、并计算hfo特征
            emhapp_save_temp, hfo_events_temp, \
            hfo_hilbert_amplitude_temp, hfo_line_length_temp, hfo_mean_power_temp, hfo_std_temp, \
            hfo_center_frequency_temp, hfo_tf_area_temp, hfo_tf_frequency_range_temp, hfo_tf_entropy_temp, \
            ied_line_length_temp, ied_mean_power_temp, ied_std_temp, ied_peak2peak_amplitude_temp, \
            ied_peak_amplitude_temp, ied_half_peak_slope_temp, ied_bad_power_middle_to_low_temp, \
            ied_bad_power_high_to_low_temp, ied_tf_entropy_temp = hfo.cal_hfo_and_features(
                ieds_segment_samples=ied_time, max_oris_hfo=max_oris_hfo, max_oris_ied=max_oris_ied,
                inv_covariance_hfo=inv_covariance_hfo, inv_covariance_ied=inv_covariance_ied,
                hfo_window_samples=hfo_window_samples, baseline_window_samples_hfo=baseline_window_samples_hfo,
                hfo_amplitude_threshold=hfo_amplitude_threshold, hfo_duration_threshold=hfo_duration_threshold,
                hfo_oscillation_threshold=hfo_oscillation_threshold, hfo_entropy_threshold=hfo_entropy_threshold,
                hfo_power_ratio_threshold=hfo_power_ratio_threshold,
                hfo_frequency_range=hfo_frequency_range, ied_frequency_range=ied_frequency_range,
                ied_window=ied_window, hfo_window=hfo_window, tf_bw_threshold=tf_bw_threshold)

            if len(hfo_events_temp) > 0:
                emhapp_save.append(emhapp_save_temp)
                fif_index = torch.cat([fif_index, torch.ones(len(hfo_events_temp)) * index]).long()
                hfo_events = hfo_events + list(hfo_events_temp)
                hfo_hilbert_amplitude = hfo_hilbert_amplitude + list(hfo_hilbert_amplitude_temp)
                hfo_line_length = hfo_line_length + list(hfo_line_length_temp)
                hfo_mean_power, hfo_std = hfo_mean_power + list(hfo_mean_power_temp), hfo_std + list(hfo_std_temp)
                hfo_center_frequency = hfo_center_frequency + list(hfo_center_frequency_temp)
                hfo_tf_area = hfo_tf_area + list(hfo_tf_area_temp)
                hfo_tf_frequency_range = hfo_tf_frequency_range + list(hfo_tf_frequency_range_temp)
                hfo_tf_entropy = hfo_tf_entropy + list(hfo_tf_entropy_temp)
                ied_line_length = ied_line_length + list(ied_line_length_temp)
                ied_mean_power, ied_std = ied_mean_power + list(ied_mean_power_temp), ied_std + list(ied_std_temp)
                ied_peak2peak_amplitude = ied_peak2peak_amplitude + list(ied_peak2peak_amplitude_temp)
                ied_peak_amplitude = ied_peak_amplitude + list(ied_peak_amplitude_temp)
                ied_half_peak_slope = ied_half_peak_slope + list(ied_half_peak_slope_temp)
                ied_bad_power_middle_to_low = ied_bad_power_middle_to_low + list(ied_bad_power_middle_to_low_temp)
                ied_bad_power_high_to_low = ied_bad_power_high_to_low + list(ied_bad_power_high_to_low_temp)
                ied_tf_entropy = ied_tf_entropy + list(ied_tf_entropy_temp)

        return emhapp_save, fif_index, hfo_events, \
               hfo_hilbert_amplitude, hfo_line_length, hfo_mean_power, hfo_std, hfo_center_frequency, hfo_tf_area, \
               hfo_tf_frequency_range, hfo_tf_entropy, \
               ied_line_length, ied_mean_power, ied_std, ied_peak2peak_amplitude, ied_peak_amplitude, \
               ied_half_peak_slope, ied_bad_power_middle_to_low, ied_bad_power_high_to_low, ied_tf_entropy

    def cal_hfo_clustering(self, ied_time_s, max_oris_hfo_s, max_oris_ied_s,
                           inv_covariance_hfo_s, inv_covariance_ied_s,
                           hfo_window_samples, baseline_window_samples_hfo,
                           hfo_amplitude_threshold=2, hfo_duration_threshold=0.02, hfo_oscillation_threshold=4,
                           hfo_entropy_threshold=1.25, hfo_power_ratio_threshold=1.25,
                           hfo_frequency_range=(80, 200), ied_frequency_range=(3, 80),
                           ied_window=0.1, hfo_window=0.25, tf_bw_threshold=0.5,
                           n_segments_per_event=5, n_component=1, is_whitening=False,
                           gmm_n_init=10, gmm_covariance_type='diag'):

        # 使用阈值法，检测所有fif数据中的hfo
        emhapp_save, fif_index, hfo_events, \
        hfo_hilbert_amplitude, hfo_line_length, hfo_mean_power, hfo_std, hfo_center_frequency, hfo_tf_area, \
        hfo_tf_frequency_range, hfo_tf_entropy, \
        ied_line_length, ied_mean_power, ied_std, ied_peak2peak_amplitude, ied_peak_amplitude, \
        ied_half_peak_slope, ied_bad_power_middle_to_low, ied_bad_power_high_to_low, ied_tf_entropy = \
            self.get_hfo_from_multi_fifs(
                ied_time_s=ied_time_s, max_oris_hfo_s=max_oris_hfo_s, max_oris_ied_s=max_oris_ied_s,
                inv_covariance_hfo_s=inv_covariance_hfo_s, inv_covariance_ied_s=inv_covariance_ied_s,
                hfo_window_samples=hfo_window_samples, baseline_window_samples_hfo=baseline_window_samples_hfo,
                hfo_amplitude_threshold=hfo_amplitude_threshold, hfo_duration_threshold=hfo_duration_threshold,
                hfo_oscillation_threshold=hfo_oscillation_threshold, hfo_entropy_threshold=hfo_entropy_threshold,
                hfo_power_ratio_threshold=hfo_power_ratio_threshold,
                hfo_frequency_range=hfo_frequency_range, ied_frequency_range=ied_frequency_range,
                ied_window=ied_window, hfo_window=hfo_window, tf_bw_threshold=tf_bw_threshold)

        if len(hfo_events) == 0:
            return [], [], [], [], [], []
        # 提取每个hfo对应的特征
        features = [torch.stack([a, b, c, d, e], dim=-1) for a, b, c, d, e in
                    zip(hfo_hilbert_amplitude, hfo_tf_entropy, ied_peak_amplitude, ied_half_peak_slope,
                        ied_bad_power_middle_to_low)]

        # 根据hfo特征进行聚类
        cls, cluster_fit_value, processing_segments_index = self.cal_clustering(
            hfo_amplitude=hfo_hilbert_amplitude, ied_amplitude=ied_peak_amplitude, features=features,
            n_segments_per_event=n_segments_per_event, n_component=n_component, is_whitening=is_whitening,
            gmm_n_init=gmm_n_init, gmm_covariance_type=gmm_covariance_type)
        # 根据拟合度，使用手肘法计算最优类别数
        best_cls = self.cal_best_cls_number(cluster_fit_value=cluster_fit_value, value_threshold=0.95)

        # 将cls和processing_segments_index分配到每个对应的fif
        cluster = [[z.cpu().numpy() for y, z in zip(fif_index, cls) if y == x] for x in fif_index.unique_consecutive()]
        picked_channel_index = [[z.cpu().numpy() for y, z in zip(fif_index, processing_segments_index) if y == x]
                                for x in fif_index.unique_consecutive()]
        # 输出的list每一项对应的输入index
        fif_index = fif_index.unique_consecutive()

        return emhapp_save, best_cls, cluster, picked_channel_index, cluster_fit_value, fif_index

    def cal_best_cls_number(self, cluster_fit_value, value_threshold=0.95):
        """
        Description:
            用手肘法，计算最优类别个数。认为拟合度到达value_threshold为平台，为最优。

        Input:
            :param cluster_fit_value: torch.tensor, double, shape(n_clusters_numbers)
                每个cluster个数，对应的拟合度。
            :param value_threshold: number, double
                判断拟合度达到平台的阈值
        Return:
            :return best_cls: list, number, long
                距离value_threshold最近的最优cls个数
        """

        # modify cluster_fit_value，将cluster_fit_value变为单调
        cluster_fit_value_modified = \
            torch.stack([torch.cat([cluster_fit_value[:x + 1],
                                    torch.ones(cluster_fit_value.shape[0] - x) * cluster_fit_value[x]])[:-1]
                         for x in range(cluster_fit_value.shape[0])]).amin(dim=-1)
        cluster_fit_value_modified = -cluster_fit_value_modified
        # 将cluster_fit_value_modified归一化到0到1之间
        cluster_fit_value_modified = cluster_fit_value_modified if cluster_fit_value_modified.shape[0] == 1 else \
            -cluster_fit_value_modified.diff().cumsum(dim=0) / (cluster_fit_value_modified[0] -
                                                                cluster_fit_value_modified[-1])
        # 计算距离value_threshold最近的cls number
        best_cls = cluster_fit_value_modified - value_threshold
        best_cls[best_cls < 0] = 1
        best_cls = best_cls.argmin()
        self.del_var()

        return best_cls

    def cal_clustering(self, hfo_amplitude, ied_amplitude, features, n_segments_per_event=5,
                       n_component=1, is_whitening=False, gmm_n_init=10, gmm_covariance_type='diag'):
        """
        Description:
            使用高斯混合模型，根据hfo特征，对hfo片段进行聚类

        Input:
            :param hfo_amplitude: list, [torch.tensor, double, shape(n_segments)], shape(n_events), in_device
                hfo片段对应的hfo幅度
            :param ied_amplitude: list, [torch.tensor, double, shape(n_segments)], shape(n_events), in_device
                hfo片段对应的ied幅度
            :param features: list, [torch.tensor, double, shape(n_segments*n_features)], shape(n_events), in_device
                hfo片段对应用于聚类的特征值
            :param n_segments_per_event: number, long
                每个hfo event，需要提取的最大ied/hfo幅度的hfo 片段个数。
            :param n_component: number, double
                去除feature之间相关性的程度，越小代表去除的feature个数越多，为1的时候，不去除
            :param is_whitening: number, bool
                是否将不同feature的分布宽度，缩放到相同幅度
            :param gmm_n_init: number, long
                使用kmeans确定gmm聚类中心时，kmeans计算的次数
            :param gmm_covariance_type: number, string
                gmm拟合时，使用的协方的类型
        Return:
            :return cls: torch.tensor, long, shape(n_segments*max_cls_numbers)
                不同类别个数的，聚类结果
            :return cluster_fit_value: torch.tensor, double, shape(max_cls_numbers)
                不同类别个数的，聚类拟合程度
            :return processing_segments_index: list, [torch.tensor, long, shape(n_segments_picked)], shape(n_events)
                每个hfo event，选取的hfo片段的index
        """

        # step1: 对于每一个hfo evnet，获取ied和hfo幅度最大的前n_segments_per_event个segment
        processing_segments_index = [(self.zscore(x) + self.zscore(y)).topk(n_segments_per_event)[1]
                                     if x.shape[0] > n_segments_per_event else torch.arange(x.shape[0])
                                     for x, y in zip(hfo_amplitude, ied_amplitude)]
        features_picked = torch.cat([x[y] for x, y in zip(features, processing_segments_index)], dim=0)

        # step2: 对每一个feature，在sample维度进行去均值，然后降低feature之间的想关性
        features_picked = features_picked if features_picked.shape[0] == 1 else \
            (features_picked - features_picked.mean(dim=0, keepdim=True)) / features_picked.std(dim=0, keepdim=True)
        features_picked = features_picked if features_picked.shape[0] <= features_picked.shape[1] \
            else self.reduce_feature_dimension(features_picked, n_component=n_component, is_whitening=is_whitening)

        # step3: 使用gmm进行聚类，最大类别为16类
        if features_picked.shape[0] > 1:
            gmm, bic, aic = [], [], []
            for n_components in range(1, min(int(features_picked.shape[0] / 2) + 1, 16)):
                # fit gmm模型
                gmm_temp = GaussianMixture(n_components=n_components, covariance_type=gmm_covariance_type,
                                           random_state=0, n_init=gmm_n_init)
                gmm_temp.fit(features_picked)
                gmm.append(gmm_temp)
                # 计算aic和bic
                bic.append(gmm_temp.bic(features_picked))
                aic.append(gmm_temp.aic(features_picked))
            cls = [torch.stack([torch.tensor(gmm_temp.predict(x)) for gmm_temp in gmm], dim=-1)
                   for x in features_picked.split([x.shape[0] for x in processing_segments_index])]
        else:
            cls, aic, bic = torch.tensor([[0]]), [-1.], [-1.]
        # 计算cluster数据评估的指标
        cluster_fit_value = self.zscore(torch.tensor(bic), dim=0) + self.zscore(torch.tensor(aic), dim=0)

        return cls, cluster_fit_value, processing_segments_index

    def reduce_feature_dimension(self, feature, n_component=1, is_whitening=True):
        """
        Description:
            对feature进行svd分解，然后保留满足要求的成分个数，从而减少相关性比较大的feature。

        Input:
            :param feature: torch.tensor, double, shape(n_samples*n_features), in_device
                特征
            :param n_component: tuple, double
                保留的component对应的s需要大于n_component*max(s)，如果n_component=1，则不行处理
            :param is_whitening: number, bool
                是否进行白化处理，将各个feature的分布归一化到相同水兵
        Return:
            :return data_rd: tensor, double, shape(n_samples*n_features), in_device
                减少维度后的特征
        """
        if n_component == 1:
            data_rd = feature.clone().cpu()
        else:
            feature_zero_mean = feature.double().to(self.device) if feature.shape[0] == 1 \
                else (feature - feature.mean(dim=0, keepdim=True)).double().to(self.device)
            # SVD分解
            u, s, v = torch.tensor(np.cov(feature_zero_mean.t().cpu())).to(self.device).svd()
            u, s = u[:, s > s.max() * 2.2204e-16], s[s > s.max() * 2.2204e-16]
            # 获取满足要求的component
            n_component = torch.where((s / s.sum()).cumsum(dim=0) > n_component - 10e-8)[0].max() + 1
            u, s = u[:, :n_component], s[:n_component]
            # 白化
            if is_whitening:
                u = u / (s + 10e-8).sqrt().unsqueeze(0)
            data_rd = torch.mm(feature_zero_mean, u).cpu()
        return data_rd

    @staticmethod
    def export_emhapp(emhapp_save, best_cls, cluster, picked_channel_index, cluster_fit_value):
        """
        Description:
            将聚类结果添加到emhapp_save中

        Input:
            :param emhapp_save: dict
                用于保存mat文件的字典
            :param best_cls: number, long
                使用手肘法计算的最优类别数目
            :param cluster: list, [np.array, long, shape(n_segments*max_cls)], shape(n_fifs)
                选择的segments在不同的类别个数中，所属的类别
            :param picked_channel_index: list, [np.array, long, shape(n_segments)], shape(n_fifs)
                每个events选取的segment的index
            :param cluster_fit_value: np.array, double, shape(max_cls)
                不同类别个数的拟合度
        Return:
            :return emhapp_save_new: dict
                用于保存mat文件的字典
        """

        emhapp_save_new = []
        for x, y, z in zip(emhapp_save, cluster, picked_channel_index):
            x['Bic'] = cluster_fit_value.cpu().numpy()
            x['Cls'] = np.array([x + 1 for x in y] + [0], dtype=object)[:-1]
            x['ClsIdx'] = np.array([x + 1 for x in z] + [0], dtype=object)[:-1]
            x['Best_Cls'] = np.array(best_cls + 1)
            emhapp_save_new.append(x)

        return emhapp_save_new

    def plot_cluster_results(self, emhapp_save, ied_time_s, windows=0.3, select_cls=None,
                             plot_column=3, figsize=(30, 12)):
        import matplotlib

        # step1: 画出拟合度曲线
        cluster_fit_value = emhapp_save[0]['Bic']
        best_cls = emhapp_save[0]['Best_Cls']
        plt.figure(num=9334)
        plt.plot(cluster_fit_value, '-o', color=[0, 1, 0])
        plt.plot(best_cls, cluster_fit_value[best_cls], '-*', color=[1, 0, 0])
        plt.show()

        # step2: 画出每类数据
        cls = torch.cat([torch.cat([torch.tensor(y - 1) for y in x['Cls']]) for x in emhapp_save])
        color = list(matplotlib.colors.BASE_COLORS.items())[:-1] + list(matplotlib.colors.BASE_COLORS.items())[:-1] + \
                list(matplotlib.colors.BASE_COLORS.items())[:-1]
        # 获取数据:
        vs_ied_all, vs_hfo_all = [], []
        windows_half = int(windows * self.data_info_s[0]['sfreq'] / 2)
        for x, y, ied, hfo in zip(emhapp_save, ied_time_s, self.get_raw_data_s_ied(), self.get_raw_data_s_hfo()):
            hfo_weight = torch.cat([torch.tensor(m[n - 1]) for m, n in zip(x['EntHFO_Weight'], x['ClsIdx'])])
            hfo_center = torch.tensor([torch.tensor(m[:, 2:] - 1).float().mean().long() for m in x['EntHFO']])
            ied_weight = torch.cat([torch.tensor(m[n - 1]) for m, n in zip(x['EntHFO_Weight_S'], x['ClsIdx'])])
            ieds_index = torch.tensor([torch.tensor(m[0, 0] - 1) for m in x['EntHFO']])
            vs_ied, vs_hfo = [], []
            for index, ied_index, center in \
                    zip(torch.arange(ied_weight.shape[0]).split([event.shape[0] for event in x['ClsIdx']]),
                        ieds_index, hfo_center):
                hfo_temp = torch.tensor(hfo[:, y[ied_index]]).to(self.device)
                vs_hfo_temp = torch.mm(hfo_weight[index].to(self.device), hfo_temp)
                vs_hfo_temp = (vs_hfo_temp - vs_hfo_temp.mean(dim=-1, keepdim=True)) / vs_hfo_temp.std(dim=-1,
                                                                                                       keepdim=True)
                ied_temp = torch.tensor(ied[:, y[ied_index]]).to(self.device)
                vs_ied_temp = torch.mm(ied_weight[index].to(self.device), ied_temp)
                vs_ied_temp = (vs_ied_temp - vs_ied_temp.mean(dim=-1, keepdim=True)) / vs_ied_temp.std(dim=-1,
                                                                                                       keepdim=True)
                vs_ied.append(vs_ied_temp[:, int(ied_temp.shape[-1] / 2) + torch.arange(-windows_half, windows_half)])
                vs_hfo.append(vs_hfo_temp[:, center + torch.arange(-windows_half, windows_half)])
            vs_ied_all = vs_ied_all + vs_ied
            vs_hfo_all = vs_hfo_all + vs_hfo
        vs_ied_all = torch.cat(vs_ied_all).cpu()
        vs_hfo_all = torch.cat(vs_hfo_all).cpu()
        # plot 数据
        if select_cls is None:
            select_cls = best_cls
        fig = plt.figure(num=9333, figsize=figsize, clear=True)
        plot_row = np.ceil((select_cls + 1) / plot_column).astype('int64')
        plot_column_row = torch.stack((torch.arange(0, plot_row).repeat(plot_column).reshape(-1),
                                       torch.arange(0, plot_column).repeat_interleave(plot_row).reshape(-1))).t()
        for index in range(select_cls + 1):
            ax = plt.Axes(fig, [1 / plot_column * plot_column_row[index][1], 1 / plot_row * plot_column_row[index][0],
                               0.95 / plot_column, 0.95 / plot_row])
            ax.set_axis_off()
            fig.add_axes(ax)
            hfo_temp = vs_hfo_all[np.where(cls[:, select_cls] == index)[0]]
            ied_temp = vs_ied_all[np.where(cls[:, select_cls] == index)[0]]
            temp = torch.cat([ied_temp + 10, hfo_temp])
            plt.plot(temp.t().cpu(), linewidth=0.5, color=color[index][0])
        plt.show()


def load_param_matlab(mat_file):
    mat_file = mat_file.split(',')
    Param = [scipy.io.loadmat(x)['Param'][0] for x in mat_file]
    return [x['bad_seg'][0][0] > 0 for x in Param], [x['fifPath'][0][0] for x in Param], \
           [x['SpikeTime'][0] for x in Param], \
           Param[0]['Resample'][0][0][0], \
           Param[0]['FreqRange'][0][0], Param[0]['SignalTime'][0][0][0], Param[0]['HFOTime'][0][0][0], \
           Param[0]['AmpThre'][0][0][0], Param[0]['MinPeaks'][0][0][0], Param[0]['MinWin'][0][0][0], \
           Param[0]['HighThroughRatio'][0][0][0], Param[0]['MaxMinEntropy'][0][0][0], \
           Param[0]['BwThres'][0][0][0], Param[0]['ImageEntropy'][0][0][0], \
           Param[0]['FreqWidth'][0][0][0], Param[0]['ImageArea'][0][0][0], \
           Param[0]['OriMethod'][0][0][0], \
           [x.mean(axis=0) for x in Param[0]['SpikeTemplate'][0][0].tolist()], \
           Param[0]['HFO_Time'][0].astype('int64'), Param[0]['BS_Time'][0].astype('int64'), \
           [x['ReconParam'][0][0]['CovInv'][0] for x in Param], \
           [x['ReconParam'][0][0]['CovInv_S'][0] for x in Param], \
           [x['ReconParam'][0][0]['noiseCovInv'][0] for x in Param], \
           [x['ReconParam'][0][0]['noiseCovInv_S'][0] for x in Param], \
           [x['ReconParam'][0][0]['lf'][0] for x in Param], \
           [x['ReconParam'][0][0]['lfReduce'][0] for x in Param], \
           [x['ReconParam'][0][0]['FixOir'][0] for x in Param], \
           [x['ReconParam'][0][0]['FixOir_S'][0] for x in Param], \
           [x['ReconParam'][0][0]['MaxOir'][0] for x in Param], \
           [x['ReconParam'][0][0]['MaxOir_S'][0] for x in Param], \
           [[[z for z in y[0]] for y in x['PeakChanAmpSlop'][0][0]] for x in Param], \
           Param[0]['Device'][0][0][0]


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(1)

    bad_seg, fifPath, SpikeTime, Resample, FreqRange, SignalTime, HFOTime, AmpThre, MinPeaks, MinWin, HighThroughRatio, \
    MaxMinEntropy, BwThres, ImageEntropy, FreqWidth, ImageArea, OriMethod, SpikeTemplate, HFO_Time, BS_Time, \
    CovInv, CovInv_S, noiseCovInv, noiseCovInv_S, lf, lfReduce, FixOir, FixOir_S, MaxOir, MaxOir_S, \
    PeakChanAmpSlop, Device = load_param_matlab(matFile)

    RAW_data_s, RAW_data_ied_s, RAW_data_hfo_s, Info_s = [], [], [], []
    for fif in fifPath:
        RAW = mne.io.read_raw_fif(fif, verbose='error', preload=True)
        Info = RAW.info
        RAW.resample(1000, n_jobs='cuda')
        RAW_data = RAW.get_data(picks='meg')
        RAW_data_ied = mne.filter.filter_data(RAW_data, RAW.info['sfreq'], 3, 80,
                                              fir_design='firwin', pad="reflect_limited", verbose=False, n_jobs='cuda')
        RAW_data_ied = mne.filter.notch_filter(RAW_data_ied, RAW.info['sfreq'], 50., verbose=False, n_jobs='cuda')
        RAW_data_hfo = mne.filter.filter_data(RAW_data, RAW.info['sfreq'], 80, 200,
                                              fir_design='firwin', pad="reflect_limited", verbose=False, n_jobs='cuda')
        RAW_data_s.append(RAW_data)
        RAW_data_ied_s.append(RAW_data_ied)
        RAW_data_hfo_s.append(RAW_data_hfo)
        Info_s.append(Info)

    HFO = hfo_cluster(raw_data_s=RAW_data_s, raw_data_s_ied=RAW_data_ied_s, raw_data_s_hfo=RAW_data_hfo_s,
                      data_info_s=Info_s, leadfield_s=lf, device=5, n_jobs=10)

    Emhapp_save, Best_cls, Cluster, Picked_channel_index, Cluster_fit_value, Fif_index = \
        HFO.cal_hfo_clustering(ied_time_s=SpikeTime, max_oris_hfo_s=MaxOir, max_oris_ied_s=MaxOir_S,
                               inv_covariance_hfo_s=CovInv, inv_covariance_ied_s=CovInv_S,
                               hfo_window_samples=HFO_Time[0], baseline_window_samples_hfo=BS_Time[0],
                               hfo_amplitude_threshold=2, hfo_duration_threshold=0.02, hfo_oscillation_threshold=4,
                               hfo_entropy_threshold=1.25, hfo_power_ratio_threshold=1.25,
                               hfo_frequency_range=(80, 200), ied_frequency_range=(3, 80),
                               ied_window=0.1, hfo_window=0.25, tf_bw_threshold=0.5,
                               n_segments_per_event=5, n_component=1, is_whitening=False,
                               gmm_n_init=10, gmm_covariance_type='diag')
    Emhapp_save = HFO.export_emhapp(emhapp_save=Emhapp_save, best_cls=Best_cls, cluster=Cluster,
                                    picked_channel_index=Picked_channel_index, cluster_fit_value=Cluster_fit_value)
