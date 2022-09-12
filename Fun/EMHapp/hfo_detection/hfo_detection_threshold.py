# -*- coding:utf-8 -*-
# @Time    : 2022/2/4
# @Author  : cuiwei
# @File    : hfo_clustering.py
# @Software: PyCharm
# @Script to:
#   - 对于单个MEG文件，使用阈值法检测HFO

import mne
import numpy as np
import torch
import torch.fft
import scipy.io
from torch.nn.utils.rnn import pad_sequence
import os
import matplotlib.pyplot as plt
import copy
from ied_detection import ied_peak_feature
from skimage import measure

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class hfo_detection_threshold:
    def __init__(self, raw_data=None, raw_data_ied=None, raw_data_hfo=None, data_info=None, leadfield=None,
                 device=-1, n_jobs=10):
        """
        Description:
            初始化，使用GPU或者CPU。

        Input:
            :param raw_data: ndarray, double, shape(channel, samples)
                MEG未滤波数据
            :param raw_data_ied: ndarray, double, shape(channel, samples)
                MEG滤波后数据(IED)
            :param raw_data_hfo: ndarray, double, shape(channel, samples)
                MEG滤波后数据(HFO)
            :param data_info: dict
                MEG数据的信息, MNE读取的raw.info
            :param leadfield: ndarray, double, shape(n_dipoles*3*n_channels)
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
        self.raw_data, self.raw_data_ied, self.raw_data_hfo, self.data_info, self.leadfield = \
            raw_data, raw_data_ied, raw_data_hfo, data_info, leadfield
        self.set_raw_data(raw_data=raw_data)
        self.set_raw_data_ied(raw_data_ied=raw_data_ied)
        self.set_raw_data_hfo(raw_data_hfo=raw_data_hfo)
        self.set_data_info(data_info=data_info)
        self.set_leadfield(leadfield=leadfield)

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

    def hilbert(self, sig, h=None):
        """
        Description:
            计算hilbert变换，计算最后一维

        Input:
            :param sig: torch.tensor, double, shape(...*n_samples)
                输入数据
            :param h: torch.tensor, double, shape(n_samples)
                hilbert_h输出
        Return:
            :return hilbert_sig: torch.tensor, double, shape(...*n_sample)
                hilbert变换输出值
        """
        if (h is None) or (h.shape[0] != sig.shape[-1]):
            h = self.hilbert_h(sig.shape[-1]).to(self.device)
        sig = torch.fft.fft(sig, axis=-1)
        hilbert_sig = torch.abs(torch.fft.ifft(sig * h, axis=-1))
        return hilbert_sig

    def hilbert_h(self, n):
        h = torch.zeros(n)
        if n % 2 == 0:
            h[0] = h[n // 2] = 1
            h[1:n // 2] = 2
        else:
            h[0] = 1
            h[1:(n + 1) // 2] = 2
        return h.to(self.device)

    def set_raw_data(self, raw_data=None):
        assert raw_data is not None
        self.raw_data = torch.tensor(raw_data)

    def get_raw_data(self):
        return self.raw_data

    def set_raw_data_ied(self, raw_data_ied=None):
        assert raw_data_ied is not None
        self.raw_data_ied = torch.tensor(raw_data_ied)

    def get_raw_data_ied(self):
        return self.raw_data_ied

    def set_raw_data_hfo(self, raw_data_hfo=None):
        assert raw_data_hfo is not None
        self.raw_data_hfo = torch.tensor(raw_data_hfo)

    def get_raw_data_hfo(self):
        return self.raw_data_hfo

    def set_data_info(self, data_info=None):
        assert data_info is not None
        self.data_info = data_info

    def get_data_info(self):
        return self.data_info

    def set_leadfield(self, leadfield=None):
        assert leadfield is not None
        self.leadfield = torch.tensor(leadfield)

    def get_leadfield(self):
        return self.leadfield

    def filter(self, raw_data, ied_band=(3., 80.), hfo_band=(80., 200.)):
        """
        Description:
            读数据进行ied/hfo频段滤波，如果输入频段为None，则返回None
        Input:
            :param raw_data: np.array, double, shape(n_channels*n_samples)
                输入原始数据
            :param ied_band: list, double, shape(2)
                ied滤波频段
            :param hfo_band: list, double, shape(2)
                hfo滤波频段
        Return:
            :return raw_data_ied: torch.tensor, double, shape(n_channels*n_samples)
                ied滤波后数据，如果ied_band为None，返回None
            :return raw_data_hfo: torch.tensor, double, shape(n_channels*n_samples)
                hfo滤波后数据，如果hfo_band为None，返回None
        """
        # IED频段滤波
        if hfo_band is None:
            raw_data_ied = mne.filter.filter_data(raw_data, self.data_info['sfreq'], ied_band[0], ied_band[1],
                                                  fir_design='firwin', pad="reflect_limited", verbose=False,
                                                  n_jobs='cuda' if self.isCuda else self.Jobs)
            raw_data_ied = mne.filter.notch_filter(raw_data_ied, self.data_info['sfreq'], 50.,
                                                   verbose=False, n_jobs='cuda' if self.isCuda else self.Jobs)
            raw_data_ied = torch.tensor(raw_data_ied)
        else:
            raw_data_ied = None
        # HFO频段滤波
        if hfo_band is None:
            raw_data_hfo = mne.filter.filter_data(raw_data, self.data_info['sfreq'], hfo_band[0], hfo_band[1],
                                                  fir_design='firwin', pad="reflect_limited", verbose=False,
                                                  n_jobs='cuda' if self.isCuda else self.Jobs)
            raw_data_hfo = torch.tensor(raw_data_hfo)
        else:
            raw_data_hfo = None
        return raw_data_ied, raw_data_hfo

    def zscore_data(self, data, baseline_samples):
        """
        Description:
            读数据进行归一化，使用指定sample的值求取mean和std

        Input:
            :param data: torch.tensor, double, shape(n_batch*n_samples)
                需要归一化的数据
            :param baseline_samples: torch.tensor, long, shape(n_batch*n_samples) or shape(1*n_samples)
                baseline的sample点
        Return:
            :return data_zscore: torch.tensor, double, shape(n_batch*n_samples)
                归一化后的数据

        :param data:
        :param baseline_samples:
        :return:
        """
        baseline = data.to(self.device).take_along_dim(baseline_samples.to(self.device), dim=-1)
        data_zscore = (data - baseline.mean(dim=-1, keepdim=True)) / baseline.std(dim=-1, keepdim=True)
        self.del_var()

        return data_zscore

    def peak_detect(self, data):
        """
        Description:
            计算data中的peak位置
            并存储为shape(n_batch*max_peaks_num_along_batch)形式，用100补足

        Input:
            :param data: torch.tensor, double, shape(n_batch*n_samples)
                输入数据
        Return:
            :return peaks: torch.tensor, long, shape(n_batch*max_peaks_num_along_batch), in_device
                peaks位置
            :return peaks_number: torch.tensor, long, shape(n_batch), in_device
                peaks的个数
        """
        with torch.no_grad():
            # 获取每个batch的peak位置
            data_diff = ((data[:, 1:] - data[:, :-1]) >= 0).float()
            data_diff = data_diff[:, 1:] - data_diff[:, :-1]
            peaks = torch.where(data_diff.abs() == 1)
            # 将peaks转换为shape(n_batch*max_peaks_num_along_batch)
            peaks_number = torch.zeros(data.shape[0]).long().to(self.device)
            temp = peaks[0].unique_consecutive(return_counts=True)
            peaks_number[temp[0]] = temp[1]
            peaks = pad_sequence((peaks[1] + 1).split(peaks_number.tolist()), padding_value=100,
                                 batch_first=True)

        return peaks, peaks_number

    def morlet_transform(self, data, sfreq, frequency_range, fc=1, fwhm_tc=3):
        """
        Description:
            使用morlet计算时频图
            from brainstorm

        Input:
            :param data: torch.tensor, double, shape(n_batch*n_samples)
                输入数据
            :param sfreq: number, double
                信号采样率
            :param frequency_range: torch.tensor, double, shape(n_frequency_samples)
                频率的范围
            :param fc: number, double
                频率维度
            :param fwhm_tc: number, double
                时间维度
        Return:
            :return time_frequency: torch.tensor, double, shape(n_batch*n_samples*n_frequency_samples), in_device
                peaks位置
        """
        # 准备数据
        data = data.unsqueeze(0).to(self.device) if len(data.shape) == 1 else data.to(self.device)
        data = data - data.mean(dim=-1, keepdim=True)
        # 计算morlet小波变换参数
        scales = frequency_range / fc
        sigma_tc = fwhm_tc / np.sqrt(8 * np.log(2))
        sigma_t = sigma_tc / scales
        # 对每一个scale计算卷积核大小和权重
        x_vale = [y * torch.arange(-3 * x, 3 * x, 1 / sfreq) for x, y in zip(sigma_t, scales)]
        weight = [x * (sigma_tc * torch.tensor(np.pi).sqrt()).pow(-0.5) *
                  (-y.pow(2) / (2 * sigma_tc * sigma_tc)).exp() * (1j * 2 * np.pi * fc * y).exp()
                  for x, y in zip(scales.sqrt(), x_vale)]
        # 计算小波参数
        conv_real, conv_imag = [], []
        with torch.no_grad():
            # 计算卷积核
            for x in weight:
                conv_1 = torch.nn.Conv2d(1, 1, [1, x.shape[0]],
                                         padding=(0, int(np.ceil((x.shape[0] - 1) / 2))), bias=False).to(self.device)
                conv_1.state_dict()['weight'].copy_(x.real.unsqueeze(0).unsqueeze(0).unsqueeze(0))
                conv_real.append(conv_1)
                conv_2 = torch.nn.Conv2d(1, 1, [1, x.shape[0]],
                                         padding=(0, int(np.ceil((x.shape[0] - 1) / 2))), bias=False).to(self.device)
                conv_2.state_dict()['weight'].copy_(x.imag.unsqueeze(0).unsqueeze(0).unsqueeze(0))
                conv_imag.append(conv_2)
            real = torch.stack([x(data.unsqueeze(0).unsqueeze(0).float())[0, 0][:, :data.shape[1]]
                                for x in conv_real], dim=-1) / sfreq
            imag = torch.stack([x(data.unsqueeze(0).unsqueeze(0).float())[0, 0][:, :data.shape[1]]
                                for x in conv_imag], dim=-1) / sfreq
        time_frequency = real.pow(2) + imag.pow(2)
        time_frequency = time_frequency.unsqueeze(0) if len(time_frequency.shape) == 2 else time_frequency
        self.del_var()

        return time_frequency

    def cal_time_frequency_maps(self, data, sfreq=1000., frequency_range=(80., 200.), method=0, chunk=1000):
        """
        Description:
            计算时频图

        Input:
            :param data: torch.tensor, double, shape(n_batch*n_samples)
                输入数据
            :param sfreq: number, double
                信号采样率
            :param frequency_range: tuple, double
                [频率下限， 频率上限]
            :param method: number, long
                1: morlet transform; 2: s-t transform; 3: short-time-fft
            :param chunk: number, long
                用于分块运行的，每块个数
        Return:
            :return data_tf: torch.tensor, double, shape(n_batch*n_samples*n_frequency_samples), in_device
                信号的时频图
        """
        if method == 0:
            # 使用morlet小波变换
            data_tf = torch.zeros(data.shape[0], data.shape[1],
                                  torch.arange(frequency_range[0], frequency_range[1]).shape[0]).to(self.device)
            for idx in range(int(np.ceil(data.shape[0] / chunk))):
                index = torch.arange(idx * chunk, min((idx + 1) * chunk, data.shape[0]))
                data_tf[index] = self.morlet_transform(data[index], sfreq,
                                                       torch.arange(frequency_range[0], frequency_range[1]),
                                                       fc=1, fwhm_tc=3)
                self.del_var()
        elif method == 1:
            # 使用ST变换
            data_tf = mne.time_frequency.tfr_array_stockwell(data.unsqueeze(0).cpu().numpy(), sfreq,
                                                             frequency_range[0], frequency_range[1], n_jobs=10)[0]
            data_tf = torch.tensor(data_tf).to(self.device).transpose(1, 2)
        elif method == 2:
            # 使用短时傅立叶变换
            data_tf = torch.zeros(data.shape[0], data.shape[1],
                                  torch.arange(frequency_range[0], frequency_range[1]).shape[0]).to(self.device)
            for idx in range(int(np.ceil(data.shape[0] / chunk))):
                index = torch.arange(idx * chunk, min((idx + 1) * chunk, data.shape[0]))
                data_tf[index] = torch.stft(
                    data[index].to(self.device), n_fft=int(sfreq), hop_length=1,
                    window=torch.hann_window(64).to(self.device),
                    win_length=64).pow(2).sum(dim=-1)[:,
                                 torch.arange(frequency_range[0], frequency_range[1]).long(), :-1].transpose(1, 2)
                self.del_var()

        return data_tf

    def cal_time_frequency_maps_windows(self, data, sfreq=1000., frequency_range=(80., 200.), method=0,
                                        extend_window=0.15, tf_window=0.25):
        """
        Description:
            计算data内, 中间时间的tf_window长度的时频图。

        Input:
            :param data: torch.tensor, double, shape(n_batch*n_samples)
                输入数据
            :param sfreq: number, double
                信号采样率
            :param frequency_range: tuple, double
                [频率下限， 频率上限]
            :param method: number, long
                1: morlet transform; 2: s-t transform; 3: short-time-fft
            :param extend_window: number, double
                扩展的时间窗长度，用于排除边界效应，单位S
            :param tf_window: number, double
                计算tf的时间窗长度，单位S
        Return:
            :return data_tf: torch.tensor, double, shape(n_batch*n_tf_window_samples*n_frequency_samples), in_device
                信号的时频图
        """

        # 计算截取时间信号的窗
        extend_window_half = int(extend_window * sfreq)
        tf_window_half = int(tf_window / 2 * sfreq)
        tf_samples = torch.arange(-tf_window_half - extend_window_half, tf_window_half + extend_window_half)
        # 截取时间信号
        data_temp = data[:, int(data.shape[-1] / 2) + tf_samples]
        # 计算tf
        tf_temp = self.cal_time_frequency_maps(data=data_temp, sfreq=sfreq,
                                               frequency_range=frequency_range, method=method)
        # 去除扩展时间窗信号
        tf = tf_temp[:, extend_window_half:-extend_window_half, :]

        return tf

    @staticmethod
    def cal_vs_signal(leadfield, source_ori, inv_covariance, meg_data=None):
        """
        Description:
            使用SAM-Beamformer计算源空间信号。

        Input:
            :param leadfield: torch.tensor, double, shape(n_sources*3*n_channels), in_device
                leadfield矩阵
            :param source_ori: torch.tensor, double, shape(n_batch*n_sources*3), in_device
                源空间信号方向矩阵，n_batch为同时计算的矩阵个数
            :param inv_covariance: torch.tensor, double, shape(n_channels*n_channels), in_device
                数据协方差矩阵的逆矩阵
            :param meg_data: torch.tensor, double, shape(n_channels*n_samples), in_device
                MEG数据
        Return:
            :return weight: torch.tensor, double, shape(n_batch*n_sources*n_channels), in_device
                溯源逆算子
            :return data_vs: torch.tensor, double, shape(n_batch*n_sources*n_samples), in_device
                源空间信号
        """
        gain = torch.bmm(source_ori.transpose(0, 1), leadfield)
        temp = torch.bmm(gain, inv_covariance.unsqueeze(0).repeat(gain.shape[0], 1, 1))
        weight = (temp / torch.sum(temp * gain, dim=-1, keepdim=True)).transpose(0, 1)
        if meg_data is not None:
            data_vs = torch.mm(weight.reshape(-1, weight.shape[-1]), meg_data)
            data_vs = data_vs.reshape(weight.shape[0], weight.shape[1], -1)
        else:
            data_vs = None

        return weight, data_vs

    def cal_vs_signal_multi_band(self, leadfield=None, ori_hfo=None, ori_ied=None, inv_covariance_hfo=None,
                                 inv_covariance_ied=None, meg_data_hfo=None, meg_data_ied=None, meg_data_raw=None):
        """
        Description:
            计算HFO和IED频段的源空间信号。
            如果ori_hfo, inv_covariance_hfo, meg_data_hfo为None，则不计算HFO频段，返回值为None。
            如果ori_ied, inv_covariance_ied, meg_data_ied为None，则不计算IED频段，返回值为None。
            如果meg_data_raw不为None，则计算IED/HFO最优方向的未滤波源空间信号。

        Input:
            :param leadfield: torch.tensor, double, shape(n_dipoles*3*n_channels), in_device
                多个dipole的leadfield矩阵
            :param meg_data_raw: torch.tensor, double, shape(n_channels*n_samples)
                未滤波MEG数据
            :param ori_hfo: torch.tensor, double, shape(n_batch*n_sources*3)
                源空间信号方向矩阵，n_batch为同时计算的矩阵个数(HFO的最优方向)
            :param inv_covariance_hfo: torch.tensor, double, shape(n_channels*n_channels)
                HFO频段数据协方差矩阵的逆矩阵(HFO的最优方向)
            :param meg_data_hfo: torch.tensor, double, shape(n_channels*n_samples)
                HFO频段滤波MEG数据
            :param ori_ied: torch.tensor, double, shape(n_batch*n_sources*3)
                源空间信号方向矩阵，n_batch为同时计算的矩阵个数(IED的最优方向)
            :param inv_covariance_ied: torch.tensor, double, shape(n_channels*n_channels)
                IED频段数据协方差矩阵的逆矩阵(HFO的最优方向)
            :param meg_data_ied: torch.tensor, double, shape(n_channels*n_samples)
                IED频段滤波MEG数据
        Return:
            :return data_vs_hfo_filter: torch.tensor, double, shape(n_batch*n_sources*n_channels), in_device
                HFO最优方向的源空间信号(HFO频段滤波)
            :return data_vs_hfo_raw: torch.tensor, double, shape(n_batch*n_sources*n_channels), in_device
                HFO最优方向的源空间信号(未滤波)
            :return data_vs_ied_filter: torch.tensor, double, shape(n_batch*n_sources*n_channels), in_device
                IED最优方向的源空间信号(IED频段滤波)
            :return data_vs_ied_raw: torch.tensor, double, shape(n_batch*n_sources*n_channels), in_device
                IED最优方向的源空间信号(未滤波)
        """
        data_vs_hfo_filter, data_vs_hfo_raw, data_vs_ied_filter, data_vs_ied_raw = None, None, None, None

        # 计算HFO最优方向的源空间信号
        if ori_hfo is not None and meg_data_hfo is not None and inv_covariance_hfo is not None:
            # HFO频段滤波
            ori_hfo, inv_covariance_hfo, meg_data_hfo = \
                ori_hfo.to(self.device), inv_covariance_hfo.to(self.device), meg_data_hfo.to(self.device)
            weight_hfo, data_vs_hfo_filter = \
                self.cal_vs_signal(leadfield=leadfield, source_ori=ori_hfo, inv_covariance=inv_covariance_hfo,
                                   meg_data=meg_data_hfo)
            # 不进行滤波
            if meg_data_raw is not None:
                meg_data_raw = meg_data_raw.to(self.device)
                data_vs_hfo_raw = torch.mm(weight_hfo.reshape(-1, weight_hfo.shape[-1]), meg_data_raw)
                data_vs_hfo_raw = data_vs_hfo_raw.reshape(weight_hfo.shape[0], weight_hfo.shape[1], -1)

        # 计算IED最优方向的源空间信号
        if ori_ied is not None and meg_data_ied is not None and inv_covariance_ied is not None:
            # IED频段滤波
            ori_ied, inv_covariance_ied, meg_data_ied = \
                ori_ied.to(self.device), inv_covariance_ied.to(self.device), meg_data_ied.to(self.device)
            weight_ied, data_vs_ied_filter = \
                self.cal_vs_signal(leadfield=leadfield, source_ori=ori_ied, inv_covariance=inv_covariance_ied,
                                   meg_data=meg_data_ied)
            # 不进行滤波
            if meg_data_raw is not None:
                meg_data_raw = meg_data_raw.to(self.device)
                data_vs_ied_raw = torch.mm(weight_ied.reshape(-1, weight_ied.shape[-1]), meg_data_raw)
                data_vs_ied_raw = data_vs_ied_raw.reshape(weight_ied.shape[0], weight_ied.shape[1], -1)

        return data_vs_hfo_filter.float(), data_vs_hfo_raw.float(), data_vs_ied_filter.float(), data_vs_ied_raw.float()

    def cal_segments_envelope_threshold(self, signal_envelope, envelope_threshold=2, segments_duration_threshold=0.02):
        """
        Description:
            计算满足幅度和持续时间要求的片段。


        Input:
            :param signal_envelope: torch.tensor, double, shape(n_batch*n_sources*n_channels), in_device
                HFO最优方向的源空间信号(HFO频段滤波)
            :param envelope_threshold: number, double
                包络幅度阈值
            :param segments_duration_threshold: number, double
                片段最小持续时间阈值，单位s
        Return:
            :return segments: torch.tensor, long, shape(n_segments*4)
                满足要求的segments: [batch_index, vs_channel_index, segment_begin, segment_end]
        """

        # 计算满足幅度要求的采样点
        signal_envelope_diff = torch.cat([torch.zeros_like(signal_envelope)[:, :, :1],
                                          (signal_envelope > envelope_threshold).float(),
                                          torch.zeros_like(signal_envelope)[:, :, :1]], dim=-1).diff(dim=-1)
        # 计算segment的开始点
        segments_begin = torch.where(signal_envelope_diff.reshape(-1) == 1)[0]
        # 计算segment的结束点
        segments_end = torch.where(signal_envelope_diff.reshape(-1) == -1)[0]
        # 计算满足持续时间要求的segment
        segments_index = torch.where(segments_end - segments_begin >=
                                     segments_duration_threshold * self.data_info['sfreq'])[0]
        # 转换为[batch_index, vs_channel_index, segment_begin, segment_end]格式
        segments = torch.tensor(
            np.unravel_index(segments_begin[segments_index].cpu().numpy(), signal_envelope_diff.shape) +
            np.unravel_index(segments_end[segments_index].cpu().numpy(), signal_envelope_diff.shape)[2:3]).t()
        segments = segments if len(segments.shape) == 2 else segments.unsqueeze(0)
        segments[:, -1] = segments[:, -1] - 1

        return segments

    def cal_segments_oscillation_threshold(self, data, oscillation_threshold=4):
        """
        Description:
            计算满足振荡个数要求的片段。

        Input:
            :param data: torch.tensor, double, shape(n_batch*n_samples), in_device
                需要判断的数据片段
            :param oscillation_threshold: number, double
                振荡个数阈值
        Return:
            :return index: torch.tensor, bool, shape(n_batch)
                每个batch数据是否满足要求
        """
        _, peaks_number = self.peak_detect(data)
        index = peaks_number >= oscillation_threshold * 2 - 1

        return index

    def cal_segments_tf_entropy_threshold(self, tf, entropy_threshold=1.25):
        """
        Description:
            计算满最大时频熵要求的片段。

        Input:
            :param tf: torch.tensor, double, shape(n_batch*n_time_samples*n_frequency_samples), in_device
                需要判断的时频图片段
            :param entropy_threshold: number, double
                时频熵的阈值
        Return:
            :return index: torch.tensor, bool, shape(n_batch)
                每个batch数据是否满足要求
        """
        tf_temp = tf / tf.sum(dim=-1, keepdim=True)
        tf_entropy = -(tf_temp * tf_temp.log10()).sum(dim=-1).to(self.device)
        tf_entropy_max_min_ratio = tf_entropy.nan_to_num(-100).amax(dim=-1) / tf_entropy.nan_to_num(100).amin(dim=-1)
        index = tf_entropy_max_min_ratio <= entropy_threshold

        return index

    def cal_segments_tf_power_ratio_threshold(self, tf, frequency_range=(80., 200.), power_ratio_threshold=1.25):
        """
        Description:
            计算满最小能量比(high/trough)要求的片段。

        Input:
            :param tf: torch.tensor, double, shape(n_batch*n_time_samples*n_frequency_samples), in_device
                需要判断的时频图片段
            :param frequency_range: tuple, double
                [频率下限， 频率上限]
            :param power_ratio_threshold: number, double
                最小能量比阈值
        Return:
            :return index: torch.tensor, bool, shape(n_batch)
                每个batch数据是否满足要求
        """
        # 计算最大能量的瞬时功率谱
        max_power_time = tf.nan_to_num(-1000)[:, :, frequency_range[0] - 1:].amax(dim=-1).argmax(dim=-1)
        max_instant_power = tf.take_along_dim(max_power_time.unsqueeze(-1).unsqueeze(-1), dim=1).squeeze()
        max_instant_power = max_instant_power.unsqueeze(0) if len(max_instant_power.shape) == 1 else max_instant_power
        # 计算高能量（frequency_range[0]之后的最大能量）
        high_power = max_instant_power[:, frequency_range[0] - 1:].amax(dim=-1)
        high_power_frequency = max_instant_power[:, frequency_range[0] - 1:].argmax(dim=-1)
        # 计算谷能量（40Hz到high_power之间的最大能量）
        trough_power = torch.tensor([x[40 - 1: y + frequency_range[0] - 1].min()
                                     for x, y in zip(max_instant_power, high_power_frequency)]).to(self.device)
        # 计算能量比
        high_to_trough_power_ratio = high_power / trough_power
        index = high_to_trough_power_ratio >= power_ratio_threshold

        return index

    def cal_hfo_events_thresholding_method(self, ieds_segment_samples, max_oris_hfo, max_oris_ied,
                                           inv_covariance_hfo, inv_covariance_ied,
                                           hfo_window_samples, baseline_window_samples_hfo,
                                           amplitude_threshold=2, duration_threshold=0.02, oscillation_threshold=4,
                                           frequency_range=(80, 200), entropy_threshold=1.25,
                                           power_ratio_threshold=1.25):
        """
        Description:
            使用阈值的方法，计算hfo segments和hfo events

        Input:
            :param ieds_segment_samples: np.array, double, shape(n_ieds*n_samples)
                ied片段在原始数据中的采样点位置
            :param max_oris_hfo: np.array, double, shape(n_ieds*n_segments*n_vs_channels*3)
                hfo最优源空间信号方向
            :param max_oris_ied: np.array, double, shape(n_ieds*n_segments*n_vs_channels*3)
                ied最优源空间信号方向
            :param inv_covariance_hfo: np.array, double, shape(n_channels*n_channels)
                hfo频段的协方差矩阵逆矩阵
            :param inv_covariance_ied: np.array, double, shape(n_channels*n_channels)
                ied频段的协方差矩阵逆矩阵
            :param hfo_window_samples: np.array, double, shape(n_segments*n_hfo_samples)
                hfo segments在ieds_segment_samples中的采样点位置
            :param baseline_window_samples_hfo: np.array, double, shape(n_segments*n_baseline_samples)
                hfo segments对应的baseline在ieds_segment_samples中的采样点位置
            :param amplitude_threshold: number, double
                hfo最小幅度阈值
            :param duration_threshold: number, double
                hfo小持续时间阈值，单位s
            :param oscillation_threshold: number, double
                hfo最小振荡个数阈值
            :param frequency_range: tuple, double
                hfo的[频率下限， 频率上限]
            :param entropy_threshold: number, double
                hfo最大时频熵的阈值
            :param power_ratio_threshold: number, double
                hfo最小最小能量比阈值
        Return:
            :return hfo_events:list, [torch.tensor, double, shape(n_segments*9)], shape(n_events), in_device
                每个hfo event的hfo segments，[ied index, hfo segments index, vs channel,
                                            sample index in 100 ms hfo segments(begin),
                                            sample index in 100 ms hfo segments(end),
                                            sample index in hfo segments(begin),
                                            sample index in hfo segments(end),
                                            sample index in raw data(begin),
                                            sample index in raw data(end),]
            :return vs_hfo_filter: list, [torch.tensor, double, shape(n_segments*n_samples)], shape(n_events), in_device
                每个hfo event的hfo segments，HFO最优方向的源空间信号(HFO频段滤波)
            :return vs_hfo_raw: list, [torch.tensor, double, shape(n_segments*n_samples)], shape(n_events), in_device
                每个hfo event的hfo segments，HFO最优方向的源空间信号(未滤波)
            :return vs_ied_filter: list, [torch.tensor, double, shape(n_segments*n_samples)], shape(n_events), in_device
                每个hfo event的hfo segments，IED最优方向的源空间信号(IED频段滤波)
            :return vs_ied_raw: list, [torch.tensor, double, shape(n_segments*n_samples)], shape(n_events), in_device
                每个hfo event的hfo segments，IED最优方向的源空间信号(未滤波)
        """

        # 将数据转转换为torch.tensor
        ieds_segment_samples, max_oris_hfo, max_oris_ied = \
            torch.tensor(ieds_segment_samples), torch.tensor(max_oris_hfo), torch.tensor(max_oris_ied)
        inv_covariance_hfo, inv_covariance_ied = torch.tensor(inv_covariance_hfo), torch.tensor(inv_covariance_ied)
        hfo_window_samples, baseline_window_samples_hfo = torch.tensor(hfo_window_samples), \
                                                          torch.tensor(baseline_window_samples_hfo)
        # 获取hfo segments
        hfo_segments, vs_hfo_filter, vs_hfo_raw, vs_ied_filter, vs_ied_raw = [], [], [], [], []
        for ied_index, ied_segment_samples, ori_hfo, ori_ied in \
                zip(torch.arange(ieds_segment_samples.shape[0]), ieds_segment_samples, max_oris_hfo, max_oris_ied):
            hfo_segments_temp, vs_hfo_filter_temp, vs_hfo_raw_temp, vs_ied_filter_temp, vs_ied_raw_temp = \
                self.cal_hfo_segments_thresholding_method(
                    ied_segment_samples=ied_segment_samples, ori_hfo=ori_hfo, ori_ied=ori_ied,
                    inv_covariance_hfo=inv_covariance_hfo, inv_covariance_ied=inv_covariance_ied,
                    hfo_samples=hfo_window_samples, baseline_samples_hfo=baseline_window_samples_hfo,
                    amplitude_threshold=amplitude_threshold, duration_threshold=duration_threshold,
                    oscillation_threshold=oscillation_threshold, frequency_range=frequency_range,
                    entropy_threshold=entropy_threshold,
                    power_ratio_threshold=power_ratio_threshold)
            if len(hfo_segments_temp) > 0:
                hfo_segments.append(torch.cat([torch.ones_like(hfo_segments_temp[:, :1]).to(self.device) * ied_index,
                                               hfo_segments_temp.to(self.device)], dim=1))
                vs_hfo_filter.append(vs_hfo_filter_temp)
                vs_hfo_raw.append(vs_hfo_raw_temp)
                vs_ied_filter.append(vs_ied_filter_temp)
                vs_ied_raw.append(vs_ied_raw_temp)
        # 将hfo segments聚类成hfo events
        if len(hfo_segments) > 0:
            hfo_segments, vs_hfo_filter, vs_hfo_raw, vs_ied_filter, vs_ied_raw = \
                torch.cat(hfo_segments), torch.cat(vs_hfo_filter), torch.cat(vs_hfo_raw), \
                torch.cat(vs_ied_filter), torch.cat(vs_ied_raw)
            # 聚类
            hfo_event_clusters = self.cluster_events(segments_times=hfo_segments[:, -2:])
            # 删除events内重复的通道
            hfo_event_clusters = self.remove_repeat_channels(hfo_segments=hfo_segments, clusters=hfo_event_clusters,
                                                             middle_segments_index=int(hfo_window_samples.shape[1] / 2))
            # 输出
            hfo_events = [hfo_segments[x] for x in hfo_event_clusters]
            vs_hfo_filter = [vs_hfo_filter[x] for x in hfo_event_clusters]
            vs_hfo_raw = [vs_hfo_raw[x] for x in hfo_event_clusters]
            vs_ied_filter = [vs_ied_filter[x] for x in hfo_event_clusters]
            vs_ied_raw = [vs_ied_raw[x] for x in hfo_event_clusters]
        else:
            hfo_events, vs_hfo_filter, vs_hfo_raw, vs_ied_filter, vs_ied_raw = [], [], [], [], []

        return hfo_events, vs_hfo_filter, vs_hfo_raw, vs_ied_filter, vs_ied_raw

    def cal_hfo_segments_thresholding_method(self, ied_segment_samples, ori_hfo, ori_ied,
                                             inv_covariance_hfo, inv_covariance_ied, hfo_samples, baseline_samples_hfo,
                                             amplitude_threshold=2, duration_threshold=0.02, oscillation_threshold=4,
                                             frequency_range=(80, 200), entropy_threshold=1.25,
                                             power_ratio_threshold=1.25):
        """
        Description:
            使用阈值的方法，计算hfo segments和hfo events

        Input:
            :param ied_segment_samples: np.array, double, shape(n_samples)
                ied片段在原始数据中的采样点位置
            :param ori_hfo: np.array, double, shape(n_segments*n_vs_channels*3)
                hfo最优源空间信号方向
            :param ori_ied: np.array, double, shape(n_segments*n_vs_channels*3)
                ied最优源空间信号方向
            :param inv_covariance_hfo: np.array, double, shape(n_channels*n_channels)
                hfo频段的协方差矩阵逆矩阵
            :param inv_covariance_ied: np.array, double, shape(n_channels*n_channels)
                ied频段的协方差矩阵逆矩阵
            :param hfo_samples: np.array, double, shape(n_segments*n_hfo_samples)
                hfo segments在ieds_segment_samples中的采样点位置
            :param baseline_samples_hfo: np.array, double, shape(n_segments*n_baseline_samples)
                hfo segments对应的baseline在ieds_segment_samples中的采样点位置
            :param amplitude_threshold: number, double
                hfo最小幅度阈值
            :param duration_threshold: number, double
                hfo小持续时间阈值，单位s
            :param oscillation_threshold: number, double
                hfo最小振荡个数阈值
            :param frequency_range: tuple, double
                hfo的[频率下限， 频率上限]
            :param entropy_threshold: number, double
                hfo最大时频熵的阈值
            :param power_ratio_threshold: number, double
                hfo最小最小能量比阈值
        Return:
            :return hfo_segments:torch.tensor, double, shape(n_segments*9), in_device
                每个fo segments，[ied index, hfo segments index, vs channel,
                                 sample index in 100 ms hfo segments(begin),
                                 sample index in 100 ms hfo segments(end),
                                 sample index in hfo segments(begin),
                                 sample index in hfo segments(end),
                                 sample index in raw data(begin),
                                 sample index in raw data(end),]
            :return vs_hfo_filter: torch.tensor, double, shape(n_segments*n_samples), in_device
                每个hfo segments，HFO最优方向的源空间信号(HFO频段滤波)
            :return vs_hfo_raw: torch.tensor, double, shape(n_segments*n_samples), in_device
                每个hfo segments，HFO最优方向的源空间信号(未滤波)
            :return vs_ied_filter: torch.tensor, double, shape(n_segments*n_samples), in_device
                每个hfo segments，IED最优方向的源空间信号(IED频段滤波)
            :return vs_ied_raw: torch.tensor, double, shape(n_segments*n_samples), in_device
                每个hfo segments，IED最优方向的源空间信号(未滤波)
        """

        # step1: 计算ied segments的源空间信号(HFO和IED频段)
        meg_data_raw = self.get_raw_data()[:, ied_segment_samples]
        meg_data_ied = self.get_raw_data_ied()[:, ied_segment_samples]
        meg_data_hfo = self.get_raw_data_hfo()[:, ied_segment_samples]
        vs_hfo_filter, vs_hfo_raw, vs_ied_filter, vs_ied_raw = \
            self.cal_vs_signal_multi_band(ori_hfo=ori_hfo, leadfield=self.get_leadfield().clone().to(self.device),
                                          inv_covariance_hfo=inv_covariance_hfo, meg_data_hfo=meg_data_hfo,
                                          ori_ied=ori_ied,
                                          inv_covariance_ied=inv_covariance_ied, meg_data_ied=meg_data_ied,
                                          meg_data_raw=meg_data_raw)

        # step2: 计算满足幅度和持续时间要求的hfo segments
        # 获取vs信号的包络
        hilbert_hfo_filter = self.hilbert(sig=vs_hfo_filter)
        hilbert_hfo_filter_hfo = hilbert_hfo_filter.take_along_dim(
            hfo_samples.unsqueeze(1).repeat(1, hilbert_hfo_filter.shape[1], 1).to(self.device), dim=-1)
        hilbert_hfo_filter_baseline = hilbert_hfo_filter.take_along_dim(
            baseline_samples_hfo.unsqueeze(1).repeat(1, hilbert_hfo_filter.shape[1], 1).to(self.device), dim=-1)
        # 对vs信号的包络进行z-score归一化，获得hfo相对于baseline的强度
        hilbert_hfo_filter_hfo = (hilbert_hfo_filter_hfo - hilbert_hfo_filter_baseline.mean(dim=-1, keepdim=True)) / \
                                 hilbert_hfo_filter_baseline.std(dim=-1, keepdim=True)
        # 获取满足幅度和持续时间要求的segments
        hfo_segments = self.cal_segments_envelope_threshold(signal_envelope=hilbert_hfo_filter_hfo,
                                                            envelope_threshold=amplitude_threshold,
                                                            segments_duration_threshold=duration_threshold)
        self.del_var()
        if len(hfo_segments) == 0:
            return hfo_segments, [], [], [], []
        hfo_segments = hfo_segments.unsqueeze(0) if len(hfo_segments.shape) == 1 else hfo_segments

        # step3: 计算满足最小振荡个数要求的hfo segments
        # 获取vs信号数据
        vs_hfo_filter_hfo = vs_hfo_filter.take_along_dim(
            hfo_samples.unsqueeze(1).repeat(1, vs_hfo_filter.shape[1], 1).to(self.device), dim=-1)
        vs_hfo_filter_hfo_segments = pad_sequence([vs_hfo_filter_hfo[x[0], x[1], x[2]:x[3]] for x in hfo_segments],
                                                  padding_value=np.nan).t()
        # 计算满足最小振荡个数的index
        segments_index_oscillation = self.cal_segments_oscillation_threshold(
            data=vs_hfo_filter_hfo_segments, oscillation_threshold=oscillation_threshold)
        hfo_segments = hfo_segments[segments_index_oscillation]
        if len(hfo_segments) == 0:
            return hfo_segments, [], [], [], []
        hfo_segments = hfo_segments.unsqueeze(0) if len(hfo_segments.shape) == 1 else hfo_segments
        self.del_var()

        # step4: 计算满足时频熵要求的hfo segments, 确保hfo的时频图是"孤岛"
        # 计算时频图
        tf_window = ((hfo_samples - vs_hfo_raw.shape[-1] / 2).abs().max().long() +
                     self.data_info['sfreq'] * 0.05) * 2 / self.data_info['sfreq']
        vs_hfo_raw_hfo_segments = vs_hfo_raw[(hfo_segments[:, 0], hfo_segments[:, 1])]
        tf_hfo_segments = \
            self.cal_time_frequency_maps_windows(data=vs_hfo_raw_hfo_segments, sfreq=self.data_info['sfreq'],
                                                 frequency_range=(1., frequency_range[1]),
                                                 method=0, extend_window=0.15, tf_window=tf_window)
        self.del_var()
        # 截取hfo segments的时频图
        tf_hfo_segments_temp = pad_sequence([x[hfo_samples[y[0]][y[2]:y[3]] +
                                               int(self.data_info['sfreq'] * tf_window / 2 - vs_hfo_raw.shape[-1] / 2)]
                                             for x, y in zip(tf_hfo_segments, hfo_segments)], padding_value=np.nan)
        tf_hfo_segments_temp = tf_hfo_segments_temp.transpose(0, 1)
        # 计算满足时频熵要求的index
        segments_index_tf_entropy = self.cal_segments_tf_entropy_threshold(
            tf=tf_hfo_segments_temp[:, :, frequency_range[0] - 1:], entropy_threshold=entropy_threshold)
        hfo_segments = hfo_segments[segments_index_tf_entropy]
        if len(hfo_segments) == 0:
            return hfo_segments, [], [], [], []
        hfo_segments = hfo_segments.unsqueeze(0) if len(hfo_segments.shape) == 1 else hfo_segments

        # step5: 计算满足最小能量比(high/trough)要求的hfo segments, 确保hfo的时频图是"孤岛"
        tf_hfo_segments_temp = tf_hfo_segments[segments_index_tf_entropy]
        segments_index_power_ratio = self.cal_segments_tf_power_ratio_threshold(
            tf=tf_hfo_segments_temp, frequency_range=frequency_range, power_ratio_threshold=power_ratio_threshold)
        hfo_segments = hfo_segments[segments_index_power_ratio]
        if len(hfo_segments) == 0:
            return hfo_segments, [], [], [], []
        hfo_segments = hfo_segments.unsqueeze(0) if len(hfo_segments.shape) == 1 else hfo_segments
        self.del_var()

        # Step 6: 输出
        vs_hfo_filter = vs_hfo_filter[:, hfo_segments[:, 1]].take_along_dim(
            hfo_segments[:, 0].reshape(1, hfo_segments.shape[0], 1).to(self.device), dim=0)[0]
        vs_hfo_raw = vs_hfo_raw[:, hfo_segments[:, 1]].take_along_dim(
            hfo_segments[:, 0].reshape(1, hfo_segments.shape[0], 1).to(self.device), dim=0)[0]
        vs_ied_filter = vs_ied_filter[:, hfo_segments[:, 1]][0]
        vs_ied_raw = vs_ied_raw[:, hfo_segments[:, 1]][0]
        # 将hfo的时间加入到hfo_segments
        hfo_segments = torch.cat([hfo_segments,
                                  hfo_samples[hfo_segments[:, 0].long(), :].take_along_dim(hfo_segments[:, 2:4], dim=1),
                                  ied_segment_samples[hfo_samples[hfo_segments[:, 0].long(),
                                                      :].take_along_dim(hfo_segments[:, 2:4], dim=1)]], dim=1)
        return hfo_segments, vs_hfo_filter, vs_hfo_raw, vs_ied_filter, vs_ied_raw

    def cluster_events(self, segments_times):
        """
        Description:
            根据segments的时间，将segments进行聚类: 如果时间之间的overlap重和50%以上，则聚为一类

        Input:
            :param segments_times: torch.tensor, double, shape(n_segments*2), in_device
                segments的起始和终止时间
        Return:
            :return clusters: list, long, shape(n_events)
                每个cluster内包含的segment index
        """
        segments = torch.cat([segments_times.long(),
                              torch.arange(segments_times.shape[0]).to(self.device).unsqueeze(1)], dim=1)
        segments = segments[segments[:, 0].sort()[1]]
        # 计算cluster
        clustered_segments, remained_segments = [segments[:1]], segments[1:]
        while len(remained_segments) > 0:
            segments_temp, remained_segments = remained_segments[0], remained_segments[1:]
            segment_in_cluster_index = []
            for index, cls in enumerate(clustered_segments):
                # 计算segments_temp和现有的cluster之间的，时间上的overlap.
                temp = torch.tensor([cls[:, 0].float().mean().long(), cls[:, 1].float().mean().long()]).to(self.device)
                time_overlap = torch.where(
                    torch.cat([torch.arange(segments_temp[0], segments_temp[1]),
                               torch.arange(temp[0], temp[1])]).unique(return_counts=True)[1] > 1)[0].shape[0]
                # 如果time_overlap > segment_duration / 2 and time_overlap > cluster_duration / 2，认为是重合
                if time_overlap >= segments_temp[:2].diff() / 2:
                    segment_in_cluster_index.append(index)
            if len(segment_in_cluster_index) == 0:
                # segments_temp属于新的cluster
                clustered_segments.append(segments_temp.unsqueeze(0))
            elif len(segment_in_cluster_index) == 1:
                # segments_temp属于之前的一个cluster
                clustered_segments[segment_in_cluster_index[0]] = \
                    torch.cat([clustered_segments[segment_in_cluster_index[0]],
                               segments_temp.unsqueeze(0)])
            elif len(segment_in_cluster_index) > 1:
                # segments_temp属于之前的几个个cluster，将clusters聚集在一起
                segment_in_cluster_index.sort(reverse=True)
                temp = torch.cat([clustered_segments.pop(x).reshape(-1, 3) for x in segment_in_cluster_index])
                temp = torch.cat([temp, segments_temp.unsqueeze(0)])
                clustered_segments.append(temp)
        clusters = [x[:, -1] for x in clustered_segments]

        return clusters

    def remove_repeat_channels(self, hfo_segments, clusters, middle_segments_index=3):
        """
        Description:
            去除cluster的events内重复的通道

        Input:
            :param hfo_segments: torch.tensor, double, shape(n_segments*9), in_device
                cal_hfo_segments_thresholding_method的输出
            :param clusters: list, long, shape(n_events)
                每个cluster内包含的segment index
            :param middle_segments_index: number, long,
                距离ied主peak最近的hfo segments
        Return:
            :return clusters: list, long, shape(n_events)
                每个cluster内包含的segment index
        """
        hfo_events = [hfo_segments[x] for x in clusters]
        for index, hfo_event in enumerate(hfo_events):
            # 获取events内重复的通道
            temp = hfo_event[:, 2].unique(return_inverse=True, return_counts=True)
            removed_channels = []
            for x in torch.where(temp[2] > 1)[0]:
                # 根据距离主peak的距离，保留最近的hfo通道
                repeated_channels = torch.where(temp[1] == x)[0]
                removed_channels_tmep = repeated_channels[
                    torch.where((hfo_event[repeated_channels, 1] - middle_segments_index).abs() >
                                (hfo_event[repeated_channels, 1] - middle_segments_index).abs().min())[0]]
                removed_channels.append(removed_channels_tmep)
            # 删除通道
            if len(removed_channels) > 0:
                clusters[index] = torch.stack([x for i, x in enumerate(clusters[index])
                                               if i not in torch.cat(removed_channels)])
        self.del_var()

        return clusters

    def cal_max_peak_feature(self, peak_index, features, interest_windows):
        """
        Description:
            计算在interest_windows内的peak，对应的特征的最大值。
            使用pad_sequence将list转换为矩阵

        Input:
            :param peak_index: list, torch.tensor, double, shape(n_batch), in_device
                每个n_batch中peak的采样点位置
            :param features: list, torch.tensor, double, shape(n_batch), in_device
                每个n_batch中peak的特征
            :param interest_windows: list, long, shape(2)
                感兴趣时间窗的起始和结束
        Return:
            :return features_interest_max: torch.tensor, double, shape(n_batch), in_device
                interest_windows内的最大peak特征值
        """

        features_interest_max = torch.zeros(len(peak_index)).to(self.device)
        # 将peak_index转换为矩阵
        peak_index_samples = pad_sequence(peak_index).t()
        # 回去感兴趣时间窗内的peak index
        temp = ((peak_index_samples >= interest_windows[0]) & (peak_index_samples <= interest_windows[1])).any(dim=-1)
        peak_index_samples = torch.where((peak_index_samples >= interest_windows[0]) &
                                         (peak_index_samples <= interest_windows[1]))
        # 获取感兴趣时间窗内最大peak特征值
        features_interest_numbers = peak_index_samples[0].unique(return_counts=True)[1]
        features_interest = pad_sequence(features).t()[peak_index_samples]
        features_interest = pad_sequence(features_interest.split(features_interest_numbers.tolist()), padding_value=0)
        features_interest_max[temp] = features_interest.amax(dim=0)
        self.del_var()

        return features_interest_max

    def cal_hfo_ied_features(self, hfo_events, vs_hfo_filter, vs_hfo_raw, vs_ied_filter, vs_ied_raw,
                             baseline_window_samples_hfo, ied_window=0.1, hfo_window=0.25,
                             hfo_frequency_range=(80, 200), ied_frequency_range=(3, 80), tf_bw_threshold=0.5):
        """
        Description:
            计算hfo event内的hfo segment，时域、频域、时频域特征

        Input:
            :param hfo_events:list, [torch.tensor, double, shape(n_segments*9)], shape(n_events), in_device
                每个hfo event的hfo segments，[ied index, hfo segments index, vs channel,
                                            sample index in 100 ms hfo segments(begin),
                                            sample index in 100 ms hfo segments(end),
                                            sample index in hfo segments(begin),
                                            sample index in hfo segments(end),
                                            sample index in raw data(begin),
                                            sample index in raw data(end),]
            :param vs_hfo_filter: list, [torch.tensor, double, shape(n_segments*n_samples)], shape(n_events), in_device
                每个hfo event的hfo segments，HFO最优方向的源空间信号(HFO频段滤波)
            :param vs_hfo_raw: list, [torch.tensor, double, shape(n_segments*n_samples)], shape(n_events), in_device
                每个hfo event的hfo segments，HFO最优方向的源空间信号(未滤波)
            :param vs_ied_filter: list, [torch.tensor, double, shape(n_segments*n_samples)], shape(n_events), in_device
                每个hfo event的hfo segments，IED最优方向的源空间信号(IED频段滤波)
            :param vs_ied_raw: list, [torch.tensor, double, shape(n_segments*n_samples)], shape(n_events), in_device
                每个hfo event的hfo segments，IED最优方向的源空间信号(未滤波)
            :param baseline_window_samples_hfo: torch.tensor, double, shape(n_segments*n_samples), in_device
                hfo segments，对应的baseline的sample index
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
            :return hfo_hilbert_amplitude: list, [torch.tensor, double, shape(n_segments)], shape(n_events), in_device
                每个hfo event的hfo segments，hfo平均包络幅度
            :return hfo_line_length: list, [torch.tensor, double, shape(n_segments)], shape(n_events), in_device
                每个hfo event的hfo segments，hfo的line length
            :return hfo_mean_power: list, [torch.tensor, double, shape(n_segments)], shape(n_events), in_device
                每个hfo event的hfo segments，hfo的平均能量
            :return hfo_std: list, [torch.tensor, double, shape(n_segments)], shape(n_events), in_device
                每个hfo event的hfo segments，hfo的标准差
            :return hfo_center_frequency: list, [torch.tensor, double, shape(n_segments)], shape(n_events), in_device
                每个hfo event的hfo segments，hfo的中心频率
            :return hfo_tf_area: list, [torch.tensor, double, shape(n_segments)], shape(n_events), in_device
                每个hfo event的hfo segments，hfo的时频图"孤岛"面积
            :return hfo_tf_frequency_range: list, [torch.tensor, double, shape(n_segments)], shape(n_events), in_device
                每个hfo event的hfo segments，hfo的时频图"孤岛"频率范围
            :return hfo_tf_entropy: list, [torch.tensor, double, shape(n_segments)], shape(n_events), in_device
                每个hfo event的hfo segments，hfo的时频图熵
            :return ied_line_length: list, [torch.tensor, double, shape(n_segments)], shape(n_events), in_device
                每个hfo event的hfo segments，ied的line length
            :return ied_mean_power: list, [torch.tensor, double, shape(n_segments)], shape(n_events), in_device
                每个hfo event的hfo segments，ied的平均能量
            :return ied_std: list, [torch.tensor, double, shape(n_segments)], shape(n_events), in_device
                每个hfo event的hfo segments，ied的标准差
            :return ied_peak2peak_amplitude: list, [torch.tensor, double, shape(n_segments)], shape(n_events), in_device
                每个hfo event的hfo segments，ied的最大峰峰值
            :return ied_peak_amplitude: list, [torch.tensor, double, shape(n_segments)], shape(n_events), in_device
                每个hfo event的hfo segments，ied的最大peak幅度
            :return ied_half_peak_slope: list, [torch.tensor, double, shape(n_segments)], shape(n_events), in_device
                每个hfo event的hfo segments，ied的最大peak半高宽斜率
            :return ied_bad_power_middle_to_low: list, [torch.tensor, double, shape(n_segments)], shape(n_events), in_device
                每个hfo event的hfo segments，ied的中间频带和低频带能量比
            :return ied_bad_power_high_to_low: list, [torch.tensor, double, shape(n_segments)], shape(n_events), in_device
                每个hfo event的hfo segments，ied的高频带和低频带能量比
            :return ied_tf_entropy: list, [torch.tensor, double, shape(n_segments)], shape(n_events), in_device
                每个hfo event的hfo segments，ied的时频图熵
        """

        if len(hfo_events) == 0:
            return [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

        # 计算时域特征
        hfo_hilbert_amplitude, hfo_line_length, hfo_mean_power, hfo_std, \
        ied_line_length, ied_mean_power, ied_std, ied_peak2peak_amplitude, ied_peak_amplitude, \
        ied_half_peak_slope = self.cal_time_domain_feature(
            hfo_events=hfo_events, vs_hfo_filter=vs_hfo_filter, vs_ied_filter=vs_ied_filter,
            baseline_window_samples_hfo=baseline_window_samples_hfo, ied_window=ied_window)
        self.del_var()

        # 计算频域特征
        ied_bad_power_middle_to_low, ied_bad_power_high_to_low = self.cal_frequency_domain_feature(
            vs_ied_filter=vs_ied_filter, ied_window=ied_window)
        self.del_var()

        # 计算时频域特征
        hfo_center_frequency, hfo_tf_area, hfo_tf_frequency_range, hfo_tf_entropy, ied_tf_entropy = \
            self.cal_time_frequency_domain_feature(
                hfo_events=hfo_events, vs_hfo_raw=vs_hfo_raw, vs_ied_raw=vs_ied_raw, hfo_window=hfo_window,
                hfo_frequency_range=hfo_frequency_range, ied_window=ied_window, ied_frequency_range=ied_frequency_range,
                tf_bw_threshold=tf_bw_threshold)
        self.del_var()

        return hfo_hilbert_amplitude, hfo_line_length, hfo_mean_power, hfo_std, hfo_center_frequency, hfo_tf_area, \
               hfo_tf_frequency_range, hfo_tf_entropy, \
               ied_line_length, ied_mean_power, ied_std, ied_peak2peak_amplitude, ied_peak_amplitude, \
               ied_half_peak_slope, ied_bad_power_middle_to_low, ied_bad_power_high_to_low, ied_tf_entropy

    def cal_time_domain_feature(self, hfo_events, vs_hfo_filter, vs_ied_filter, baseline_window_samples_hfo,
                                ied_window=0.1):
        """
        Description:
            计算hfo event内的hfo segment，时域特征

        Input:
            :param hfo_events:list, [torch.tensor, double, shape(n_segments*9)], shape(n_events), in_device
                每个hfo event的hfo segments，[ied index, hfo segments index, vs channel,
                                            sample index in 100 ms hfo segments(begin),
                                            sample index in 100 ms hfo segments(end),
                                            sample index in hfo segments(begin),
                                            sample index in hfo segments(end),
                                            sample index in raw data(begin),
                                            sample index in raw data(end),]
            :param vs_hfo_filter: list, [torch.tensor, double, shape(n_segments*n_samples)], shape(n_events), in_device
                每个hfo event的hfo segments，HFO最优方向的源空间信号(HFO频段滤波)
            :param vs_ied_filter: list, [torch.tensor, double, shape(n_segments*n_samples)], shape(n_events), in_device
                每个hfo event的hfo segments，IED最优方向的源空间信号(IED频段滤波)
            :param baseline_window_samples_hfo: torch.tensor, double, shape(n_events*n_segments*n_samples), in_device
                每个hfo event的hfo segments，对应的baseline的sample index
            :param ied_window: number, double
                ide的感兴趣时间窗长度，单位S
        Return:
            :return hfo_hilbert_amplitude: list, [torch.tensor, double, shape(n_segments)], shape(n_events), in_device
                每个hfo event的hfo segments，hfo平均包络幅度
            :return hfo_line_length: list, [torch.tensor, double, shape(n_segments)], shape(n_events), in_device
                每个hfo event的hfo segments，hfo的line length
            :return hfo_mean_power: list, [torch.tensor, double, shape(n_segments)], shape(n_events), in_device
                每个hfo event的hfo segments，hfo的平均能量
            :return hfo_std: list, [torch.tensor, double, shape(n_segments)], shape(n_events), in_device
                每个hfo event的hfo segments，hfo的标准差
            :return ied_line_length: list, [torch.tensor, double, shape(n_segments)], shape(n_events), in_device
                每个hfo event的hfo segments，ied的line length
            :return ied_mean_power: list, [torch.tensor, double, shape(n_segments)], shape(n_events), in_device
                每个hfo event的hfo segments，ied的平均能量
            :return ied_std: list, [torch.tensor, double, shape(n_segments)], shape(n_events), in_device
                每个hfo event的hfo segments，ied的标准差
            :return ied_peak2peak_amplitude: list, [torch.tensor, double, shape(n_segments)], shape(n_events), in_device
                每个hfo event的hfo segments，ied的最大峰峰值
            :return ied_peak_amplitude: list, [torch.tensor, double, shape(n_segments)], shape(n_events), in_device
                每个hfo event的hfo segments，ied的最大peak幅度
            :return ied_half_peak_slope: list, [torch.tensor, double, shape(n_segments)], shape(n_events), in_device
                每个hfo event的hfo segments，ied的最大peak半高宽斜率
        """

        with torch.no_grad():
            # step0: 准备数据
            ied_window = int(ied_window * self.data_info['sfreq'])
            segments_in_events = [x.shape[0] for x in hfo_events]
            hfo_segments = torch.cat(hfo_events)
            # 确定ied的baseline
            baseline_window_samples_ied = \
                torch.cat([torch.arange(baseline_window_samples_hfo[0][0],
                                        int((vs_ied_filter[0].shape[1] - ied_window) / 2)),
                           torch.arange(int((vs_ied_filter[0].shape[1] + ied_window) / 2),
                                        baseline_window_samples_hfo[0][-1])]).unsqueeze(0)
            # hfo频段数据，zscore之后结果
            vs_hfo_filter_zscore = \
                self.zscore_data(data=torch.cat(vs_hfo_filter),
                                 baseline_samples=baseline_window_samples_hfo[hfo_segments[:, 1]])
            # 每段hfo segments，zscore之后结果
            vs_hfo_segments_zscore = [y[x[5]:x[6]] for x, y in zip(hfo_segments, vs_hfo_filter_zscore)]
            # hfo频段数据包络，zscore之后结果
            vs_hfo_hilbert_zscore = \
                self.zscore_data(data=self.hilbert(torch.cat(vs_hfo_filter)),
                                 baseline_samples=baseline_window_samples_hfo[hfo_segments[:, 1]])
            # ied频段数据，zscore之后结果
            vs_ied_filter_zscore = self.zscore_data(data=torch.cat(vs_ied_filter),
                                                    baseline_samples=baseline_window_samples_ied)

            # step1: 计算hfo频段数据的振荡个数
            temp = pad_sequence(vs_hfo_segments_zscore, padding_value=np.nan)
            _, hfo_peak_numbers = self.peak_detect(temp.t())

            # Step 2: 计算hfo频段的hilbert能量
            hfo_hilbert_amplitude = torch.tensor([y[x[5]:x[6]].mean()
                                                  for x, y in zip(hfo_segments, vs_hfo_hilbert_zscore)])

            # step3: 计算hfo频段的line length, 平均能量和标准差
            hfo_line_length = torch.tensor([(x[1:] - x[:-1]).abs().mean(dim=-1) for x in vs_hfo_segments_zscore])
            hfo_mean_power = torch.tensor([x.pow(2).mean() for x in vs_hfo_segments_zscore])
            hfo_std = torch.tensor([x.std() for x in vs_hfo_segments_zscore])

            # step4: 计算ied频段的line length, 平均能量和标准差
            # 计算ied主峰周围最大line length的ied_window_time长度数据
            cut_dat_conv = torch.nn.Unfold(kernel_size=(1, ied_window)).to(self.device)
            vs_ied_filter_zscore_temp = vs_ied_filter_zscore[:, int(vs_ied_filter_zscore.shape[-1] / 2) +
                                                                torch.arange(-ied_window, ied_window)]
            ied_segments = cut_dat_conv(vs_ied_filter_zscore_temp.unsqueeze(1).unsqueeze(1))
            ied_segments = (ied_segments[:, 1:] - ied_segments[:, :-1]).abs().mean(dim=1).argmax(dim=-1)
            ied_segments = (ied_segments + torch.arange(0, ied_window).unsqueeze(1).to(self.device)).t()
            vs_ied_segments_zscore = vs_ied_filter_zscore.take_along_dim(ied_segments, dim=-1)
            # 计算line length, 平均能量和标准差
            ied_line_length = torch.tensor([(x[1:] - x[:-1]).abs().mean(dim=-1) for x in vs_ied_segments_zscore])
            ied_mean_power = vs_ied_segments_zscore.pow(2).mean(dim=-1)
            ied_std = vs_ied_segments_zscore.std(dim=-1)

            # step5: 计算ied频段的最大峰峰值(30ms 时间窗)
            vs_ied_filter_zscore_temp = vs_ied_filter_zscore[:, (vs_ied_filter_zscore.shape[-1] / 2 + torch.arange(
                -ied_window / 2, ied_window / 2)).long()]
            cut_dat_conv = torch.nn.Unfold(kernel_size=(1, int(0.03 * self.data_info['sfreq']))).to(self.device)
            vs_ied_segments_zscore = cut_dat_conv(vs_ied_filter_zscore_temp.unsqueeze(1).unsqueeze(1))
            ied_peak2peak_amplitude = (vs_ied_segments_zscore.amax(dim=1) - vs_ied_segments_zscore.amin(dim=1)).amax(-1)

            # Step 6: 计算ied频段的最大peak幅度和peak半高宽斜率
            interest_ied_windows = [int(vs_ied_filter_zscore.shape[-1] / 2 - ied_window / 2),
                                    int(vs_ied_filter_zscore.shape[-1] / 2 + ied_window / 2)]
            peak = ied_peak_feature.cal_peak_feature(raw_data=vs_ied_filter_zscore.cpu().numpy(),
                                                     data_info=self.get_data_info(), device=self.device_number)
            signal_peaks = peak.cal_signal_peaks(smooth_windows=0.02, smooth_iterations=2)
            # 计算ied_window_time内的peak index
            peak_index = peak.cal_peak_index(pos_peak_idx=signal_peaks['pos_peak_idx'],
                                             neg_peak_idx=signal_peaks['neg_peak_idx'])
            # 计算ied频段的最大peak幅度
            peak_amplitude = peak.cal_peak_amplitude(pos_peak_data=signal_peaks['pos_peak_data'],
                                                     neg_peak_data=signal_peaks['neg_peak_data'])
            ied_peak_amplitude = self.cal_max_peak_feature(peak_index=peak_index, features=peak_amplitude,
                                                           interest_windows=interest_ied_windows)
            # 计算ied频段的peak半高宽斜率
            half_peak_slope = peak.cal_half_peak_slope(pos_peak_data=signal_peaks['pos_peak_data'],
                                                       neg_peak_data=signal_peaks['neg_peak_data'],
                                                       pos_half_peak_idx_num=signal_peaks['pos_half_peak_idx_num'],
                                                       neg_half_peak_idx_num=signal_peaks['neg_half_peak_idx_num'])
            ied_half_peak_slope = self.cal_max_peak_feature(peak_index=peak_index, features=half_peak_slope,
                                                            interest_windows=interest_ied_windows)

        # 输出，将输出值放入cpu中, 并转换为event形式
        hfo_hilbert_amplitude, hfo_line_length, hfo_mean_power, hfo_std = \
            hfo_hilbert_amplitude.cpu().split(segments_in_events), hfo_line_length.cpu().split(segments_in_events), \
            hfo_mean_power.cpu().split(segments_in_events), hfo_std.cpu().split(segments_in_events)
        ied_line_length, ied_mean_power, ied_std, ied_peak2peak_amplitude, ied_peak_amplitude, ied_half_peak_slope = \
            ied_line_length.cpu().split(segments_in_events), ied_mean_power.cpu().split(segments_in_events), \
            ied_std.cpu().split(segments_in_events), ied_peak2peak_amplitude.cpu().split(segments_in_events), \
            ied_peak_amplitude.cpu().split(segments_in_events), ied_half_peak_slope.cpu().split(segments_in_events)
        self.del_var()

        return hfo_hilbert_amplitude, hfo_line_length, hfo_mean_power, hfo_std, \
               ied_line_length, ied_mean_power, ied_std, ied_peak2peak_amplitude, ied_peak_amplitude, \
               ied_half_peak_slope

    def cal_frequency_domain_feature(self, vs_ied_filter, ied_window=0.1):
        """
        Description:
            对于每个hfo segment，计算vs信号的频域特征:
                (1). ied_bad_power_middle_to_low: ied信号的中间频段和低频段能量比值。
                (2). ied_bad_power_high_to_low: ied信号的高频段和低频段能量比值。

        Input:
            :param vs_ied_filter: list, [torch.tensor, double, shape(n_segments*n_samples)], shape(n_events), in_device
                每个hfo事件的hfo segments, IED最优方向的源空间信号(IED频段滤波)
            :param ied_window: number, double
                ied segments的时间长度，单位S
        Return:
            :return ied_bad_power_middle_to_low: list, [torch.tensor, double, shape(n_segments)], shape(n_events)
                ied信号的中间频段和低频段能量比值
            :return ied_bad_power_high_to_low: list, [torch.tensor, double, shape(n_segments)], shape(n_events)
                ied信号的高频段和低频段能量比值
        """

        with torch.no_grad():
            # 准备数据
            ied_window = int(ied_window * self.data_info['sfreq'])
            segments_in_events = [x.shape[0] for x in vs_ied_filter]
            vs_ied = torch.cat(vs_ied_filter)

            # 获取ied segments信号
            interest_ied_windows = [int(vs_ied.shape[-1] / 2 - ied_window / 2),
                                    int(vs_ied.shape[-1] / 2 + ied_window / 2)]
            vs_ied_segment = vs_ied[:, interest_ied_windows[0]:interest_ied_windows[1]]

            # 计算ied频段的子段的power
            freq_samples = torch.fft.fftfreq(vs_ied_segment.shape[-1], 1 / self.get_data_info()['sfreq'])
            fft_ied_segment = torch.fft.fft(vs_ied_segment, dim=-1).abs()
            ied_bad_power = torch.stack(
                (fft_ied_segment[:, torch.where((freq_samples > 3) & (freq_samples <= 15))[0]].mean(dim=-1),
                 fft_ied_segment[:, torch.where((freq_samples > 15) & (freq_samples <= 30))[0]].mean(dim=-1),
                 fft_ied_segment[:, torch.where((freq_samples > 30) & (freq_samples <= 80))[0]].mean(dim=-1)), dim=1)

            # 计算子段power的比值
            ied_bad_power_middle_to_low = (ied_bad_power[:, 1] / ied_bad_power[:, 0]).cpu().split(segments_in_events)
            ied_bad_power_high_to_low = (ied_bad_power[:, 2] / ied_bad_power[:, 0]).cpu().split(segments_in_events)
            self.del_var()

        return ied_bad_power_middle_to_low, ied_bad_power_high_to_low

    def cal_time_frequency_domain_feature(self, hfo_events, vs_hfo_raw, vs_ied_raw,
                                          hfo_window=0.25, hfo_frequency_range=(80, 200),
                                          ied_window=0.1, ied_frequency_range=(3, 80), tf_bw_threshold=0.5):
        """
        Description:
            计算hfo event内的hfo segment，时频域特征

        Input:
            :param hfo_events:list, [torch.tensor, double, shape(n_segments*9)], shape(n_events), in_device
                每个hfo event的hfo segments，[ied index, hfo segments index, vs channel,
                                            sample index in 100 ms hfo segments(begin),
                                            sample index in 100 ms hfo segments(end),
                                            sample index in hfo segments(begin),
                                            sample index in hfo segments(end),
                                            sample index in raw data(begin),
                                            sample index in raw data(end),]
            :param vs_hfo_raw: list, [torch.tensor, double, shape(n_segments*n_samples)], shape(n_events), in_device
                每个hfo event的hfo segments，HFO最优方向的源空间信号(未滤波)
            :param vs_ied_raw: list, [torch.tensor, double, shape(n_segments*n_samples)], shape(n_events), in_device
                每个hfo event的hfo segments，IED最优方向的源空间信号(未滤波)
            :param hfo_window: number, double
                hfo的感兴趣时间窗长度，用于计算时频图，单位S
            :param ied_window: number, double
                ied segments的时间长度，单位S
            :param hfo_frequency_range: list, double, shape(2)
                hfo的起始和结束频率
            :param ied_frequency_range: list, double, shape(2)
                ied的起始和结束频率
            :param tf_bw_threshold: number, double
                用于二值化时频图的阈值，最大值的倍数
        Return:
            :return hfo_center_frequency: list, [torch.tensor, double, shape(n_segments)], shape(n_events), in_device
                每个hfo event的hfo segments，hfo的中心频率
            :return hfo_tf_area: list, [torch.tensor, double, shape(n_segments)], shape(n_events), in_device
                每个hfo event的hfo segments，hfo的时频图"孤岛"面积
            :return hfo_tf_frequency_range: list, [torch.tensor, double, shape(n_segments)], shape(n_events), in_device
                每个hfo event的hfo segments，hfo的时频图"孤岛"频率范围
            :return hfo_tf_entropy: list, [torch.tensor, double, shape(n_segments)], shape(n_events), in_device
                每个hfo event的hfo segments，hfo的时频图熵
            :return ied_tf_entropy: list, [torch.tensor, double, shape(n_segments)], shape(n_events), in_device
                每个hfo event的hfo segments，ied的时频图熵
        """

        with torch.no_grad():
            # step0: 准备数据
            hfo_segments = torch.cat(hfo_events)
            segments_in_events = [x.shape[0] for x in vs_hfo_raw]
            vs_hfo, vs_ied = torch.cat(vs_hfo_raw), torch.cat(vs_ied_raw)
            # 计算hfo时频图
            # 扩展hfo windows
            extend_hfo_window = (hfo_segments[:, 5:7] - vs_hfo.shape[-1] / 2).abs().max() * 2 / self.data_info[
                'sfreq'] + \
                                0.05
            hfo_window_temp = extend_hfo_window + hfo_window
            # 计算hfo时频图
            tf_hfo_segments = \
                self.cal_time_frequency_maps_windows(data=vs_hfo, sfreq=self.data_info['sfreq'],
                                                     frequency_range=(1., hfo_frequency_range[1]),
                                                     method=0, extend_window=0.15, tf_window=hfo_window_temp)
            # 裁剪hfo segments为中心，hfo_window长度的视频图
            samples_temp = (hfo_segments[:, 5:7].float().mean(dim=-1) - vs_hfo.shape[-1] / 2 +
                            hfo_window_temp / 2 * self.data_info['sfreq']).long()
            samples_temp = samples_temp + \
                           torch.arange(-int(hfo_window * self.data_info['sfreq'] / 2),
                                        int(hfo_window * self.data_info['sfreq'] / 2)).unsqueeze(1).to(self.device)
            tf_hfo_segments = tf_hfo_segments.take_along_dim(indices=samples_temp.t().unsqueeze(-1), dim=1)
            # 计算ied时频图
            tf_ied_segments = \
                self.cal_time_frequency_maps_windows(data=vs_ied, sfreq=self.data_info['sfreq'],
                                                     frequency_range=(1., ied_frequency_range[1]),
                                                     method=0, extend_window=0.15, tf_window=ied_window)

            # step1: 计算hfo的中心频率
            # 获取hfo segments中心周围数据
            hfo_center_frequency = tf_hfo_segments[:,
                                   torch.arange(int((hfo_window / 2 - 0.05) * self.data_info['sfreq']),
                                                int((hfo_window / 2 + 0.05) * self.data_info['sfreq']))]
            hfo_center_time = hfo_center_frequency[:, :, (hfo_frequency_range[0] - 1):].amax(dim=-1).argmax(dim=1) + \
                              int((hfo_window / 2 - 0.05) * self.data_info['sfreq'])
            hfo_center_frequency = hfo_center_frequency[:, :, (hfo_frequency_range[0] - 1):].amax(dim=1).argmax(dim=-1) \
                                   + hfo_frequency_range[0]

            # step2: 计算hfo的面积和频率范围
            # 计算小于tf_bw_threshold倍最大值的tf
            tf_hfo_segments_bw = tf_hfo_segments[:, :, (hfo_frequency_range[0] - 1):]
            tf_hfo_segments_bw = tf_hfo_segments_bw >= \
                                 tf_bw_threshold * tf_hfo_segments_bw.amax(dim=(-1, -2), keepdim=True)
            # 计算hfo"孤岛"
            tf_hfo_segments_label = torch.tensor([measure.label(x.cpu()) for x in tf_hfo_segments_bw])
            tf_hfo_segments_label = [torch.stack(torch.where(x == x[y[0], y[1] - hfo_frequency_range[0]])) for x, y in
                                     zip(tf_hfo_segments_label,
                                         torch.stack([hfo_center_time, hfo_center_frequency]).t())]
            hfo_tf_area = torch.tensor([x.shape[1] for x in tf_hfo_segments_label])
            hfo_tf_frequency_range = torch.tensor([x[1].max() - x[1].min() for x in tf_hfo_segments_label])

            # step3: 计算hfo的时频图熵
            hfo_tf_entropy = tf_hfo_segments[:, :, (hfo_frequency_range[0] - 1):].sum(dim=-1)
            hfo_tf_entropy = (hfo_tf_entropy - hfo_tf_entropy.amin(dim=-1, keepdim=True)) / \
                             hfo_tf_entropy.sum(dim=-1, keepdim=True)
            hfo_tf_entropy = -(hfo_tf_entropy.pow(2) * (hfo_tf_entropy.pow(2) + 2.2204e-16).log()).sum(dim=-1)

            # step4: 计算ied的时频图熵
            ied_tf_entropy = tf_ied_segments[:, :, (ied_frequency_range[0] - 1):].sum(dim=-1)
            ied_tf_entropy = (ied_tf_entropy - ied_tf_entropy.amin(dim=-1, keepdim=True)) / \
                             ied_tf_entropy.sum(dim=-1, keepdim=True)
            ied_tf_entropy = -(ied_tf_entropy.pow(2) * (ied_tf_entropy.pow(2) + 2.2204e-16).log()).sum(dim=-1)

            # 输出，将输出值放入cpu中, 并转换为event形式
            hfo_center_frequency, hfo_tf_area, hfo_tf_frequency_range, hfo_tf_entropy = \
                hfo_center_frequency.cpu().split(segments_in_events), hfo_tf_area.cpu().split(segments_in_events), \
                hfo_tf_frequency_range.cpu().split(segments_in_events), hfo_tf_entropy.cpu().split(segments_in_events)
            ied_tf_entropy = ied_tf_entropy.cpu().split(segments_in_events)
            self.del_var()

        return hfo_center_frequency, hfo_tf_area, hfo_tf_frequency_range, hfo_tf_entropy, ied_tf_entropy

    def cal_export_for_emhapp(self, ieds_segment_samples, hfo_events, inv_covariance_hfo, inv_covariance_ied,
                              oris_hfo, oris_ied):
        """
        Description:
            根据hfo_events，计算用于EMHapp的输出。

        Input:
            :param ieds_segment_samples: np.array, double, shape(n_ieds*n_samples)
                ied片段在原始数据中的采样点位置
            :param oris_hfo: np.array, double, shape(n_ieds*n_segments*n_vs_channels*3)
                hfo最优源空间信号方向
            :param oris_ied: np.array, double, shape(n_ieds*n_segments*n_vs_channels*3)
                ied最优源空间信号方向
            :param inv_covariance_hfo: np.array, double, shape(n_channels*n_channels)
                hfo频段的协方差矩阵逆矩阵
            :param inv_covariance_ied: np.array, double, shape(n_channels*n_channels)
                ied频段的协方差矩阵逆矩阵
            :param hfo_events:list, [torch.tensor, double, shape(n_segments*9)], shape(n_events), in_device
                每个hfo event的hfo segments，[ied index, hfo segments index, vs channel,
                                            sample index in 100 ms hfo segments(begin),
                                            sample index in 100 ms hfo segments(end),
                                            sample index in hfo segments(begin),
                                            sample index in hfo segments(end),
                                            sample index in raw data(begin),
                                            sample index in raw data(end),]
        Return:
            :return hfo_events_time: list, [np.array, long, shape(n_hfo_channels, 4)], shape(n_events)
                每个hfo event的hfo segments, [ied index, vs channel, sample index in hfo segments(begin),
                                             sample index in hfo segments(end)]
            :return hfo_window_samples: list, [np.array, double, shape(n_hfo_channels, n_samples)], shape(n_events)
                每个hfo event对应的数据片段在原始数据中的采样点
            :return hfo_events_weight_hfo: list, [np.array, double, shape(n_hfo_channels, n_channels)], shape(n_events)
                每个hfo event的beamformer weight(hfo频段)
            :return hfo_events_weight_ied: list, [torch.tensor, double, shape(n_hfo_channels, n_samples)], shape(n_events)
                每个hfo event的beamformer weight(ied频段)
        """

        # step1: 计算检测到hfo的时间
        # 对于每个hfo event，可能对应多个ied，将出现最多的ied作为hfo event的ied
        ied_index = [x[:, 0].unique()[x[:, 0].unique(return_counts=True)[-1].argmax()].long() for x in hfo_events]
        # 计算hfo event对应的ied片段在原始数据中的采样点
        hfo_window_samples = [torch.tensor(ieds_segment_samples)[x].unsqueeze(0).repeat(y.shape[0], 1).to(self.device)
                              for x, y in zip(ied_index, hfo_events)]
        # 计算hfo events的每个vs channel，在ied segment中的位置
        hfo_samples = [torch.stack([torch.tensor([torch.where(q == p[7])[0][0], torch.where(q == p[8])[0][0]])
                                    for p, q in zip(x, y)]) for x, y in zip(hfo_events, hfo_window_samples)]
        # 计算hfo event时间，[ied index, vs channel, hfo begin, hfo end]
        hfo_events_time = [torch.cat([x[:, [0, 2]].cpu(), y], dim=1).long().cpu().numpy()
                           for x, y in zip(hfo_events, hfo_samples)]
        hfo_window_samples = [x.cpu().numpy() for x in hfo_window_samples]

        # step2: 计算hfo以及对应ied的beamformer weight
        hfo_events_weight_hfo, hfo_events_weight_ied = [], []
        leadfield = self.get_leadfield().clone().to(self.device)
        oris_hfo, oris_ied = torch.tensor(oris_hfo), torch.tensor(oris_ied)
        inv_covariance_hfo = torch.tensor(inv_covariance_hfo).to(self.device)
        inv_covariance_ied = torch.tensor(inv_covariance_ied).to(self.device)
        for event in hfo_events:
            leadfield_temp = torch.stack([leadfield[x[2]] for x in event])
            # 计算hfo频段
            ori_hfo = torch.stack([oris_hfo[x[0]][x[1]][x[2]] for x in event]).unsqueeze(0).to(self.device)
            weight, _ = self.cal_vs_signal(leadfield=leadfield_temp, source_ori=ori_hfo,
                                           inv_covariance=inv_covariance_hfo, meg_data=None)
            hfo_events_weight_hfo.append(weight[0].cpu().numpy())
            # 计算ied频段
            ori_ied = torch.stack([oris_ied[x[0]][0][x[2]] for x in event]).unsqueeze(0).to(self.device)
            weight, _ = self.cal_vs_signal(leadfield=leadfield_temp, source_ori=ori_ied,
                                           inv_covariance=inv_covariance_ied, meg_data=None)
            hfo_events_weight_ied.append(weight[0].cpu().numpy())
        self.del_var()

        return hfo_events_time, hfo_window_samples, hfo_events_weight_hfo, hfo_events_weight_ied

    def cal_hfo_source_maps(self, hfo_events, ieds_segment_samples, inv_covariance_hfo, oris_hfo,
                            hfo_center_frequency, frequency_range=50):
        """
        Description:
            计算hfo的激活图，对每一个vs channel:
                (1). 重建VS信号
                (2). 根据中心频率和frequency_range对vs信号进行滤波，计算包络信号
                (3). 计算滤波后信号的包络
                (4). 使用baseline信号对数据计算zscore，相对于baseline的幅度
                (5). hfo时间窗的平均包络信号为激活值


        Input:
            :param hfo_events:list, [torch.tensor, double, shape(n_segments*9)], shape(n_events), in_device
                每个hfo event的hfo segments，[ied index, hfo segments index, vs channel,
                                            sample index in 100 ms hfo segments(begin),
                                            sample index in 100 ms hfo segments(end),
                                            sample index in hfo segments(begin),
                                            sample index in hfo segments(end),
                                            sample index in raw data(begin),
                                            sample index in raw data(end),]
            :param ieds_segment_samples: np.array, double, shape(n_ieds*n_samples)
                ied片段在原始数据中的采样点位置
            :param oris_hfo: np.array, double, shape(n_ieds*n_segments*n_vs_channels*3)
                hfo最优源空间信号方向
            :param inv_covariance_hfo: np.array, double, shape(n_channels*n_channels)
                hfo频段的协方差矩阵逆矩阵
            :param hfo_center_frequency: list, [torch.tensor, double, shape(n_segments)], shape(n_events), in_device
                每个hfo event的hfo segments，hfo的中心频率
            :param frequency_range: number, double
                hfo滤波的带宽

        Return:
            :return hfo_source_maps: np.array, long, shape(n_events*n_vs_channels)
                每个hfo event的source maps
        """
        hfo_source_maps = []
        leadfield = self.get_leadfield().clone().to(self.device)
        inv_covariance_hfo = torch.tensor(inv_covariance_hfo).to(self.device)
        oris_hfo = torch.tensor(oris_hfo).to(self.device)
        for event, center_frequency in zip(hfo_events, hfo_center_frequency):
            # 获取channel数目最多的ied index和segment index
            temp = event[:, :2].unique(dim=0)[event[:, :2].unique(return_counts=True, dim=0)[1].argmax()]
            ied_index, segment_index = temp[0], temp[1]
            # 计算vs信号
            meg_data_raw = self.get_raw_data()[:, ieds_segment_samples[ied_index]].to(self.device)
            _, vs_signal = self.cal_vs_signal(leadfield=leadfield,
                                              source_ori=oris_hfo[ied_index][segment_index].unsqueeze(0),
                                              inv_covariance=inv_covariance_hfo, meg_data=meg_data_raw)
            # 滤波
            center_frequency = center_frequency.float().mean()
            frequency_band = [center_frequency - frequency_range / 2, center_frequency + frequency_range / 2]
            vs_signal = mne.filter.filter_data(vs_signal.cpu()[0], self.data_info['sfreq'],
                                               frequency_band[0], frequency_band[1],
                                               fir_design='firwin', pad="reflect_limited", verbose=False,
                                               n_jobs='cuda' if self.is_cuda else self.n_jobs)
            vs_signal = torch.tensor(vs_signal).to(self.device)
            # 计算vs信号的包络
            vs_signal_hilbert = self.hilbert(vs_signal)
            # 计算hfo的windows和对应的baseline
            cut_samples = int(0.05 * self.data_info['sfreq'])
            hfo_windows = event[(event[:, 0] == ied_index) & (event[:, 1] == segment_index), 5:7]
            hfo_windows = hfo_windows.float().mean(dim=0).long()
            baseline_windows = torch.cat([torch.arange(cut_samples, hfo_windows[0]),
                                          torch.arange(hfo_windows[1], vs_signal_hilbert.shape[-1] - cut_samples)])
            # 计算source maps
            vs_hilbert_zscore = (vs_signal_hilbert - vs_signal_hilbert[:, baseline_windows].mean(dim=-1,
                                                                                                 keepdim=True)) / \
                                vs_signal_hilbert[:, baseline_windows].std(dim=-1, keepdim=True)
            hfo_source_maps_temp = vs_hilbert_zscore[:, hfo_windows[0]:hfo_windows[1]].mean(dim=-1)
            hfo_source_maps.append(hfo_source_maps_temp)

        hfo_source_maps = torch.stack(hfo_source_maps).cpu().numpy()
        self.del_var()

        return hfo_source_maps

    def export_emhapp(self, ieds_segment_samples, hfo_events, inv_covariance_hfo, inv_covariance_ied,
                      oris_hfo, oris_ied, hfo_center_frequency, frequency_range=50):
        """
        Description:
            根据hfo_events，计算用于EMHapp的输出。

        Input:
            :param ieds_segment_samples: np.array, double, shape(n_ieds*n_samples)
                ied片段在原始数据中的采样点位置
            :param oris_hfo: np.array, double, shape(n_ieds*n_segments*n_vs_channels*3)
                hfo最优源空间信号方向
            :param oris_ied: np.array, double, shape(n_ieds*n_segments*n_vs_channels*3)
                ied最优源空间信号方向
            :param inv_covariance_hfo: np.array, double, shape(n_channels*n_channels)
                hfo频段的协方差矩阵逆矩阵
            :param inv_covariance_ied: np.array, double, shape(n_channels*n_channels)
                ied频段的协方差矩阵逆矩阵
            :param hfo_events:list, [torch.tensor, double, shape(n_segments*9)], shape(n_events), in_device
                每个hfo event的hfo segments，[ied index, hfo segments index, vs channel,
                                            sample index in 100 ms hfo segments(begin),
                                            sample index in 100 ms hfo segments(end),
                                            sample index in hfo segments(begin),
                                            sample index in hfo segments(end),
                                            sample index in raw data(begin),
                                            sample index in raw data(end),]
            :param hfo_center_frequency: list, [torch.tensor, double, shape(n_segments)], shape(n_events), in_device
                每个hfo event的hfo segments，hfo的中心频率
            :param frequency_range: number, double
                hfo滤波的带宽
        Return:
            :return emhapp_save: dict
                用于保存mat文件的字典
        """
        # 计算source maps
        hfo_source_maps = self.cal_hfo_source_maps(hfo_events=hfo_events, ieds_segment_samples=ieds_segment_samples,
                                                   inv_covariance_hfo=inv_covariance_hfo, oris_hfo=oris_hfo,
                                                   hfo_center_frequency=hfo_center_frequency,
                                                   frequency_range=frequency_range)

        # 计算EMHapp需要的格式输出(检测到的hfo)
        hfo_events_time, hfo_window_samples, hfo_events_weight_hfo, hfo_events_weight_ied = \
            self.cal_export_for_emhapp(ieds_segment_samples=ieds_segment_samples, hfo_events=hfo_events,
                                       inv_covariance_hfo=inv_covariance_hfo, inv_covariance_ied=inv_covariance_ied,
                                       oris_hfo=oris_hfo, oris_ied=oris_ied)
        hfo_segments_time = np.concatenate(hfo_events_time)
        hfo_segments_window_samples = np.concatenate(hfo_window_samples)
        self.del_var()

        # 计算EMHapp需要的格式输出(检测到hfo周围的vs channel)
        # 计算hfo周围的vs channel
        hfo_events_neighbor = []
        for x in hfo_events:
            temp = torch.ones(self.get_leadfield().shape[0]) < 0
            vs_channel_neighbor = (x[:, 2].repeat(4).cpu() + torch.tensor([[-2], [-1], [1], [2]])).unique()
            vs_channel_neighbor = vs_channel_neighbor[(vs_channel_neighbor > 0) & (vs_channel_neighbor < temp.shape[0])]
            temp[vs_channel_neighbor] = True
            temp[x[:, 2]] = False
            vs_channel_neighbor = torch.where(temp)[0]
            hfo_events_neighbor_temp = torch.cat([x[:1, :2].repeat(vs_channel_neighbor.shape[0], 1),
                                                  vs_channel_neighbor.unsqueeze(1).to(self.device),
                                                  x[:1, 3:].repeat(vs_channel_neighbor.shape[0], 1)], dim=1)
            hfo_events_neighbor.append(hfo_events_neighbor_temp)
        # 计算EMHapp需要的格式输出
        hfo_neighbor_time, hfo_neighbor_window_samples, hfo_neighbor_weight_hfo, hfo_neighbor_weight_ied = \
            self.cal_export_for_emhapp(ieds_segment_samples=ieds_segment_samples, hfo_events=hfo_events_neighbor,
                                       inv_covariance_hfo=inv_covariance_hfo, inv_covariance_ied=inv_covariance_ied,
                                       oris_hfo=oris_hfo, oris_ied=oris_ied)

        # 重构输出格式
        emhapp_save = \
            {'SourceMapsHFO': hfo_source_maps,
             # np.array(xxxx + [0], dtype=object)[:-1] 防止只有一个hfo event转换格式出错
             'SusHFO': hfo_segments_time + 1, 'SusHFO_BsTime': hfo_segments_window_samples + 1, 'SusHFO_Weight': [],
             'ArdEntHFO': np.array([x + 1 for x in hfo_neighbor_time] + [0], dtype=object)[:-1],
             'ArdEntHFO_BsTime': np.array([x + 1 for x in hfo_neighbor_window_samples] + [0], dtype=object)[:-1],
             'ArdEntHFO_Weight': np.array(hfo_neighbor_weight_hfo + [0], dtype=object)[:-1],
             'ArdEntHFO_Weight_S': np.array(hfo_neighbor_weight_ied + [0], dtype=object)[:-1],
             'EntHFO': np.array([x + 1 for x in hfo_events_time] + [0], dtype=object)[:-1],
             'EntHFO_BsTime': np.array([x + 1 for x in hfo_window_samples] + [0], dtype=object)[:-1],
             'EntHFO_Weight': np.array(hfo_events_weight_hfo + [0], dtype=object)[:-1],
             'EntHFO_Weight_S': np.array(hfo_events_weight_ied + [0], dtype=object)[:-1]}

        return emhapp_save

    def cal_hfo_and_features(self, ieds_segment_samples, max_oris_hfo, max_oris_ied,
                             inv_covariance_hfo, inv_covariance_ied,
                             hfo_window_samples, baseline_window_samples_hfo,
                             hfo_amplitude_threshold=2, hfo_duration_threshold=0.02, hfo_oscillation_threshold=4,
                             hfo_entropy_threshold=1.25, hfo_power_ratio_threshold=1.25,
                             hfo_frequency_range=(80, 200), ied_frequency_range=(3, 80),
                             ied_window=0.1, hfo_window=0.25, tf_bw_threshold=0.5):

        # 使用阈值方法计算hfo segments和hfo events
        hfo_events, vs_hfo_filter, vs_hfo_raw, vs_ied_filter, vs_ied_raw = self.cal_hfo_events_thresholding_method(
            ieds_segment_samples=ieds_segment_samples, max_oris_hfo=max_oris_hfo, max_oris_ied=max_oris_ied,
            inv_covariance_hfo=inv_covariance_hfo, inv_covariance_ied=inv_covariance_ied,
            hfo_window_samples=hfo_window_samples, baseline_window_samples_hfo=baseline_window_samples_hfo,
            amplitude_threshold=hfo_amplitude_threshold, duration_threshold=hfo_duration_threshold,
            oscillation_threshold=hfo_oscillation_threshold, entropy_threshold=hfo_entropy_threshold,
            power_ratio_threshold=hfo_power_ratio_threshold,
            frequency_range=hfo_frequency_range)
        self.del_var()

        # 计算hfo segments的特征
        hfo_hilbert_amplitude, hfo_line_length, hfo_mean_power, hfo_std, hfo_center_frequency, hfo_tf_area, \
        hfo_tf_frequency_range, hfo_tf_entropy, \
        ied_line_length, ied_mean_power, ied_std, ied_peak2peak_amplitude, ied_peak_amplitude, \
        ied_half_peak_slope, ied_bad_power_middle_to_low, ied_bad_power_high_to_low, ied_tf_entropy = \
            self.cal_hfo_ied_features(hfo_events=hfo_events, vs_hfo_filter=vs_hfo_filter, vs_hfo_raw=vs_hfo_raw,
                                      vs_ied_filter=vs_ied_filter, vs_ied_raw=vs_ied_raw,
                                      baseline_window_samples_hfo=torch.tensor(baseline_window_samples_hfo),
                                      ied_window=ied_window, hfo_window=hfo_window,
                                      hfo_frequency_range=hfo_frequency_range, ied_frequency_range=ied_frequency_range,
                                      tf_bw_threshold=tf_bw_threshold)
        self.del_var()

        # 计算EMHapp需要的格式输出
        if len(hfo_events) > 0:
            emhapp_save = self.export_emhapp(ieds_segment_samples=ieds_segment_samples, hfo_events=hfo_events,
                                             inv_covariance_hfo=inv_covariance_hfo,
                                             inv_covariance_ied=inv_covariance_ied,
                                             oris_hfo=max_oris_hfo, oris_ied=max_oris_ied,
                                             hfo_center_frequency=hfo_center_frequency, frequency_range=50)
            # 添加特征到EMHapp输出
            hfo_features = \
                {'LL_HFO': np.array([x.numpy() for x in hfo_line_length] + [0], dtype=object)[:-1],
                 'HilAmp_HFO': np.array([x.numpy() for x in hfo_hilbert_amplitude] + [0], dtype=object)[:-1],
                 'TFEntropy_HFO': np.array([x.numpy() for x in hfo_tf_entropy] + [0], dtype=object)[:-1],
                 'MeanPower_HFO': np.array([x.numpy() for x in hfo_mean_power] + [0], dtype=object)[:-1],
                 'Std_HFO': np.array([x.numpy() for x in hfo_std] + [0], dtype=object)[:-1],
                 'CenterFreq_HFO': np.array([x.numpy() for x in hfo_center_frequency] + [0], dtype=object)[:-1],
                 'TFArea_HFO': np.array([x.numpy() for x in hfo_tf_area] + [0], dtype=object)[:-1],
                 'TFFreqRange_HFO': np.array([x.numpy() for x in hfo_tf_frequency_range] + [0], dtype=object)[:-1],
                 'LL_Spike': np.array([x.numpy() for x in ied_line_length] + [0], dtype=object)[:-1],
                 'Peak2Peak_Spike': np.array([x.numpy() for x in ied_peak2peak_amplitude] + [0], dtype=object)[:-1],
                 'PeakAmp_Spike': np.array([x.numpy() for x in ied_peak_amplitude] + [0], dtype=object)[:-1],
                 'PeakSlope_Spike': np.array([x.numpy() for x in ied_half_peak_slope] + [0], dtype=object)[:-1],
                 'Mid2LowPower_Spike': np.array([x.numpy() for x in ied_bad_power_middle_to_low] + [0],
                                                dtype=object)[:-1],
                 'High2LowPower_Spike': np.array([x.numpy() for x in ied_bad_power_high_to_low] + [0],
                                                 dtype=object)[:-1],
                 'MeanPower_Spike': np.array([x.numpy() for x in ied_mean_power] + [0], dtype=object)[:-1],
                 'Std_Spike': np.array([x.numpy() for x in ied_std] + [0], dtype=object)[:-1],
                 'TFEntropy_Spike': np.array([x.numpy() for x in ied_tf_entropy] + [0], dtype=object)[:-1]}
            emhapp_save['Features'] = hfo_features
        else:
            emhapp_save = []

        self.del_var()

        return emhapp_save, hfo_events, \
               hfo_hilbert_amplitude, hfo_line_length, hfo_mean_power, hfo_std, hfo_center_frequency, hfo_tf_area, \
               hfo_tf_frequency_range, hfo_tf_entropy, \
               ied_line_length, ied_mean_power, ied_std, ied_peak2peak_amplitude, ied_peak_amplitude, \
               ied_half_peak_slope, ied_bad_power_middle_to_low, ied_bad_power_high_to_low, ied_tf_entropy


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

    RAW = mne.io.read_raw_fif(fifPath[0], verbose='error', preload=True)
    Info = RAW.info
    RAW.resample(1000, n_jobs='cuda')
    RAW_data = RAW.get_data(picks='meg')
    RAW_data_ied = mne.filter.filter_data(RAW_data, RAW.info['sfreq'], 3, 80,
                                          fir_design='firwin', pad="reflect_limited", verbose=False, n_jobs='cuda')
    RAW_data_ied = mne.filter.notch_filter(RAW_data_ied, RAW.info['sfreq'], 50., verbose=False, n_jobs='cuda')
    RAW_data_hfo = mne.filter.filter_data(RAW_data, RAW.info['sfreq'], 80, 200,
                                          fir_design='firwin', pad="reflect_limited", verbose=False, n_jobs='cuda')

    HFO = hfo_detection_threshold(raw_data=RAW_data, raw_data_ied=RAW_data_ied, raw_data_hfo=RAW_data_hfo,
                                  data_info=Info, leadfield=lf[0], device=5)

    HFO.cal_hfo_and_features(ieds_segment_samples=SpikeTime[0], max_oris_hfo=MaxOir[0], max_oris_ied=MaxOir_S[0],
                             inv_covariance_hfo=CovInv[0], inv_covariance_ied=CovInv_S[0],
                             hfo_window_samples=HFO_Time[0], baseline_window_samples_hfo=BS_Time[0],
                             hfo_amplitude_threshold=2, hfo_duration_threshold=0.02, hfo_oscillation_threshold=4,
                             hfo_entropy_threshold=1.25, hfo_power_ratio_threshold=1.25,
                             hfo_frequency_range=(80, 200), ied_frequency_range=(3, 80),
                             ied_window=0.1, hfo_window=0.25, tf_bw_threshold=0.5)
