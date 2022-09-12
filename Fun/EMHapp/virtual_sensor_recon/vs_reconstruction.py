# -*- coding:utf-8 -*-
# @Time    : 2022/2/4
# @Author  : cuiwei
# @File    : vs_reconstruction.py
# @Software: PyCharm
# @Script to:
#   - 使用非线性搜索重建最优源空间信号方向，以及源空间信号

import os
import mne
import numpy as np
import torch
import torch.fft
import scipy.io
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class vs_recon:
    def __init__(self, raw_data_ied=None, raw_data_hfo=None, data_info=None, leadfield=None, bad_segment=None,
                 device=-1, n_jobs=10):
        """
        Description:
            初始化，使用GPU或者CPU。

        Input:
            :param raw_data_ied: ndarray, double, shape(channel, samples)
                MEG滤波后数据(IED)
            :param raw_data_hfo: ndarray, double, shape(channel, samples)
                MEG滤波后数据(HFO)
            :param data_info: dict
                MEG数据的信息, MNE读取的raw.info
            :param leadfield: ndarray, double, shape(n_dipoles*3*n_channels)
                leadfield矩阵
            :param bad_segment: ndarray, double, shape(samples)
                坏片段
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
        self.raw_data_ied, self.raw_data_hfo, self.data_info, self.bad_segment, self.leadfield = \
            raw_data_ied, raw_data_hfo, data_info, bad_segment, leadfield
        self.set_raw_data_ied(raw_data_ied=raw_data_ied)
        self.set_raw_data_hfo(raw_data_hfo=raw_data_hfo)
        self.set_data_info(data_info=data_info)
        self.set_bad_segment(bad_segment=bad_segment)
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

    def set_bad_segment(self, bad_segment=None):
        if bad_segment is None:
            bad_segment = np.ones(self.set_raw_data_ied().shape[1]) < 0
        self.bad_segment = torch.tensor(bad_segment)

    def get_bad_segment(self):
        return self.bad_segment

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

    def peak_detect(self, data):
        """
        Description:
            计算data中的peak位置
            并存储为shape(n_batch*max_peaks_num_along_batch)形式，用-1补

        Input:
            :param data: torch.tensor, double, shape(n_batch*n_samples)
                输入数据
        Return:
            :return peaks: torch.tensor, long, shape(n_batch*max_peaks_num_along_batch), in_device
                peaks位置
        """
        with torch.no_grad():
            # 获取每个batch的peak位置
            data_diff = ((data[:, 1:] - data[:, :-1]) >= 0).float()
            data_diff = data_diff[:, 1:] - data_diff[:, :-1]
            peaks = torch.where(data_diff.abs() == 1)
            # 将peaks转换为shape(n_batch*max_peaks_num_along_batch)
            peaks_num_along_batch = torch.zeros(data.shape[0]).long().to(self.device)
            temp = peaks[0].unique_consecutive(return_counts=True)
            peaks_num_along_batch[temp[0]] = temp[1]
            peaks = pad_sequence((peaks[1] + 1).split(peaks_num_along_batch.tolist()), padding_value=-1,
                                 batch_first=True)

        return peaks

    def cal_data_covariance_using_whole_recording(self, raw_data_ied=None, raw_data_hfo=None, re_lambda=0.05):
        """
        Description:
            计算整段数据的协方差矩阵，和对应的逆矩阵.
            如果输入数据为None，则返回的协方差矩阵也为None.

        Input:
            :param raw_data_ied: torch.tensor, double, shape(n_channels*n_samples)
                ied滤波后的原始数据数据
            :param raw_data_hfo: torch.tensor, double, shape(n_channels*n_samples)
                hfo滤波后的原始数据数据
            :param re_lambda: number, double
                求违逆时候的正则化系数
        Return:
            :return cov_whole_ied: torch.tensor, double, shape(n_channels*n_channels), in_device
                ied频段数据的协方差矩阵
            :return inv_cov_whole_ied: torch.tensor, double, shape(n_channels*n_channels), in_device
                ied频段数据的协方差矩阵逆矩阵
            :return cov_whole_hfo: torch.tensor, double, shape(n_channels*n_channels), in_device
                hfo频段数据的协方差矩阵
            :return inv_cov_whole_hfo: torch.tensor, double, shape(n_channels*n_channels), in_device
                hfo频段数据的协方差矩阵逆矩阵
        """
        with torch.no_grad():
            # 计算ied频段的数据协方差矩阵，使用整段数据。
            if raw_data_ied is not None:
                raw_data_ied_temp = raw_data_ied[:, ~self.bad_segment].to(self.device)
                raw_data_ied_temp = raw_data_ied_temp - raw_data_ied_temp.mean(dim=1, keepdim=True)
                cov_whole_ied = torch.mm(raw_data_ied_temp, raw_data_ied_temp.t()) / \
                                raw_data_ied_temp.reshape(-1).shape[0]
                inv_cov_whole_ied = self.pinv(cov_whole_ied, re_lambda=re_lambda)
                self.del_var('raw_data_ied_temp')
            else:
                cov_whole_ied, inv_cov_whole_ied = None, None

            # 计算hfo频段的数据协方差矩阵，使用整段数据。
            if raw_data_hfo is not None:
                raw_data_hfo_temp = raw_data_hfo[:, ~self.bad_segment].to(self.device)
                raw_data_hfo_temp = raw_data_hfo_temp - raw_data_hfo_temp.mean(dim=1, keepdim=True)
                cov_whole_hfo = torch.mm(raw_data_hfo_temp, raw_data_hfo_temp.t()) / \
                                raw_data_hfo_temp.reshape(-1).shape[0]
                inv_cov_whole_hfo = self.pinv(cov_whole_hfo, re_lambda=re_lambda)
                self.del_var('raw_data_hfo_temp')
            else:
                cov_whole_hfo, inv_cov_whole_hfo = None, None

        return cov_whole_ied, inv_cov_whole_ied, cov_whole_hfo, inv_cov_whole_hfo

    def cal_noise_covariance(self, raw_data_ied=None, raw_data_hfo=None, re_lambda=0.05, noise_win=10,
                             ied_peak_sample=None):
        """
        Description:
            计算噪声数据的协方差矩阵，和对应的逆矩阵。噪声数据段定义为数据开始noise_win时间内的，良好数据段。
            如果输入数据为None，则返回的协方差矩阵也为None.

        Input:
            :param raw_data_ied: torch.tensor, double, shape(n_channels*n_samples)
                ied滤波后的原始数据数据
            :param raw_data_hfo: torch.tensor, double, shape(n_channels*n_samples)
                hfo滤波后的原始数据数据
            :param ied_peak_sample: np.array, double, shape(n_ieds)
                ied主peak的采样点
            :param re_lambda: number, double
                求伪逆时候的正则化系数
            :param noise_win: number, double
                噪声数据的时间段长度，单位s
        Return:
            :return cov_noise_ied: torch.tensor, double, shape(n_channels*n_channels), in_device
                ied频段噪声的协方差矩阵
            :return inv_cov_noise_ied: torch.tensor, double, shape(n_channels*n_channels), in_device
                ied频段噪声的协方差矩阵逆矩阵
            :return cov_noise_hfo: torch.tensor, double, shape(n_channels*n_channels), in_device
                hfo频段噪声的协方差矩阵
            :return inv_cov_noise_hfo: torch.tensor, double, shape(n_channels*n_channels), in_device
                hfo频段噪声的协方差矩阵逆矩阵
        """

        with torch.no_grad():
            # 计算ied频段的数据协方差矩阵，使用整段数据。
            if raw_data_ied is not None:
                # 删除IED片段
                sample_index = torch.where(~self.bad_segment)[0][torch.arange(int(noise_win * self.data_info['sfreq']))]
                if ied_peak_sample is not None:
                    temp = int(0.2 * self.data_info['sfreq'])
                    ied_sample_index = (torch.tensor(ied_peak_sample) +
                                        torch.arange(-temp, temp).reshape(-1, 1)).reshape(-1)
                    ied_sample_index = ied_sample_index[(ied_sample_index >= sample_index[0]) &
                                                        (ied_sample_index < sample_index[-1])]
                    sample_index = torch.tensor([x for x in sample_index if x not in ied_sample_index])
                noise_data_ied_temp = raw_data_ied[:, sample_index].to(self.device)
                noise_data_ied_temp = noise_data_ied_temp - noise_data_ied_temp.mean(dim=1, keepdim=True)
                cov_noise_ied = torch.mm(noise_data_ied_temp, noise_data_ied_temp.t()) / \
                                noise_data_ied_temp.reshape(-1).shape[0]
                inv_cov_noise_ied = self.pinv(noise_data_ied_temp, re_lambda=re_lambda)
                self.del_var('noise_data_ied_temp')
            else:
                cov_noise_ied, inv_cov_noise_ied = None, None

            # 计算hfo频段的数据协方差矩阵，使用整段数据。
            if raw_data_hfo is not None:
                # 删除IED片段
                sample_index = torch.where(~self.bad_segment)[0][torch.arange(int(noise_win * self.data_info['sfreq']))]
                if ied_peak_sample is not None:
                    temp = int(0.2 * self.data_info['sfreq'])
                    ied_sample_index = (torch.tensor(ied_peak_sample) +
                                        torch.arange(-temp, temp).reshape(-1, 1)).reshape(-1)
                    ied_sample_index = ied_sample_index[(ied_sample_index >= sample_index[0]) &
                                                        (ied_sample_index < sample_index[-1])]
                    sample_index = torch.tensor([x for x in sample_index if x not in ied_sample_index])
                noise_data_hfo_temp = raw_data_hfo[:, sample_index].to(self.device)
                noise_data_hfo_temp = noise_data_hfo_temp - noise_data_hfo_temp.mean(dim=1, keepdim=True)
                cov_noise_hfo = torch.mm(noise_data_hfo_temp, noise_data_hfo_temp.t()) / \
                                noise_data_hfo_temp.reshape(-1).shape[0]
                inv_cov_noise_hfo = self.pinv(noise_data_hfo_temp, re_lambda=re_lambda)
                self.del_var('noise_data_hfo_temp')
            else:
                cov_noise_hfo, inv_cov_noise_hfo = None, None

        return cov_noise_ied, inv_cov_noise_ied, cov_noise_hfo, inv_cov_noise_hfo

    def cal_leadfield_reduce_rank(self, rank=2):
        """
        Description:
            计算leadfield的reduced rank矩阵。
            u, s, v = svd(leadfield); s[rank:] = 0; leadfield_reduce = u * s * v

        Input:
            :param rank: number, double
                将leadfield的rank减少到的值
        Return:
            :return leadfield_reduce: torch.tensor, double, shape(n_dipoles*3*n_channels), in_device
                reduced rank后的leadfield
        """
        leadfield = self.get_leadfield().clone().to(self.device)
        with torch.no_grad():
            if leadfield is not None:
                u, s, v = torch.linalg.svd(leadfield.to(self.device), full_matrices=False)
                s[:, rank:] = 0
                leadfield_reduce = torch.bmm(torch.bmm(u, torch.diag_embed(s)), v)
                self.del_var()
            else:
                leadfield_reduce = None

        return leadfield_reduce

    def cal_fix_orientation(self, inv_cov_whole_ied=None, inv_cov_whole_hfo=None):
        """
        Description:
            使用数据协方差矩阵计算，源空信号的能量（方差）的最大方向。
            首先，计算计算源空间XYZ三个方向信号的协方差矩阵(van Veen 1997)：inv(lf' * inv_cov *lf)
            然后，使用svd计算方差变化的最大变化方向。

        Input:
            :param inv_cov_whole_ied: torch.tensor, double, shape(n_channels*n_channels), in_device
                ied频段数据的协方差矩阵逆矩阵
            :param inv_cov_whole_hfo: torch.tensor, double, shape(n_channels*n_channels), in_device
                hfo频段数据的协方差矩阵逆矩阵
        Return:
            :return fix_ori_ied: torch.tensor, double, shape(n_dipoles*3*1), in_device
                ied频段信号的源空间方向
            :return fix_ori_hfo: torch.tensor, double, shape(n_dipoles*3*1), in_device
                hfo频段信号的源空间方向
        """

        leadfield = self.get_leadfield().clone().to(self.device)
        with torch.no_grad():
            # 计算源空间信号IED频段的最大能量方向。
            if inv_cov_whole_ied is not None and leadfield is not None:
                # 计算源空间信号在XYZ方向上的协方差矩阵。
                source_signal_variance_ied = torch.linalg.pinv(
                    leadfield @ inv_cov_whole_ied.unsqueeze(dim=0).repeat(leadfield.shape[0], 1, 1) @
                    leadfield.transpose(1, 2))
                fix_ori_ied = torch.linalg.svd(source_signal_variance_ied)[0][:, :1].transpose(1, 2)
                self.del_var()
            else:
                fix_ori_ied = None

            # 计算源空间信号HFO频段的最大能量方向。
            if inv_cov_whole_hfo is not None and leadfield is not None:
                # 计算源空间信号在XYZ方向上的协方差矩阵。
                source_signal_variance_hfo = torch.linalg.pinv(
                    leadfield @ inv_cov_whole_hfo.unsqueeze(dim=0).repeat(leadfield.shape[0], 1, 1) @
                    leadfield.transpose(1, 2))
                fix_ori_hfo = torch.linalg.svd(source_signal_variance_hfo)[0][:, :1].transpose(1, 2)
                self.del_var()
            else:
                fix_ori_hfo = None

        return fix_ori_ied, fix_ori_hfo

    def cal_fix_ori_beamformer_weight(self, fix_ori_ied=None, fix_ori_hfo=None,
                                      inv_cov_whole_ied=None, inv_cov_whole_hfo=None):
        """
        Description:
            使用fix_orientation, 计算sam的beamformer weight

        Input:
            :param inv_cov_whole_ied: torch.tensor, double, shape(n_channels*n_channels), in_device
                ied频段数据的协方差矩阵逆矩阵
            :param inv_cov_whole_hfo: torch.tensor, double, shape(n_channels*n_channels), in_device
                hfo频段数据的协方差矩阵逆矩阵
            :param fix_ori_ied: torch.tensor, double, shape(n_dipoles*3*1), in_device
                ied频段信号的源空间方向(fix_ori)
            :param fix_ori_hfo: torch.tensor, double, shape(n_dipoles*3*1), in_device
                hfo频段信号的源空间方向(fix_ori)
        Return:
            :return fix_ori_weight_ied: torch.tensor, double, shape(n_dipoles*n_channels), in_device
                ied频段信号的beamformer weight
            :return fix_ori_weight_hfo: torch.tensor, double, shape(n_dipoles*n_channels), in_device
                hfo频段信号的beamformer weight
        """
        leadfield = self.get_leadfield().clone().to(self.device)
        with torch.no_grad():
            # 计算ied频段信号的beamformer weight
            if inv_cov_whole_ied is not None and leadfield is not None and fix_ori_ied is not None:
                lf_fix_ori_ied = torch.bmm(leadfield.transpose(1, 2), fix_ori_ied.type_as(leadfield)).squeeze()
                temp = torch.mm(lf_fix_ori_ied, inv_cov_whole_ied)
                fix_ori_weight_ied = (temp / torch.sum(temp * lf_fix_ori_ied, dim=1, keepdim=True))
                self.del_var()
            else:
                fix_ori_weight_ied = None

            # 计算hfo频段信号的beamformer weight
            if inv_cov_whole_hfo is not None and leadfield is not None and fix_ori_hfo is not None:
                lf_fix_ori_hfo = torch.bmm(leadfield.transpose(1, 2), fix_ori_hfo.type_as(leadfield)).squeeze()
                temp = torch.mm(lf_fix_ori_hfo, inv_cov_whole_hfo)
                fix_ori_weight_hfo = (temp / torch.sum(temp * lf_fix_ori_hfo, dim=1, keepdim=True))
                self.del_var()
            else:
                fix_ori_weight_hfo = None

        return fix_ori_weight_ied, fix_ori_weight_hfo

    def cal_max_ori_beamformer_weight(self, max_ori_ied=None, max_ori_hfo=None,
                                      inv_cov_whole_ied=None, inv_cov_whole_hfo=None):
        """
        Description:
            使用max_orientation, 计算sam的beamformer weight

        Input:
            :param inv_cov_whole_ied: torch.tensor, double, shape(n_channels*n_channels), in_device
                ied频段数据的协方差矩阵逆矩阵
            :param inv_cov_whole_hfo: torch.tensor, double, shape(n_channels*n_channels), in_device
                hfo频段数据的协方差矩阵逆矩阵
            :param max_ori_ied: torch.tensor, double, shape(n_ieds*n_segments*n_dipoles*3*1), in_device
                ied频段信号的源空间方向(max_ori)
            :param max_ori_hfo: torch.tensor, double, shape(n_ieds*n_segments*n_dipoles*3*1), in_device
                hfo频段信号的源空间方向(max_ori)
        Return:
            :return max_ori_weight_ied: torch.tensor, double, shape(n_ieds*n_segments*n_dipoles*n_channels), in_device
                ied频段信号的beamformer weight
            :return max_ori_weight_hfo: torch.tensor, double, shape(n_ieds*n_segments*n_dipoles*n_channels), in_device
                hfo频段信号的beamformer weight
        """
        leadfield = self.get_leadfield().clone().to(self.device)
        with torch.no_grad():
            # 计算ied频段信号的beamformer weight
            if inv_cov_whole_ied is not None and leadfield is not None and max_ori_ied is not None:
                lf_max_ori_ied = torch.bmm(leadfield.transpose(1, 2).repeat(max_ori_ied.shape[0] *
                                                                            max_ori_ied.shape[1], 1, 1),
                                           max_ori_ied.reshape(-1, 3, 1).type_as(leadfield)).squeeze()
                temp = torch.mm(lf_max_ori_ied, inv_cov_whole_ied)
                max_ori_weight_ied = (temp / torch.sum(temp * lf_max_ori_ied, dim=1, keepdim=True))
                max_ori_weight_ied = max_ori_weight_ied.reshape(max_ori_ied.shape[0], max_ori_ied.shape[1],
                                                                leadfield.shape[0], temp.shape[1])
                self.del_var()
            else:
                max_ori_weight_ied = None

            # 计算ied频段信号的beamformer weight
            if inv_cov_whole_hfo is not None and leadfield is not None and max_ori_hfo is not None:
                lf_max_ori_hfo = torch.bmm(leadfield.transpose(1, 2).repeat(max_ori_hfo.shape[0] *
                                                                            max_ori_hfo.shape[1], 1, 1),
                                           max_ori_hfo.reshape(-1, 3, 1).type_as(leadfield)).squeeze()
                temp = torch.mm(lf_max_ori_hfo, inv_cov_whole_hfo)
                max_ori_weight_hfo = (temp / torch.sum(temp * lf_max_ori_hfo, dim=1, keepdim=True))
                max_ori_weight_hfo = max_ori_weight_hfo.reshape(max_ori_hfo.shape[0], max_ori_hfo.shape[1],
                                                                leadfield.shape[0], temp.shape[1])
                self.del_var()
            else:
                max_ori_weight_hfo = None

        return max_ori_weight_ied, max_ori_weight_hfo

    @staticmethod
    def pinv(data, re_lambda=0.05, kappa=None):
        """
        Description:
            计算数据的伪逆矩阵

        Input:
            :param data: torch.tensor, double, shape(x*y)
                输入矩阵
            :param re_lambda: number, double
                求违逆时候的正则化系数
            :param kappa: number, double
                截断参数
        Return:
            :return inv_data: torch.tensor, double, shape(y*x)
                输入矩阵的伪逆矩阵
        """
        with torch.no_grad():
            # 计算数据的svd
            u, s, v = torch.linalg.svd(data)
            v = v.t()
            # 计算re_lambda值
            re_lambda = re_lambda * s.mean()
            # 计算kappa值
            if kappa is None:
                tolerance = 10 * max(data.shape[0], data.shape[1]) * 2.2204e-16
                kappa = (s / s[0] > tolerance).sum()
            elif kappa > data.shape[0]:
                kappa = data.shape[0]
            # 计算伪逆矩阵
            s = torch.diag(1. / (s[:kappa] + re_lambda))
            inv_data = v[:, :kappa] @ s @ u[:, :kappa].t()

        return inv_data

    def cal_orientations_half_sphere(self, orientation_min_dist=0.95, orientations_subset_samples=15,
                                     subset_orientation_min_dist=0.99):
        """
        Description:
            根据方向间最小cos距离，计算方向采样样本；以及对应的精细采样方向样本。

        Input:
            :param orientation_min_dist: number, double
                方向样本间的最小cos距离
            :param orientations_subset_samples: number, long
                方向样本的子样本采样个数
            :param subset_orientation_min_dist: number, double
                方向样本的子样本间的最小cos距离
        Return:
            :return xyz_coord: torch.tensor, double, shape(n_oris*3), in_device
                方向采样
            :return sub_xyz_coord: torch.tensor, double, shape(n_oris*n_max_subsets*3), in_device
                方向采样的子样本
        """

        # 计算0-180度，满足相邻采样方向内积小于orientation_min_dist的采样个数
        ori_samples = round(np.pi / np.arccos(orientation_min_dist) * 4)
        # 根据采样点数，计算极坐标的phi和theta
        phi = np.linspace(0, np.pi, ori_samples)
        theta = np.linspace(0, np.pi, ori_samples)
        # 对根据极坐标，计算满足orientation_min_dist的方向样本，并转换为xyz坐标。
        xyz_coord, theta_phi = self.theta_phi_to_xyz(phi=phi, theta=theta, orientation_min_dist=orientation_min_dist)
        # 对每个方向样本，再进行精细采样。
        sub_xyz_coord = []
        for i, x in enumerate(theta_phi):
            sub_xyz_temp = self.theta_phi_to_xyz(
                phi=np.linspace(x[0] - np.arccos(orientation_min_dist), x[0] + np.arccos(orientation_min_dist),
                                orientations_subset_samples),
                theta=np.linspace(x[1] - np.arccos(orientation_min_dist), x[1] + np.arccos(orientation_min_dist),
                                  orientations_subset_samples),
                center_ori=xyz_coord[i], center_min_dist=orientation_min_dist,
                orientation_min_dist=subset_orientation_min_dist)[0]
            sub_xyz_coord.append(torch.tensor(sub_xyz_temp))
        # 输出
        xyz_coord = torch.tensor(xyz_coord).to(self.device).double()
        sub_xyz_coord = pad_sequence(sub_xyz_coord, batch_first=True).to(self.device).double()

        return xyz_coord, sub_xyz_coord

    @staticmethod
    def theta_phi_to_xyz(phi, theta, orientation_min_dist=0.95, center_ori=None, center_min_dist=None):
        """
        Description:
            根据极坐标，计算笛卡尔坐标系样本，并使得样本满足最小cos距离

        Input:
            :param phi: np.array, double, shape(n_oris)
                极坐标的phi角
            :param theta: np.array, double, shape(n_oris)
                极坐标的theta角
            :param orientation_min_dist: number, long
                方向样本间的最小角度距离
            :param center_ori: number, long, shape(3, 1)
                方向样本中心的xyz坐标
            :param center_min_dist: number, double
                方向样本与中新的最大角度距离
        Return:
            :return xyz_coord: np.array, double, shape(n_oris_new*3)
                方向采样
            :return sub_xyz_coord: np.array, double, shape(n_oris_new*3)
                方向采样的子样本
        """

        # 将极坐标转换为笛卡尔坐标
        xyz_coord = np.concatenate((np.outer(np.sin(phi), np.cos(theta)).reshape(-1, 1),
                                    np.outer(np.sin(phi), np.sin(theta)).reshape(-1, 1),
                                    np.outer(np.cos(phi), np.ones_like(theta)).reshape(-1, 1)), axis=1)
        # 满足和中心方向距离小于center_min_dist
        if center_ori is not None and center_min_dist is not None:
            xyz_coord = xyz_coord[np.where(np.matmul(xyz_coord, center_ori.reshape(-1, 1)) > center_min_dist)[0]]

        # 保证两两方向之间最小距离小于orientation_min_dist
        # 将两两方向之间距离小于orientation_min_dist的方向聚类再一起
        cls = [[0]]
        for i in range(1, xyz_coord.shape[0]):
            new_idx = []
            for (j, cls_idx) in enumerate(cls):
                temp = np.sum(xyz_coord[i] * xyz_coord[cls_idx], axis=1)
                if all(abs(temp) > orientation_min_dist):
                    if all(temp > orientation_min_dist):
                        new_idx = [j]
                    continue
            if len(new_idx) != 0:
                cls[new_idx[0]].append(i)
            else:
                cls.append([i])
        # 用聚类中心代表当前类别
        xyz_coord = np.array([np.mean(xyz_coord[x, :], axis=0) for x in cls])
        # 将方向转换为单位方向
        xyz_coord = np.concatenate([x.reshape(1, -1) / np.linalg.norm(x, axis=0) for x in xyz_coord], axis=0)
        # 计算每个方向对应的极坐标
        theta_phi = np.concatenate([np.array([np.arccos(x[2]), np.arctan2(x[1], x[0])]).reshape(1, -1)
                                    for x in xyz_coord], axis=0)
        return xyz_coord, theta_phi

    def cal_data_segments_around_ied(self, ied_peak_sample, segment_time=2, interest_window=0.1, interest_segment=0.3):
        """
        Description:
            截取ied周围数据片段，并去除bad segment。
            获取ied片段内，感兴趣滑动时间窗的采样点，以及对应的baseline采样点。

        Input:
            :param ied_peak_sample: np.array, double, shape(n_ieds)
                ied主peak的采样点
            :param segment_time: number, double
                ied segment的时间窗长度，单位s
            :param interest_window: number, double
                感兴趣时间窗长度，单位s
            :param interest_segment: number, double
                感兴趣片段长度，单位s
        Return:
            :return ied_segment_samples: torch.tensor, double, shape(n_ieds*n_segment_samples), in_device
                ied segments在原始数据中的采样点
            :return interest_samples: torch.tensor, double, shape(n_interest*n_interest_samples), in_device
                感兴趣时间窗，在ied segment中的采样点位置
            :return baseline_samples: torch.tensor, double, shape(n_interest*n_interest_samples), in_device
                感兴趣时间窗对应的baseline(去除感兴趣时间窗)，在ied segment中的采样点位置
        """

        # 获取ied segment采样点，确保bad segment被跳过。
        half_segment_samples = int(segment_time * self.data_info['sfreq'] / 2)
        segment_range = torch.arange(-half_segment_samples, half_segment_samples)
        # 将bad segment从数据中去除，获得在新数据中ied segment的采样点位置。
        ied_index = torch.zeros(self.bad_segment.shape[0]).long()
        ied_index[torch.tensor(ied_peak_sample).long() - 1] = 1
        ied_index = ied_index[~self.bad_segment]
        ied_index = torch.where(ied_index == 1)[0].reshape(-1, 1) + segment_range.reshape(1, -1)
        # 确保采样点在0:n_samples之间
        n_samples = torch.where(~self.bad_segment)[0].shape[0] - 1
        ied_index = ied_index[torch.where((ied_index[:, 0] >= 0) & (ied_index[:, -1] <= n_samples))[0]]
        # 获取原始数据中ied segment对应的采样点位置
        sample_index = torch.arange(self.bad_segment.shape[0]).long()[~self.bad_segment]
        ied_segment_samples = sample_index[ied_index]

        # 获取感兴趣片段的滑动时间窗采样点
        half_interest_segments = int(interest_segment * self.data_info['sfreq'] / 2)
        half_interest_windows = int(interest_window * self.data_info['sfreq'] / 2)
        interest_samples = torch.stack([torch.arange(x, x + half_interest_windows * 2) + half_segment_samples
                                        for x in torch.arange(-half_interest_segments, half_interest_segments - 1,
                                                              half_interest_windows)])
        if half_interest_segments == half_interest_windows:
            interest_samples = interest_samples[0:1]

        baseline_samples = torch.stack([torch.cat([torch.arange(0, x[0]),
                                                   torch.arange(x[-1] + 1, segment_range.shape[0])])
                                        for x in interest_samples]).long()

        return ied_segment_samples.to(self.device), interest_samples.to(self.device), baseline_samples.to(self.device)

    def sam_costfun_max_hilbert_power(self, data, leadfields, inv_covariance,
                                      xyz_coords, baseline_samples, interest_samples,
                                      xyz_coords_subs, baseline_samples_sub, interest_samples_sub,
                                      average_data_conv, hilbert_h):
        """
        Description:
            使用非线性搜索，搜索最优方向：
            1. 使用大分辨率方向，搜索最大信号包络方向
                (1).将lead field矩阵(n_dipoles*3*n_channels)投影到粗略采样的方向xyz_coords(n_dipoles*n_oris*3)
                    获得gain(n_dipoles*n_oris*n_channels)。
                (2).计算beamformer的逆算子: weight = (lf * inv_cov) / (lf * inv_cov * lf')。
                    计算源空间信号: vs = weight * meg_data; vs(n_dipoles*n_oris*n_samples)
                    计算源空间信号的包络: vs_hilbert = hilbert(vs)
                (3).提取感兴趣时间窗信号，以及对应的baseline信号:
                    data_interest: shape(n_dipoles * n_oris, segments_num, n_samples_interest)
                    data_baseline: shape(n_dipoles * n_oris, segments_num, n_samples_baseline)
                (4).使用baseline信号对时间窗信号进行z-score, 获得相对于baseline的信号强度。
                (5).计算所有感兴趣时间窗信号的平均包络，选取包络值最大的方向。
            2. 使用最大包络方向的精细方向，在进行搜索。将大分辨率方向改为精确方向采样，重复上述步骤。

        Input:
            :param data: torch.tensor, double, shape(n_channels*n_samples), in_device
                ied周围meg原始数据
            :param leadfields: torch.tensor, double, shape(n_dipoles*3*n_channels), in_device
                多个dipole的leadfield矩阵
            :param inv_covariance: torch.tensor, double, shape(n_channels*n_channels), in_device
                协方差矩阵逆矩阵
            :param xyz_coords: torch.tensor, double, shape(n_oris*3), in_device
                方向采样
            :param interest_samples: torch.tensor, double, shape(n_oris*n_segments*n_segments_samples), in_device
                感兴趣时间窗，在ied segment中的采样点位置
            :param baseline_samples: torch.tensor, double, shape(n_oris*n_segments*n_segments_samples), in_device
                感兴趣时间窗对应的baseline(去除感兴趣时间窗)，在ied segment中的采样点位置
            :param xyz_coords_subs: torch.tensor, double, shape(n_oris*n_ori_subs*3), in_device
                方向采样的子样本
            :param interest_samples_sub: torch.tensor, double, shape(n_ori_subs*n_segments*n_segments_samples), in_device
                感兴趣时间窗，在ied segment中的采样点位置
            :param baseline_samples_sub: torch.tensor, double, shape(n_ori_subs*n_segments*n_segments_samples), in_device
                感兴趣时间窗对应的baseline(去除感兴趣时间窗)，在ied segment中的采样点位置
            :param average_data_conv: torch.conv, in_device
                用于计算的，卷积核
            :param hilbert_h: torch.tensor, double, shape(n_samples), in_device
                hilbert_h输出

        Return:
            :return max_ori: torch.tensor, double, shape(n_dipoles*n_segments), in_device
                最优方向
            :return ori_average_amplitude: torch.tensor, double, shape(n_dipoles*n_segments), in_device
                最优方向对应的能量大小
        """

        grids_num, segments_num = leadfields.shape[0], interest_samples.shape[1]
        coarse_ori_samples, fine_ori_samples = interest_samples.shape[0], interest_samples_sub.shape[0]

        # step1: 粗略的最优方向搜索
        # 计算源空间数据，并计算其希尔伯特变换
        gain = torch.bmm(xyz_coords.unsqueeze(0).repeat(grids_num, 1, 1), leadfields)
        temp = torch.bmm(gain, inv_covariance.unsqueeze(0).repeat(gain.shape[0], 1, 1))
        data_hilbert = self.hilbert(
            torch.mm((temp / torch.sum(temp * gain, dim=-1, keepdim=True)).reshape(-1, temp.shape[-1]), data),
            hilbert_h).reshape(grids_num, coarse_ori_samples, -1)
        self.del_var()
        # 将每个source grids和每个segments的数据提取出来，并提取对应的baseline数据
        # data_interest: shape(grids_num * coarse_ori_samples, segments_num, n_samples_interest)
        # data_baseline: shape(grids_num * coarse_ori_samples, segments_num, n_samples_baseline)
        data_interest_index = interest_samples.reshape(1, coarse_ori_samples, -1).repeat(grids_num, 1, 1)
        data_interest = torch.gather(data_hilbert, dim=-1, index=data_interest_index)
        data_interest = data_interest.reshape(grids_num * coarse_ori_samples, segments_num, -1)
        data_baseline_index = baseline_samples.reshape(1, coarse_ori_samples, -1).repeat(grids_num, 1, 1)
        data_baseline = torch.gather(data_hilbert, dim=-1, index=data_baseline_index)
        data_baseline = data_baseline.reshape(grids_num * coarse_ori_samples, segments_num, -1)
        # 计算感兴趣数据片段的包络相对于baseline的幅度（z-score）
        data_interest = (data_interest - torch.mean(data_baseline, dim=-1, keepdim=True)) / \
                        torch.std(data_baseline, dim=-1, keepdim=True)
        self.del_var("data_baseline")
        # 计算每个时间窗的包络均值
        # data_interest: shape(grids_num, coarse_ori_samples, segments_num, n_samples)
        data_interest = average_data_conv(data_interest.unsqueeze(1).float()).squeeze()
        data_interest = data_interest.reshape(grids_num, coarse_ori_samples, segments_num, -1)
        self.del_var()

        # step2: 精细的最优方向搜索
        # 计算源空间数据，并计算其希尔伯特变换
        # xyz_coords_sub: shape(grids_num * segments_num, fine_ori_samples, 3)
        xyz_coords_sub = xyz_coords_subs[data_interest.amax(dim=-1).argmax(dim=-2).reshape(-1)]
        gain = torch.bmm(xyz_coords_sub, leadfields.unsqueeze(1).repeat(
            (1, segments_num, 1, 1)).reshape(-1, leadfields.shape[1], leadfields.shape[2]))
        temp = torch.bmm(gain, inv_covariance.unsqueeze(0).repeat(gain.shape[0], 1, 1))
        data_hilbert = self.hilbert(
            torch.mm((temp / torch.sum(temp * gain, dim=-1, keepdim=True)).reshape(-1, temp.shape[-1]), data),
            hilbert_h).reshape(grids_num, segments_num, fine_ori_samples, -1)
        self.del_var()
        # 将每个source grids和每个segments的数据提取出来，并提取对应的baseline数据
        # data_interest: shape(grids_num * segments_num, fine_ori_samples, n_samples_interest)
        # data_baseline: shape(grids_num * segments_num, fine_ori_samples, n_samples_baseline)
        data_interest_index = interest_samples_sub.transpose(0, 1).unsqueeze(0).repeat(grids_num, 1, 1, 1)
        data_interest = torch.gather(data_hilbert, dim=-1, index=data_interest_index)
        data_interest = data_interest.reshape(grids_num * segments_num, fine_ori_samples, -1)
        data_baseline_index = baseline_samples_sub.transpose(0, 1).unsqueeze(0).repeat(grids_num, 1, 1, 1)
        data_baseline = torch.gather(data_hilbert, dim=-1, index=data_baseline_index)
        data_baseline = data_baseline.reshape(grids_num * segments_num, fine_ori_samples, -1)
        self.del_var("data_baseline")
        # 计算感兴趣数据片段的包络相对于baseline的幅度（z-score）
        data_interest = (data_interest - torch.mean(data_baseline, dim=-1, keepdim=True)) / \
                        torch.std(data_baseline, dim=-1, keepdim=True)
        # 计算每个时间窗的包络均值
        # data_interest: shape(grids_num * segments_num, fine_ori_samples, n_samples)
        data_interest = average_data_conv(data_interest.unsqueeze(1).float()).squeeze()
        data_interest = data_interest.reshape(grids_num, segments_num, fine_ori_samples, -1)
        self.del_var()

        # step3: 输出方向和对应HFO幅度
        data_interest = torch.nan_to_num(data_interest, nan=-10).amax(dim=-1)
        max_ori_index = data_interest.argmax(dim=-1)
        max_ori = xyz_coords_sub.take_along_dim(max_ori_index.reshape(-1, 1, 1),
                                                dim=1).squeeze().reshape(grids_num, segments_num, 3)
        ori_average_amplitude = data_interest.amax(dim=-1)
        self.del_var()

        return max_ori, ori_average_amplitude

    def cal_max_ori_using_hfo_windows(self, ied_peak_sample, inv_covariance, segment_time=2,
                                      hfo_window=0.1, hfo_segment=0.3, mean_power_windows=0.05, chunk_number=300):
        """
        Description:
            使用非线性搜索，计算hfo的源空间信号最优方向：

        Input:
            :param ied_peak_sample: np.array, double, shape(n_ieds), in_device
                ied主peak的采样点
            :param inv_covariance: torch.tensor, double, shape(n_channels*n_channels), in_device
                协方差矩阵逆矩阵
            :param segment_time: number, double
                ied segment的时间窗长度，单位s
            :param hfo_window: number, double
                用于计算方向的hfo片段的时间窗长度，单位s
            :param hfo_segment: number, double
                对于一个ied，ied主峰值周围的时间窗长度(hfo可能出现在这个窗内)，单位s
            :param mean_power_windows: number, double
                计算平均包络的时间窗长度，单位s
            :param chunk_number: number, long
                使用chunk的方式减少显存的使用，每个chunk的大小。

        Return:
            :return max_ori: torch.tensor, double, shape(n_ieds*n_segment_samples*n_dipoles*3), in_device
                最优源空间信号方向。
            :return ied_segment_samples: torch.tensor, double, shape(n_ieds*n_ied_segment_samples), in_device
                每个ied的时间窗，在整段数据中的采样点位置。
            :return hfo_window_samples: torch.tensor, double, shape(n_ieds*n_segment_samples*n_hfo_samples), in_device
                hfo时间窗在ied segment中的采样点位置
            :return baseline_window_samples: torch.tensor, double, shape(n_ieds*n_segment_samples*n_baseline_samples), in_device
                感兴趣时间窗对应的baseline(去除hfo)，在ied segment中的采样点位置
        """
        leadfield = self.get_leadfield().clone().to(self.device)

        # step1: 获取半球方向的采样点集合，半球是因为源空间方向的正负对信号的幅度没有影响
        xyz_coords, xyz_coords_sub = \
            self.cal_orientations_half_sphere(orientation_min_dist=0.95, orientations_subset_samples=15,
                                              subset_orientation_min_dist=0.99)

        # step2: 获取IED segment的采样点，以及segment中HFO时间窗和对应baseline的采样点
        ied_segment_samples, hfo_samples, baseline_samples = \
            self.cal_data_segments_around_ied(ied_peak_sample=ied_peak_sample, segment_time=segment_time,
                                              interest_window=hfo_window, interest_segment=hfo_segment)

        # step3: 对MEG数据进行切片，计算hfo片段和baseline在切片中的采样点。
        ied_data_segments = self.get_raw_data_hfo().unsqueeze(0).to(self.device).take_along_dim(
            ied_segment_samples.unsqueeze(1), dim=-1)
        # 计算HFO时间窗
        hfo_samples_sub = hfo_samples.unsqueeze(0).repeat(xyz_coords_sub.shape[1], 1, 1)
        hfo_samples = hfo_samples.unsqueeze(0).repeat(xyz_coords.shape[0], 1, 1)
        # 计算baseline时间窗
        baseline_samples_sub = baseline_samples.unsqueeze(0).repeat(xyz_coords_sub.shape[1], 1, 1)
        baseline_samples = baseline_samples.unsqueeze(0).repeat(xyz_coords.shape[0], 1, 1)

        # step4: 计算hilbert变换参数，以及用于求平均的卷积核
        hilbert_h = self.hilbert_h(ied_segment_samples.shape[-1]).to(self.device)
        average_data_conv = \
            torch.nn.Conv2d(1, 1, [1, np.round(mean_power_windows * self.data_info['sfreq']).astype('int16')],
                            bias=False).to(self.device)
        torch.nn.init.constant_(average_data_conv.weight, 1 /
                                np.round(mean_power_windows * self.data_info['sfreq']).astype('int16'))

        # step5: 使用线性搜索，计算最优方向。
        # 在source grids维度使用chunk的方式减少显存的使用。
        max_ori = []
        for ied in range(ied_segment_samples.shape[0]):
            with torch.no_grad():
                max_ori_temp = torch.zeros((leadfield.shape[0], hfo_samples.shape[1], 3),
                                           dtype=torch.float64).to(self.device)
                # 使用chunk的方式减少显存的使用
                for chunk_idx in range(int(np.ceil(leadfield.shape[0] / chunk_number))):
                    vs_channel_idx = torch.arange(int(chunk_idx * chunk_number),
                                                  int(min(leadfield.shape[0], (chunk_idx + 1) * chunk_number)))
                    max_ori_temp[vs_channel_idx], _ = \
                        self.sam_costfun_max_hilbert_power(data=ied_data_segments[ied],
                                                           leadfields=leadfield[vs_channel_idx],
                                                           inv_covariance=inv_covariance,
                                                           xyz_coords=xyz_coords,
                                                           baseline_samples=baseline_samples,
                                                           interest_samples=hfo_samples,
                                                           xyz_coords_subs=xyz_coords_sub,
                                                           baseline_samples_sub=baseline_samples_sub,
                                                           interest_samples_sub=hfo_samples_sub,
                                                           average_data_conv=average_data_conv, hilbert_h=hilbert_h)
                    self.del_var()
                max_ori.append(max_ori_temp)

        # 输出
        max_ori = torch.stack(max_ori, dim=0).transpose(1, 2)
        hfo_window_samples = hfo_samples[:1].repeat(max_ori.shape[0], 1, 1)
        baseline_window_samples = baseline_samples[:1].repeat(max_ori.shape[0], 1, 1)
        return max_ori, ied_segment_samples, hfo_window_samples, baseline_window_samples

    def sam_costfun_peak2peak(self, data, leadfields, inv_covariance,
                              xyz_coords, baseline_samples, interest_samples,
                              xyz_coords_subs, baseline_samples_sub, interest_samples_sub, segment_data_conv):
        """
        Description:
            使用非线性搜索，搜索最优方向(峰峰值)：
            1. 使用大分辨率方向，搜索最大信号峰峰值方向
                (1).将lead field矩阵(n_dipoles*3*n_channels)投影到粗略采样的方向xyz_coords(n_dipoles*n_oris*3)
                    获得gain(n_dipoles*n_oris*n_channels)。
                (2).计算beamformer的逆算子: weight = (lf * inv_cov) / (lf * inv_cov * lf')。
                    计算源空间信号: vs = weight * meg_data; vs(n_dipoles*n_oris*n_samples)
                (3).提取感兴趣时间窗信号，以及对应的baseline信号:
                    data_interest: shape(n_dipoles, n_oris, n_samples_interest)
                    data_baseline: shape(n_dipoles,  n_oris, n_samples_baseline)
                (4).使用baseline信号对时间窗信号进行z-score, 获得相对于baseline的信号强度。
                (5).计算所有感兴趣时间窗信号的峰峰值，选取包络值最大的方向。
            2. 使用最大峰峰值方向的精细方向，在进行搜索。将大分辨率方向改为精确方向采样，重复上述步骤。

        Input:
            :param data: torch.tensor, double, shape(n_channels*n_samples), in_device
                ied周围meg原始数据
            :param leadfields: torch.tensor, double, shape(n_dipoles*3*n_channels), in_device
                多个dipole的leadfield矩阵
            :param inv_covariance: torch.tensor, double, shape(n_channels*n_channels), in_device
                协方差矩阵逆矩阵
            :param xyz_coords: torch.tensor, double, shape(n_oris*3), in_device
                方向采样
            :param interest_samples: torch.tensor, double, shape(n_oris*n_interest_samples), in_device
                感兴趣时间窗，在ied segment中的采样点位置
            :param baseline_samples: torch.tensor, double, shape(n_oris*n_baseline_samples), in_device
                感兴趣时间窗对应的baseline(去除感兴趣时间窗)，在ied segment中的采样点位置
            :param xyz_coords_subs: torch.tensor, double, shape(n_oris*n_ori_subs*3), in_device
                方向采样的子样本
            :param interest_samples_sub: torch.tensor, double, shape(n_ori_subs*n_interest_samples), in_device
                感兴趣时间窗，在ied segment中的采样点位置
            :param baseline_samples_sub: torch.tensor, double, shape(n_ori_subs*n_baseline_samples), in_device
                感兴趣时间窗对应的baseline(去除感兴趣时间窗)，在ied segment中的采样点位置
            :param segment_data_conv: torch.conv, in_device
                用于计算的，卷积核

        Return:
            :return max_ori: torch.tensor, double, shape(n_dipoles*3), in_device
                最优方向
            :return ori_average_amplitude: torch.tensor, double, shape(n_dipoles), in_device
                最优方向对应的能量大小
        """

        grids_num = leadfields.shape[0]
        coarse_ori_samples, fine_ori_samples = interest_samples.shape[0], interest_samples_sub.shape[0]

        # step1: 粗略的最优方向搜索
        # 计算源空间数据
        gain = torch.bmm(xyz_coords.unsqueeze(0).repeat(grids_num, 1, 1), leadfields)
        temp = torch.bmm(gain, inv_covariance.unsqueeze(0).repeat(gain.shape[0], 1, 1))
        data_vs = torch.mm((temp / torch.sum(temp * gain, dim=-1, keepdim=True)).reshape(-1, temp.shape[-1]),
                           data).reshape(grids_num, coarse_ori_samples, -1)
        self.del_var()
        # 将每个source grids和每个segments的数据提取出来，并提取对应的baseline数据
        # data_interest: shape(grids_num, coarse_ori_samples, n_samples_interest)
        # data_baseline: shape(grids_num, coarse_ori_samples, n_samples_baseline)
        data_interest_index = interest_samples.reshape(1, coarse_ori_samples, -1).repeat(grids_num, 1, 1)
        data_interest = torch.gather(data_vs, dim=-1, index=data_interest_index)
        data_baseline_index = baseline_samples.reshape(1, coarse_ori_samples, -1).repeat(grids_num, 1, 1)
        data_baseline = torch.gather(data_vs, dim=-1, index=data_baseline_index)
        # 计算感兴趣数据片段数据相对于baseline的幅度（z-score）
        data_interest = (data_interest - torch.mean(data_baseline, dim=-1, keepdim=True)) / \
                        torch.std(data_baseline, dim=-1, keepdim=True)
        data_interest = torch.nan_to_num(data_interest, nan=0)
        self.del_var()
        # 获取用于计算峰峰值的片段
        # data_interest: shape(grids_num, coarse_ori_samples, peak2peak_samples, n_peak2peak_windows)
        data_interest = segment_data_conv(data_interest.unsqueeze(2).float())
        data_interest = data_interest.reshape(grids_num, coarse_ori_samples, segment_data_conv.kernel_size[1], -1)
        self.del_var()
        # 计算峰峰值
        # data_interest: shape(grids_num, coarse_ori_samples)
        data_interest = (data_interest.amax(dim=-2) - data_interest.amin(dim=-2)).amax(dim=-1)
        self.del_var()

        # step2: 精细的最优方向搜索
        # 计算源空间数据
        # xyz_coords_sub: shape(grids_num, fine_ori_samples, 3)
        xyz_coords_sub = xyz_coords_subs[data_interest.argmax(dim=-1).reshape(-1)]
        gain = torch.bmm(xyz_coords_sub, leadfields)
        temp = torch.bmm(gain, inv_covariance.unsqueeze(0).repeat(gain.shape[0], 1, 1))
        data_vs = torch.mm((temp / torch.sum(temp * gain, dim=-1, keepdim=True)).reshape(-1, temp.shape[-1]),
                           data).reshape(grids_num, fine_ori_samples, -1)
        self.del_var()
        # 将每个source grids的数据提取出来，并提取对应的baseline数据
        # data_interest: shape(grids_num, fine_ori_samples, n_samples_interest)
        # data_baseline: shape(grids_num, fine_ori_samples, n_samples_baseline)
        data_interest_index = interest_samples_sub.unsqueeze(0).repeat(grids_num, 1, 1)
        data_interest = torch.gather(data_vs, dim=-1, index=data_interest_index)
        data_interest = data_interest.reshape(grids_num, fine_ori_samples, -1)
        data_baseline_index = baseline_samples_sub.unsqueeze(0).repeat(grids_num, 1, 1)
        data_baseline = torch.gather(data_vs, dim=-1, index=data_baseline_index)
        data_baseline = data_baseline.reshape(grids_num, fine_ori_samples, -1)
        self.del_var("data_baseline")
        # 计算感兴趣数据片段数据相对于baseline的幅度（z-score）
        data_interest = (data_interest - torch.mean(data_baseline, dim=-1, keepdim=True)) / \
                        torch.std(data_baseline, dim=-1, keepdim=True)
        data_interest = torch.nan_to_num(data_interest, nan=0)
        self.del_var()
        # 获取用于计算峰峰值的片段
        # data_interest: shape(grids_num, fine_ori_samples, peak2peak_samples, n_peak2peak_windows)
        data_interest = segment_data_conv(data_interest.unsqueeze(2).float())
        data_interest = data_interest.reshape(grids_num, fine_ori_samples, segment_data_conv.kernel_size[1], -1)
        self.del_var()
        # 计算峰峰值
        # data_interest: shape(grids_num, coarse_ori_samples)
        data_interest = (data_interest.amax(dim=-2) - data_interest.amin(dim=-2)).amax(dim=-1)
        self.del_var()

        # step3: 输出方向和对应HFO幅度
        max_ori_index = data_interest.argmax(dim=-1)
        max_ori = xyz_coords_sub.take_along_dim(max_ori_index.reshape(-1, 1, 1), dim=1).squeeze().reshape(grids_num, 3)
        ori_average_amplitude = data_interest.amax(dim=-1)
        self.del_var()

        return max_ori, ori_average_amplitude

    def cal_max_ori_using_ied_windows(self, ied_peak_sample, inv_covariance, segment_time=2,
                                      ied_window=0.15, ied_segment=0.15, peak2peak_windows=0.03, chunk_number=1500):
        """
        Description:
            使用非线性搜索，计算ied的源空间信号最优方向：

        Input:
            :param ied_peak_sample: np.array, double, shape(n_ieds), in_device
                ied主peak的采样点
            :param inv_covariance: torch.tensor, double, shape(n_channels*n_channels), in_device
                协方差矩阵逆矩阵
            :param segment_time: number, double
                ied segment的时间窗长度，单位s
            :param ied_window: number, double
                用于计算方向的ied片段的时间窗长度，单位s
            :param ied_segment: number, double
                对于一个ied，ied主峰值周围的时间窗长度(ied可能出现在这个窗内)，单位s
            :param peak2peak_windows: number, double
                计算峰峰值的时间窗长度，单位s
            :param chunk_number: number, long
                使用chunk的方式减少显存的使用，每个chunk的大小。

        Return:
            :return max_ori: torch.tensor, double, shape(n_ieds*n_dipoles*3), in_device
                最优源空间信号方向。
            :return ied_segment_samples: torch.tensor, double, shape(n_ieds*n_ied_segment_samples), in_device
                每个ied的时间窗，在整段数据中的采样点位置。
            :return ied_window_samples: torch.tensor, double, shape(n_ieds*n_ied_samples), in_device
                ied时间窗在ied segment中的采样点位置
            :return baseline_window_samples: torch.tensor, double, shape(n_ieds*n_baseline_samples), in_device
                感兴趣时间窗对应的baseline(去除ied)，在ied segment中的采样点位置
        """
        leadfield = self.get_leadfield().clone().to(self.device)

        # step1: 获取半球方向的采样点集合，半球是因为源空间方向的正负对信号的幅度没有影响
        xyz_coords, xyz_coords_sub = \
            self.cal_orientations_half_sphere(orientation_min_dist=0.95, orientations_subset_samples=15,
                                              subset_orientation_min_dist=0.99)

        # step2: 获取IED segment的采样点，以及segment中IED时间窗和对应baseline的采样点
        ied_segment_samples, ied_samples, baseline_samples = \
            self.cal_data_segments_around_ied(ied_peak_sample=ied_peak_sample, segment_time=segment_time,
                                              interest_window=ied_window, interest_segment=ied_segment)
        ied_samples, baseline_samples = ied_samples.squeeze(), baseline_samples.squeeze()

        # step3: 对MEG数据进行切片，计算ied片段和baseline在切片中的采样点。
        ied_data_segments = self.get_raw_data_ied().unsqueeze(0).to(self.device).take_along_dim(
            ied_segment_samples.unsqueeze(1), dim=-1)
        # 计算ied时间窗
        ied_samples_sub = ied_samples.unsqueeze(0).repeat(xyz_coords_sub.shape[1], 1)
        ied_samples = ied_samples.unsqueeze(0).repeat(xyz_coords.shape[0], 1)
        # 计算baseline时间窗
        baseline_samples_sub = baseline_samples.unsqueeze(0).repeat(xyz_coords_sub.shape[1], 1)
        baseline_samples = baseline_samples.unsqueeze(0).repeat(xyz_coords.shape[0], 1)

        # step4: 计算用于求峰峰值的卷积核
        segment_data_conv = \
            torch.nn.Unfold(kernel_size=(1, int(peak2peak_windows * self.data_info['sfreq']))).to(self.device)

        # step5: 使用非线性搜索，计算最优方向。
        # 在source grids维度使用chunk的方式减少显存的使用。
        max_ori = []
        for ied in range(ied_segment_samples.shape[0]):
            with torch.no_grad():
                max_ori_temp = torch.zeros((leadfield.shape[0], 3), dtype=torch.float64).to(self.device)
                # 使用chunk的方式减少显存的使用
                for chunk_idx in range(int(np.ceil(leadfield.shape[0] / chunk_number))):
                    vs_channel_idx = torch.arange(int(chunk_idx * chunk_number),
                                                  int(min(leadfield.shape[0], (chunk_idx + 1) * chunk_number)))
                    max_ori_temp[vs_channel_idx], _ = \
                        self.sam_costfun_peak2peak(data=ied_data_segments[ied],
                                                   leadfields=leadfield[vs_channel_idx],
                                                   inv_covariance=inv_covariance,
                                                   xyz_coords=xyz_coords,
                                                   baseline_samples=baseline_samples,
                                                   interest_samples=ied_samples,
                                                   xyz_coords_subs=xyz_coords_sub,
                                                   baseline_samples_sub=baseline_samples_sub,
                                                   interest_samples_sub=ied_samples_sub,
                                                   segment_data_conv=segment_data_conv)
                    self.del_var()
                max_ori.append(max_ori_temp)

        # 输出
        max_ori = torch.stack(max_ori, dim=0).unsqueeze(1)
        ied_window_samples = ied_samples[:1].repeat(max_ori.shape[0], 1)
        baseline_window_samples = baseline_samples[:1].repeat(max_ori.shape[0], 1)
        return max_ori, ied_segment_samples, ied_window_samples, baseline_window_samples

    def cal_hfo_vs_max_orientation(self, ied_peak_sample,
                                   ied_segment_time=2, hfo_window=0.1, hfo_segment=0.3, mean_power_windows=0.05,
                                   re_lambda=0.05, chunk_number=300):
        """
        Description:
            使用非线性搜索，计算hfo的源空间信号最优方向

        Input:
            :param ied_peak_sample: np.array, double, shape(n_ieds), in_device
                ied主peak的采样点
            :param ied_segment_time: number, double
                ied segment的时间窗长度，单位s
            :param hfo_window: number, double
                用于计算方向的hfo片段的时间窗长度，单位s
            :param hfo_segment: number, double
                对于一个ied，ied主峰值周围的时间窗长度(hfo可能出现在这个窗内)，单位s
            :param mean_power_windows: number, double
                计算平均包络的时间窗长度，单位s
            :param re_lambda: number, double
                求违逆时候的正则化系数
            :param chunk_number: number, long
                使用chunk的方式减少显存的使用，每个chunk的大小。

        Return:
            :return cov_whole_hfo: torch.tensor, double, shape(n_channels*n_channels)
                hfo频段数据的协方差矩阵
            :return inv_cov_whole_hfo: torch.tensor, double, shape(n_channels*n_channels)
                hfo频段数据的协方差矩阵逆矩阵
            :return max_ori: torch.tensor, double, shape(n_ieds*n_segment_samples*n_dipoles*3)
                最优源空间信号方向。
            :return ied_segment_samples: torch.tensor, double, shape(n_ieds*n_ied_segment_samples)
                每个ied的时间窗，在整段数据中的采样点位置。
            :return hfo_window_samples: torch.tensor, double, shape(n_ieds*n_segment_samples*n_hfo_samples)
                hfo时间窗在ied segment中的采样点位置
            :return baseline_window_samples: torch.tensor, double, shape(n_ieds*n_segment_samples*n_baseline_samples)
                感兴趣时间窗对应的baseline(去除hfo)，在ied segment中的采样点位置
        """

        # step1: 计算数据协方差矩阵
        _, _, cov_whole_hfo, inv_cov_whole_hfo = self.cal_data_covariance_using_whole_recording(
            raw_data_ied=None, raw_data_hfo=self.get_raw_data_hfo(), re_lambda=re_lambda)
        cov_whole_hfo = cov_whole_hfo.cpu()
        self.del_var()

        # step2: 计算源空间最优方向
        hfo_max_ori, ied_segment_samples, hfo_window_samples, baseline_window_samples = \
            self.cal_max_ori_using_hfo_windows(
                ied_peak_sample=ied_peak_sample, inv_covariance=inv_cov_whole_hfo,
                segment_time=ied_segment_time, hfo_window=hfo_window, hfo_segment=hfo_segment,
                mean_power_windows=mean_power_windows, chunk_number=chunk_number)
        inv_cov_whole_hfo = inv_cov_whole_hfo.cpu()
        hfo_max_ori, ied_segment_samples, hfo_window_samples, baseline_window_samples = \
            hfo_max_ori.cpu(), ied_segment_samples.cpu(), hfo_window_samples.cpu(), baseline_window_samples.cpu()
        self.del_var()

        return cov_whole_hfo, inv_cov_whole_hfo, hfo_max_ori, \
               ied_segment_samples, hfo_window_samples, baseline_window_samples

    def cal_ied_vs_max_orientation(self, ied_peak_sample,
                                   ied_segment_time=2, ied_window=0.15, ied_segment=0.15, peak2peak_windows=0.05,
                                   re_lambda=0.05, chunk_number=1500):
        """
        Description:
            使用非线性搜索，计算ied的源空间信号最优方向

        Input:
            :param ied_peak_sample: np.array, double, shape(n_ieds), in_device
                ied主peak的采样点
            :param ied_segment_time: number, double
                ied segment的时间窗长度，单位s
            :param ied_window: number, double
                用于计算方向的ied片段的时间窗长度，单位s
            :param ied_segment: number, double
                对于一个ied，ied主峰值周围的时间窗长度(ied可能出现在这个窗内)，单位s
            :param peak2peak_windows: number, double
                计算峰峰值的时间窗长度，单位s
            :param re_lambda: number, double
                求违逆时候的正则化系数
            :param chunk_number: number, long
                使用chunk的方式减少显存的使用，每个chunk的大小。

        Return:
            :return cov_whole_ied: torch.tensor, double, shape(n_channels*n_channels)
                ied频段数据的协方差矩阵
            :return inv_cov_whole_ied: torch.tensor, double, shape(n_channels*n_channels)
                ied频段数据的协方差矩阵逆矩阵
            :return max_ori: torch.tensor, double, shape(n_ieds*n_segment_samples*n_dipoles*3)
                最优源空间信号方向。
            :return ied_segment_samples: torch.tensor, double, shape(n_ieds*n_ied_segment_samples)
                每个ied的时间窗，在整段数据中的采样点位置。
            :return ied_window_samples: torch.tensor, double, shape(n_ieds*n_segment_samples*n_ied_samples)
                ied时间窗在ied segment中的采样点位置
            :return baseline_window_samples: torch.tensor, double, shape(n_ieds*n_segment_samples*n_baseline_samples)
                感兴趣时间窗对应的baseline(去除ied)，在ied segment中的采样点位置
        """

        # step1: 计算数据协方差矩阵
        cov_whole_ied, inv_cov_whole_ied, _, _ = self.cal_data_covariance_using_whole_recording(
            raw_data_ied=self.get_raw_data_ied(), raw_data_hfo=None, re_lambda=re_lambda)
        cov_whole_ied = cov_whole_ied.cpu()
        self.del_var()

        # step2: 计算源空间最优方向
        ied_max_ori, ied_segment_samples, ied_window_samples, baseline_window_samples = \
            self.cal_max_ori_using_ied_windows(
                ied_peak_sample=ied_peak_sample, inv_covariance=inv_cov_whole_ied,
                segment_time=ied_segment_time, ied_window=ied_window, ied_segment=ied_segment,
                peak2peak_windows=peak2peak_windows, chunk_number=chunk_number)
        inv_cov_whole_ied = inv_cov_whole_ied.cpu()
        ied_max_ori, ied_segment_samples, ied_window_samples, baseline_window_samples = \
            ied_max_ori.cpu(), ied_segment_samples.cpu(), ied_window_samples.cpu(), baseline_window_samples.cpu()
        self.del_var()

        return cov_whole_ied, inv_cov_whole_ied, ied_max_ori, \
               ied_segment_samples, ied_window_samples, baseline_window_samples

    def cal_vs_reconstruction(self, ied_peak_sample, re_lambda=0.05,
                              ied_segment_time=2, ied_segment=0.15, peak2peak_windows=0.03, ied_chunk_number=1500,
                              hfo_window=0.1, hfo_segment=0.2, mean_power_windows=0.05, hfo_chunk_number=250):

        # step0: 计算reduce rank后的lead field
        leadfield_reduce = self.cal_leadfield_reduce_rank(rank=2)
        leadfield_reduce = leadfield_reduce.cpu()
        leadfield = self.get_leadfield().cpu()

        # step1: 计算ied频段的源空间方向
        cov_whole_ied, inv_cov_whole_ied, ied_max_ori, _, ied_window_samples, _ = \
            self.cal_ied_vs_max_orientation(ied_peak_sample=ied_peak_sample,
                                            ied_segment_time=ied_segment_time, ied_window=ied_segment,
                                            ied_segment=ied_segment, peak2peak_windows=peak2peak_windows,
                                            re_lambda=re_lambda, chunk_number=ied_chunk_number)
        cov_whole_ied, inv_cov_whole_ied = cov_whole_ied.cpu(), inv_cov_whole_ied.cpu()
        ied_max_ori, ied_window_samples = ied_max_ori.cpu(), ied_window_samples.cpu()

        # step2: 计算hfo频段的源空间方向
        cov_whole_hfo, inv_cov_whole_hfo, hfo_max_ori, \
        ied_segment_samples, hfo_window_samples, baseline_window_samples_hfo = \
            self.cal_hfo_vs_max_orientation(ied_peak_sample=ied_peak_sample,
                                            ied_segment_time=ied_segment_time, hfo_window=hfo_window,
                                            hfo_segment=hfo_segment, mean_power_windows=mean_power_windows,
                                            re_lambda=re_lambda, chunk_number=hfo_chunk_number)
        cov_whole_hfo, inv_cov_whole_hfo = cov_whole_hfo.cpu(), inv_cov_whole_hfo.cpu()
        hfo_max_ori, hfo_window_samples = hfo_max_ori.cpu(), hfo_window_samples.cpu()
        ied_segment_samples, baseline_window_samples_hfo = ied_segment_samples.cpu(), baseline_window_samples_hfo.cpu()

        # step3: 计算噪声数据协方差矩阵
        cov_noise_ied, inv_cov_noise_ied, cov_noise_hfo, inv_cov_noise_hfo = \
            self.cal_noise_covariance(raw_data_ied=self.get_raw_data_ied(), raw_data_hfo=self.get_raw_data_hfo(),
                                      re_lambda=re_lambda, noise_win=10, ied_peak_sample=ied_peak_sample)
        cov_noise_ied, inv_cov_noise_ied = cov_noise_ied.cpu(), inv_cov_noise_ied.cpu()
        cov_noise_hfo, inv_cov_noise_hfo = cov_noise_hfo.cpu(), inv_cov_noise_hfo.cpu()

        # 计算到emhapp的输出
        emhapp_save = self.export_emhapp(
            leadfield_reduce=leadfield_reduce, leadfield=leadfield,
            cov_noise_ied=cov_noise_ied, inv_cov_noise_ied=inv_cov_noise_ied, cov_whole_ied=cov_whole_ied,
            inv_cov_whole_ied=inv_cov_whole_ied, ied_max_ori=ied_max_ori,
            cov_noise_hfo=cov_noise_hfo, inv_cov_noise_hfo=inv_cov_noise_hfo, cov_whole_hfo=cov_whole_hfo,
            inv_cov_whole_hfo=inv_cov_whole_hfo, hfo_max_ori=hfo_max_ori,
            hfo_window_samples=hfo_window_samples, baseline_window_samples_hfo=baseline_window_samples_hfo,
            ied_segment_samples=ied_segment_samples)

        return emhapp_save, ied_max_ori, inv_cov_whole_ied, hfo_max_ori, inv_cov_whole_hfo, \
               hfo_window_samples, baseline_window_samples_hfo, ied_segment_samples

    @staticmethod
    def export_emhapp(leadfield_reduce, leadfield,
                      cov_noise_ied, inv_cov_noise_ied, cov_whole_ied, inv_cov_whole_ied, ied_max_ori,
                      cov_noise_hfo, inv_cov_noise_hfo, cov_whole_hfo, inv_cov_whole_hfo, hfo_max_ori,
                      hfo_window_samples,
                      baseline_window_samples_hfo, ied_segment_samples):
        """
        Description:
            将结果添加到emhapp_save中

        Input:
            :param leadfield_reduce: ndarray, double, shape(n_dipoles*3*n_channels)
                reduce rank后的leadfield矩阵
            :param leadfield: ndarray, double, shape(n_dipoles*3*n_channels)
                leadfield矩阵
            :param cov_noise_ied: torch.tensor, double, shape(n_channels*n_channels), in_device
                ied频段噪声的协方差矩阵
            :param inv_cov_noise_ied: torch.tensor, double, shape(n_channels*n_channels), in_device
                ied频段噪声的协方差矩阵逆矩阵
            :param cov_noise_hfo: torch.tensor, double, shape(n_channels*n_channels), in_device
                hfo频段噪声的协方差矩阵
            :param inv_cov_noise_hfo: torch.tensor, double, shape(n_channels*n_channels), in_device
                hfo频段噪声的协方差矩阵逆矩阵
            :param cov_whole_hfo: torch.tensor, double, shape(n_channels*n_channels)
                hfo频段数据的协方差矩阵
            :param inv_cov_whole_hfo: torch.tensor, double, shape(n_channels*n_channels)
                hfo频段数据的协方差矩阵逆矩阵
            :param hfo_max_ori: torch.tensor, double, shape(n_ieds*n_segment_samples*n_dipoles*3)
                最优源空间信号方向。
            :param ied_segment_samples: torch.tensor, double, shape(n_ieds*n_ied_segment_samples)
                每个ied的时间窗，在整段数据中的采样点位置。
            :param hfo_window_samples: torch.tensor, double, shape(n_ieds*n_segment_samples*n_hfo_samples)
                hfo时间窗在ied segment中的采样点位置
            :param baseline_window_samples_hfo: torch.tensor, double, shape(n_ieds*n_segment_samples*n_baseline_samples)
                感兴趣时间窗对应的baseline(去除hfo)，在ied segment中的采样点位置
            :param cov_whole_ied: torch.tensor, double, shape(n_channels*n_channels)
                ied频段数据的协方差矩阵
            :param inv_cov_whole_ied: torch.tensor, double, shape(n_channels*n_channels)
                ied频段数据的协方差矩阵逆矩阵
            :param ied_max_ori: torch.tensor, double, shape(n_ieds*n_segment_samples*n_dipoles*3)
                最优源空间信号方向。
            :param ied_segment_samples: torch.tensor, double, shape(n_ieds*n_ied_segment_samples)
                每个ied的时间窗，在整段数据中的采样点位置。
        Return:
            :return emhapp_save: dict
                用于保存mat文件的字典
        """

        # 重构输出格式
        vs_recon_parameter = {'lf': leadfield.cpu().numpy(), 'lfReduce': leadfield_reduce.cpu().numpy(),
                              'Cov': cov_whole_hfo.numpy(), 'CovInv': inv_cov_whole_hfo.cpu().numpy(),
                              'noiseCov': cov_noise_hfo.cpu().numpy(), 'noiseCovInv': inv_cov_noise_hfo.cpu().numpy(),
                              'FixOir': [], 'MaxOir': hfo_max_ori.numpy(),
                              'Cov_S': cov_whole_ied.cpu().numpy(), 'CovInv_S': inv_cov_whole_ied.cpu().numpy(),
                              'noiseCov_S': cov_noise_ied.cpu().numpy(),
                              'noiseCovInv_S': inv_cov_noise_ied.cpu().numpy(),
                              'FixOir_S': [], 'MaxOir_S': ied_max_ori.cpu().numpy()}
        emhapp_save = {'ReconParam': vs_recon_parameter,
                       'SpikeTime': ied_segment_samples.cpu().numpy() + 1,
                       'HFO_Time': hfo_window_samples.cpu().numpy() + 1,
                       'BS_Time': baseline_window_samples_hfo.cpu().numpy() + 1,
                       'OriMethod': 1}

        return emhapp_save


def load_param_matlab(mat_file):
    param = scipy.io.loadmat(mat_file)['Param'][0]
    return param['bad_seg'][0][0] > 0, param['lf'][0], \
           param['SpikeTime'][0][0].astype('int64'), param['Resample'][0][0][0], \
           param['fifPath'][0][0], param['Lambda'][0][0][0], param['FreqRange'][0][0], \
           param['VolumeSurface'][0][0][0], param['voxelSize'][0][0][0], param['Surface_VertexNum'][0][0][0], \
           param['OriMethod'][0][0][0], param['CostFun'][0][0][0], \
           param['SignalWin'][0][0][0], param['HFOWin'][0][0][0], param['AllHFOWin'][0][0][0], \
           param['PowerPeakNum'][0][0][0], param['PowerWindows'][0][0][0], \
           param['VertexSim'][0][0][0], param['VertexDownSample'][0][0][0], param['Device'][0][0][0]


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(1)

    bad_seg, lf, SpikeTime, ReSample, fifPath, Lambda, FreqRange, \
    VolumeSurface, voxelSize, Surface_VertexNum, \
    OriMethod, CostFun, SignalWin, HFOWin, AllHFOWin, PowerPeakNum, PowerWindows, \
    VertexSim, VertexDownSample, Device = load_param_matlab(matFile)

    RAW = mne.io.read_raw_fif(fifPath, verbose='error', preload=True)
    Info = RAW.info
    RAW.resample(1000, n_jobs='cuda')
    RAW_data = RAW.get_data(picks='meg')
    RAW_data_ied = mne.filter.filter_data(RAW_data, RAW.info['sfreq'], 3, 80,
                                          fir_design='firwin', pad="reflect_limited", verbose=False, n_jobs='cuda')
    RAW_data_ied = mne.filter.notch_filter(RAW_data_ied, RAW.info['sfreq'], 50., verbose=False, n_jobs='cuda')
    RAW_data_hfo = mne.filter.filter_data(RAW_data, RAW.info['sfreq'], 80, 200,
                                          fir_design='firwin', pad="reflect_limited", verbose=False, n_jobs='cuda')

    vs = vs_recon(raw_data_ied=RAW_data_ied, raw_data_hfo=RAW_data_hfo, data_info=Info, leadfield=lf,
                  bad_segment=np.array(bad_seg), device=5)

    vs.cal_vs_reconstruction(SpikeTime, re_lambda=0.05, ied_segment_time=2, ied_segment=0.15,
                             peak2peak_windows=0.03, ied_chunk_number=1500,
                             hfo_window=0.1, hfo_segment=0.2, mean_power_windows=0.05,
                             hfo_chunk_number=250)
