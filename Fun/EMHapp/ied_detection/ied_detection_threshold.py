# -*- coding:utf-8 -*-
# @Time    : 2022/2/4
# @Author  : cuiwei
# @File    : ied_detection_threshold.py
# @Software: PyCharm
# @Script to:
#   - 对于单个MEG文件，使用阈值法检测IED

import os
import mne
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from ied_detection import ied_peak_feature
import matplotlib.pyplot as plt


class ied_detection_threshold:
    def __init__(self, raw_data=None, data_info=None, device=-1, n_jobs=10):
        """
        Description:
            初始化，使用GPU或者CPU。

        Input:
            :param raw_data: ndarray, double, shape(channel, samples)
                MEG滤波后数据
            :param data_info: dict
                MEG数据的信息, MNE读取的raw.info
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
        self.raw_data, self.data_info = raw_data, data_info
        self.data_gfp = None
        self.set_raw_data(raw_data)
        self.set_data_info(data_info)
        self.weighted_channel_number, self.weighted_channel_number_peaks, \
        self.data_gfp, self.gfp_amp_slope, self.gfp_peak_amp_slope, self.gfp_peak_index, self.ied_channel_number = \
            None, None, None, None, None, None, None
        # 获取MAG和GRAD通道的index
        self.chan_type = {'mag': torch.tensor(mne.pick_types(self.data_info, meg='mag')).to(self.device),
                          'grad': torch.tensor(mne.pick_types(self.data_info, meg='grad')).to(self.device)}

    def _check_cuda(self, device):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
        if device > -1:
            # Init MEN cuda
            try:
                mne.cuda.set_cuda_device(device)
                mne.utils.set_config('MNE_USE_CUDA', 'true')
                mne.cuda.init_cuda(verbose='error')
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

    def set_raw_data(self, raw_data=None):
        assert raw_data is not None
        self.raw_data = torch.tensor(raw_data)

    def get_raw_data(self):
        return self.raw_data

    def set_data_info(self, data_info=None):
        assert data_info is not None
        self.data_info = data_info

    def get_data_info(self):
        return self.data_info

    def set_data_gfp(self, data_gfp=None, gfp_amp_slope=None, gfp_peak_amp_slope=None, gfp_peak_index=None):
        self.data_gfp, self.gfp_amp_slope, self.gfp_peak_amp_slope, self.gfp_peak_index = \
            data_gfp.clone(), gfp_amp_slope.clone(), gfp_peak_amp_slope.clone(), gfp_peak_index.clone()

    def get_data_gfp(self):
        return self.data_gfp, self.gfp_amp_slope, self.gfp_peak_amp_slope, self.gfp_peak_index

    def cal_dat_gfp(self, mag_grad_ratio=0.05):
        """
        Description:
            计算MEG数据的GFP，并计算GPF的peaks，用于确定IED events的peak位置

        Input:
            :param mag_grad_ratio: number, double
                mag和grad的比值，用于rescale mag和grad到同一scale

        Return:
            :return data_gfp: torch.tensor, double, shape(1*n_sample)
                数据的GFP。
            :return gfp_peak_amp_slope: torch.tensor, double, shape(n_peaks)
                GFP的波峰值：0.75*幅度 + 0.25*斜率；仅在peak处赋值，其他位置为0
            :return gfp_amp_slope: torch.tensor, double, shape(n_sample)
                与gfp_peak_amp_slope类似，但是peak内所有点(上升边和下降边采样点)均被赋值
            :return gfp_peak_index: torch.tensor, double, shape(n_peaks)
                GFP的波峰值位置。
        """

        # 计算数据的GFP
        raw_data = self.get_raw_data().clone().to(self.device)
        raw_data[self.chan_type['grad']] = raw_data[self.chan_type['grad']] * mag_grad_ratio
        data_gfp = torch.sum(raw_data * raw_data, dim=0).unsqueeze(0)
        data_gfp_z = (data_gfp - data_gfp.mean()) / data_gfp.std()
        # 计算GFP的peaks：初始化class
        gfp_peaks = ied_peak_feature.cal_peak_feature(raw_data=data_gfp.cpu().numpy(), data_info=self.get_data_info(),
                                                      device=self.device_number, n_jobs=self.n_jobs)
        # 计算peaks
        signal_peaks = gfp_peaks.cal_signal_peaks(smooth_windows=0.015, smooth_iterations=2)
        # 计算peaks的sample位置
        gfp_peak_index = gfp_peaks.cal_peak_index(pos_peak_idx=signal_peaks['pos_peak_idx'],
                                                  neg_peak_idx=signal_peaks['neg_peak_idx'])[0]
        # 计算peaks的幅度
        gfp_peak_amp = gfp_peaks.cal_peak_amplitude(pos_peak_data=signal_peaks['pos_peak_data'],
                                                    neg_peak_data=signal_peaks['neg_peak_data'])[0]
        # 计算peaks的半高宽斜率
        gfp_half_peak_slope = \
            gfp_peaks.cal_half_peak_slope(pos_peak_data=signal_peaks['pos_peak_data'],
                                          neg_peak_data=signal_peaks['neg_peak_data'],
                                          pos_half_peak_idx_num=signal_peaks['pos_half_peak_idx_num'],
                                          neg_half_peak_idx_num=signal_peaks['neg_half_peak_idx_num'])[0]
        # 计算gpf峰值度量：0.75*幅度 + 0.25*斜率
        gfp_peak_amp_slope = 0.75 * gfp_peak_amp + 0.25 * gfp_half_peak_slope
        # 由于GPF是平方，只保留波峰，并删除data_gfp_z小于amp_threshold的波峰
        amp_threshold = np.percentile(data_gfp_z.cpu().numpy(), 80)
        gfp_peak_amp_slope = gfp_peak_amp_slope[torch.arange(1, gfp_peak_index.shape[0] - 1, 2)]
        gfp_peak_amp_slope[data_gfp_z[:, gfp_peak_index[
                                             torch.arange(1, gfp_peak_index.shape[0] - 1, 2)]][0] < amp_threshold] = 0
        gfp_amp_slope = torch.repeat_interleave(gfp_peak_amp_slope,
                                                gfp_peak_index[torch.arange(0, gfp_peak_index.shape[0], 2)].diff(),
                                                dim=0)
        gfp_peak_index = gfp_peak_index[torch.arange(1, gfp_peak_index.shape[0] - 1, 2)]
        gfp_peak_index = gfp_peak_index[data_gfp_z[:, gfp_peak_index][0] >= amp_threshold]

        self.del_var()
        return data_gfp.squeeze(), gfp_amp_slope.squeeze(), gfp_peak_amp_slope.squeeze(), gfp_peak_index.squeeze()

    def set_weighted_channel_number(self, weighted_channel_number=None, weighted_channel_number_peaks=None):
        self.weighted_channel_number, self.weighted_channel_number_peaks = weighted_channel_number.clone(), \
                                                                           weighted_channel_number_peaks.clone()

    def get_weighted_channel_number(self):
        return self.weighted_channel_number, self.weighted_channel_number_peaks

    def cal_weighted_channel_number(self, peak_amplitude=None, half_peak_slope=None, win_samples_of_picked_peak=None):
        """
        Description:
            IED的peak为：多通道同步出现IED的信号峰值(或者峰值附近)。
                          如果每个channel的IED信号之间有传播(不对齐)，计算的GFP的波峰会变宽，导致斜率变低，
                          从而不能和IED后面的慢波区分开来。
            因此我们定义了幅度加权的IED激活channel数目，方法如下：
                         (1).获得每个通道IED的peak位置，以及peak对应的峰峰值和半高宽斜率
                         (2).考虑到通道间的peak会有偏移，取peak周围10ms的时间窗，并赋值为0.75*amp + 0.25*slope。
                             生成shape(n_channel*n_sample)数组。
                         (3).在channel维度，将(2)中数据相加，获得shape(1*n_sample)的数组。
                         (4).对(3)中数据，计算peak位置

        Input:
            :param peak_amplitude: list, double, shape(n_channel*1)
                每个通道peak的峰峰值。
            :param half_peak_slope: list, double, shape(n_channel*1)
                每个通道peak的半高宽斜率。
            :param win_samples_of_picked_peak: np.array, bool, shape(n_channel*n_samples)
                peak点周围peak_win内的采样点，在原始数据(n_channel*n_sample)中对应位置

        Return:
            :return weighted_channel_number: torch.tensor, double, shape(n_sample)
                幅度加权的IED激活channel数目。
            :return weighted_channel_number_peaks: torch.tensor, double, shape(n_sample)
                和weighted_channel_number类似，但是只在weighted_channel_number中的波峰位置有值
        """
        # 复制输入变量，防止改变
        peak_amplitude, half_peak_slope, win_samples_of_picked_peak = \
            peak_amplitude.copy(),  half_peak_slope.copy(), win_samples_of_picked_peak.copy()

        # 判断peak_amplitude, half_peak_slope, win_samples_of_picked_peak是否存在
        if (peak_amplitude is None) or (half_peak_slope is None) or (win_samples_of_picked_peak is None):
            return None, None

        # 使用pad_sequence将list补零转换成数组，并在每个n_sample维度补0。
        # 因为win_samples_of_picked_peak中没有peaks的采样点值为-1，使用-1去索引，会返回最后补上的0。
        peak_amplitude, half_peak_slope = \
            pad_sequence([torch.cat([torch.tensor(x), torch.tensor([0])]).to(self.device)
                          for x in peak_amplitude]).t(), \
            pad_sequence([torch.cat([torch.tensor(x), torch.tensor([0])]).to(self.device)
                          for x in half_peak_slope]).t()
        win_samples_of_picked_peak = torch.tensor(win_samples_of_picked_peak).to(self.device)
        win_samples_of_picked_peak[torch.where(win_samples_of_picked_peak == -1)] = (peak_amplitude.shape[-1] - 1)
        # 获取peak_amplitude, half_peak_slope
        peak_amp_slope = peak_amplitude.take_along_dim(win_samples_of_picked_peak, dim=-1) * 0.75 + \
                         half_peak_slope.take_along_dim(win_samples_of_picked_peak, dim=-1) * 0.25
        # 计算weighted_channel_number
        weighted_channel_number = peak_amp_slope.sum(dim=0).unsqueeze(0)

        # 计算weighted_channel_number的peaks：初始化class
        chan_num = ied_peak_feature.cal_peak_feature(raw_data=weighted_channel_number.cpu().numpy(),
                                                     data_info=self.get_data_info(), device=self.device_number)
        # 计算peaks
        signal_peaks = chan_num.cal_signal_peaks(smooth_windows=0.015, smooth_iterations=2)
        # 如果没有找到peaks
        if len(signal_peaks['pos_peak_idx']) == 0:
            return None, None
        # 计算peaks的sample位置
        chan_num_peak_index = chan_num.cal_peak_index(pos_peak_idx=signal_peaks['pos_peak_idx'],
                                                      neg_peak_idx=signal_peaks['neg_peak_idx'])[0]
        # 由于weighted_channel_number大于0，只保留波峰
        chan_num_pos_index = chan_num_peak_index[torch.arange(1, chan_num_peak_index.shape[0] - 1, 2)]
        # 由于weighted_channel_number的波形是方波状，因此chan_num_pos_index选取平台的中间点
        temp = torch.cat([weighted_channel_number.squeeze(), torch.tensor([0]).repeat(100).to(self.device)])[
            (chan_num_pos_index.unsqueeze(-1) + torch.arange(0, 100).to(self.device))]
        # 获取每个平台的长度
        temp = torch.where(temp == temp[:, :1].repeat(1, 100))
        chan_num_pos_index = (temp[0].unique(return_counts=True)[1] / 2).long() + chan_num_pos_index
        # 计算weighted_channel_number_peaks：和weighted_channel_number类似，但是只在weighted_channel_number中的波峰位置有值
        weighted_channel_number_peaks = torch.zeros_like(weighted_channel_number)
        weighted_channel_number_peaks[:, chan_num_pos_index] = weighted_channel_number[:, chan_num_pos_index]

        self.del_var()
        return weighted_channel_number.squeeze(), weighted_channel_number_peaks.squeeze()

    def set_channel_number(self, ied_channel_number=None):
        self.ied_channel_number = ied_channel_number.clone()

    def get_channel_number(self):
        return self.ied_channel_number

    def cal_channel_number(self, all_samples_of_picked_peak=None):
        """
        Description:
            计算每个时间点，有几个channel被检测为IED。

        Input:
            :param all_samples_of_picked_peak: number, double
                peak内所有点(上升边和下降边采样点)，在原始数据(n_channel*n_sample)中对应位置

        Return:
            :return IED_channel_number: torch.tensor, double, shape(1*n_sample)
                对于每个时间点，被检测为IED的channel个数。
        """
        # 复制输入变量，防止改变
        all_samples_of_picked_peak = all_samples_of_picked_peak.copy()

        # 判断all_samples_of_picked_peak是否存在
        if all_samples_of_picked_peak is None:
            return None

        all_samples_of_picked_peak = torch.tensor(all_samples_of_picked_peak).to(self.device)
        ied_channel_number = (all_samples_of_picked_peak != -1).long().sum(dim=0)

        self.del_var()
        return ied_channel_number

    def cal_ieds_peak_in_all_data(self, chan_threshold=(5, 200), ied_win_length=0.2,
                                  ied_channel_number=None, weighted_channel_number_peaks=None,
                                  data_gfp=None, gfp_peak_index=None):
        """
        Description:
            所有数据中的IED events:
            计算流程:
                    (1). 获取满足chan_threshold阈值的sample
                    (2). 连续的sample组成event
                    (3). 间隔小于50ms的event组合在一起
                    (4). 计算ied event的peak和window

        Input:
            :param chan_threshold: list, long, shape(2)
                需要满足的channel个数，每个时间内检测为IED的channel个数。channel的上限和下限。
            :param ied_win_length: number, double
                输出ieds的时间窗长度，单位s
            :param ied_channel_number: torch.tensor, double, shape(n_sample)
                对于每个时间点，被检测为IED的channel个数。
            :param weighted_channel_number_peaks: torch.tensor, double, shape(n_sample)
                幅度加权的IED激活channel数目，仅在weighted_channel_number中的波峰位置有值
            :param data_gfp: torch.tensor, double, shape(n_sample)
                GFP
            :param gfp_peak_index: torch.tensor, long, shape(n_sample)
                GFP的波峰位置

        Return:
            :return ieds_peak: torch.tensor, long, shape(n)
                ieds的peak位置
            :return ieds_win: torch.tensor, long, shape(n*ied_win_length)
                ieds events的duration
        """
        # 复制输入变量，防止改变
        ied_channel_number, weighted_channel_number_peaks, data_gfp, gfp_peak_index = \
            ied_channel_number.clone(), weighted_channel_number_peaks.clone(), data_gfp.clone(), gfp_peak_index.clone()

        # 判断输入是否存在
        if (weighted_channel_number_peaks is None) or (ied_channel_number is None) or \
                (data_gfp is None) or (gfp_peak_index is None):
            return None, None

        ied_win_half = int(ied_win_length * self.data_info['sfreq'] / 2)
        # 根据ied_channel_number，获得满足阈值chan_threshold的IED采样点
        is_ied_index = ((ied_channel_number >= chan_threshold[0]) & (ied_channel_number < chan_threshold[1])).float()
        # 如果相邻IED采样点之间时间少于50ms，认为这些从采样点是IED采样点
        ied_index = torch.where(is_ied_index)[0]
        ied_index = [torch.tensor(range(ied_index[x], ied_index[x + 1]))
                     for x in torch.where((ied_index[1:] - ied_index[:-1] <= 0.05 * self.data_info['sfreq']) &
                                          (ied_index[1:] - ied_index[:-1] > 1))[0]]
        if len(ied_index) > 0:
            is_ied_index[torch.cat(ied_index)] = 1.
        is_ied_index = torch.cat([torch.tensor([0]).to(self.device), is_ied_index, torch.tensor([0]).to(self.device)])
        # 将连续的IED采样点组合成，IED events时间窗
        ied_events = torch.stack((torch.where(is_ied_index.diff() == 1)[0],
                                  torch.where(is_ied_index.diff() == -1)[0])).t()

        # 计算IED events的ieds_peak和ieds_win
        ieds_peak, ieds_win = [], []
        for ied_event in ied_events:
            candidate_ied_win_center = ied_event.float().mean().long().cpu().tolist()
            candidate_ied_win_beg = max(0, candidate_ied_win_center - ied_win_half) - \
                                    ((candidate_ied_win_center + ied_win_half) -
                                     min(data_gfp.shape[0], candidate_ied_win_center + ied_win_half))
            candidate_ied_win_end = min(data_gfp.shape[0], candidate_ied_win_center + ied_win_half) + \
                                    max(0, candidate_ied_win_center - ied_win_half) - \
                                    (candidate_ied_win_center - ied_win_half)
            candidate_ied_win = np.arange(candidate_ied_win_beg, candidate_ied_win_end)
            ied_peak, ied_win = self.cal_ieds_peak_in_ieds_windows(candidate_ied_win=candidate_ied_win,
                                                                   weighted_channel_number=weighted_channel_number_peaks,
                                                                   data_gfp=data_gfp,
                                                                   gfp_peak_index=gfp_peak_index)
            if (ied_peak is not None) and (ied_win is not None):
                ieds_peak.append(ied_peak)
                ieds_win.append(ied_win)

        # 输出
        if len(ieds_peak) > 0:
            ieds_peak, ieds_win = torch.tensor(ieds_peak), torch.stack(ieds_win)
        else:
            ieds_peak, ieds_win = None, None

        self.del_var()
        return ieds_peak, ieds_win

    def cal_peaks_closest_to_ieds_peaks(self, ieds_peak=None, ieds_win=None, samples_of_picked_peak=None):
        """
        Description:
            对于每个ied event中每个通道，计算ied peak周围两侧的peak位置，用于数据片段的shift

        Input:
            :param ieds_peak: torch.tensor, long, shape(n)
                ieds的peak位置
            :param ieds_win: torch.tensor, long, shape(n*n_ied_win)
                ieds events的duration
            :param samples_of_picked_peak: torch.tensor, long, shape(n_channel*n_sample)
                peak点，在原始数据(n_channel*n_sample)中对应位置

        Return:
            :return ieds_peak_closest_idx: torch.tensor, double, shape(n*n_channel*2)
                每个通道中，ied peak两侧最近的峰，在raw meg中的采样点位置
        """
        # 判断输入是否存在
        if (ieds_peak is None) or (ieds_win is None) or (samples_of_picked_peak is None):
            return None

        # 复制输入变量，防止改变
        samples_of_picked_peak = samples_of_picked_peak.clone()

        ied_win_half = int(ieds_win[0].shape[0] * self.data_info['sfreq'] / 2 / 1000)
        n_channel = samples_of_picked_peak.shape[0]
        samples_of_picked_peak = samples_of_picked_peak.to(self.device)

        # 计算每个通道，左边距离ied_peak最近的peak，右边距离ied_peak最近的peak。没有peak返回n_sample
        # 使用矩阵方式：samples_of_picked_peak中存储每个channel中的peak的位置，采样点如果不是peak则赋值为max_peaks。
        # 因此对于每个ied event:
        #   (1). left: 提取ied_peak-ied_win_half到ied_peak之间的samples_of_picked_peak数据, shape(n_channel, ied_win_half)
        #   (2). right: 提取ied_peak到ied_peak+ied_win_half之间的samples_of_picked_peak数据, shape(n_channel, ied_win_half)
        #   (3). left中最大值为左边距离ied_peak最近的peak，right中最小值为左边距离ied_peak最近的peak。
        # 扩展samples_of_picked_peak，防止超出范围
        samples_of_picked_peak_temp = torch.cat([
            torch.tensor([[-1]]).repeat(n_channel, ied_win_half).to(self.device),
            samples_of_picked_peak,
            torch.tensor([[-1]]).repeat(n_channel, ied_win_half).to(self.device)], dim=1)
        # 获取ied_peak-ied_win_half到ied_peak之间，以及ied_peak到ied_peak+ied_win_half之间的index
        ieds_peak_ex = ieds_peak.to(self.device) + ied_win_half
        ieds_peak_left_idx = ieds_peak_ex.unsqueeze(-1) + torch.arange(-ied_win_half, 0).to(self.device)
        ieds_peak_right_idx = ieds_peak_ex.unsqueeze(-1) + torch.arange(0, ied_win_half).to(self.device)
        # ieds_peak_left中最大值为左边距离ied_peak最近的peak
        ieds_peak_left = samples_of_picked_peak_temp.unsqueeze(0).take_along_dim(
            ieds_peak_left_idx.unsqueeze(1).repeat(1, samples_of_picked_peak.shape[0], 1), dim=-1)
        ieds_peak_left_closest_idx = ieds_peak_left.argmax(dim=-1)
        ieds_peak_left_closest_idx[ieds_peak_left.sum(dim=-1) == -ied_win_half] = -10000000000
        # ieds_peak_right中最小值为左边距离ied_peak最近的peak，
        ieds_peak_right = samples_of_picked_peak_temp.unsqueeze(0).take_along_dim(
            ieds_peak_right_idx.unsqueeze(1).repeat(1, samples_of_picked_peak.shape[0], 1), dim=-1)
        # 将ieds_peak_right中-1赋值为
        ieds_peak_right[ieds_peak_right == -1] = samples_of_picked_peak_temp.max() + 1
        ieds_peak_right_closest_idx = ieds_peak_right.argmin(dim=-1)
        ieds_peak_right_closest_idx[ieds_peak_right.sum(dim=-1) ==
                                    (samples_of_picked_peak_temp.max() + 1) * ied_win_half] = -10000000000
        # 判断通道内是否存在peak，以及ied_peak左右是否存在peak，并将不存在的通道赋值为n_sample
        ieds_peak_closest_idx = torch.stack([ieds_peak_left_closest_idx - ied_win_half,
                                             ieds_peak_right_closest_idx], dim=-1) + \
                                ieds_peak.to(self.device).reshape(-1, 1, 1)
        ieds_peak_closest_idx[ieds_peak_closest_idx < 0] = -1

        self.del_var()
        return ieds_peak_closest_idx

    def cal_ieds_peaks_features(self, samples_of_picked_peak=None, ieds_peak_closest_idx=None,
                                peak_amplitude=None, half_peak_slope=None, peak_duration=None, peak_sharpness=None):
        """
        Description:
            对于每个ied event中每个通道，计算ied peak周围两侧peak的feature
            用于提取最大特征的segment，用作IED template的计算
            如果peak_amplitude, half_peak_slope, peak_duration, peak_sharpness, gfp_amp_slope为None, 对应特征返回为None

        Input:
            :param samples_of_picked_peak: torch.tensor, long, shape(n_channel*n_sample)
                peak点，在原始数据(n_channel*n_sample)中对应位置
            :param ieds_peak_closest_idx: torch.tensor, double, shape(n*n_channel*2)
                每个通道中，ied peak两侧最近的峰，在raw meg中的采样点位置
            :param peak_amplitude: list, double, shape(n_channel)
                每个channel peak的幅度
            :param half_peak_slope: list, double, shape(n_channel)
                每个channel peak的半高宽斜率
            :param peak_duration: list, long, shape(n_channel)
                每个channel peak的持续时间
            :param peak_sharpness: list, double, shape(n_channel)
                每个channel peak的锐度

        Return:
            :return ied_amplitude_peaks: torch.tensor, double, shape(n*n_channel*2)
                ieds周围peaks的幅度
            :return ied_half_slope_peaks: torch.tensor, double, shape(n*n_channel*2)
                ieds周围peaks的斜率
            :return ied_sharpness_peaks: torch.tensor, double, shape(n*n_channel*2)
                ieds周围peaks的锐度
            :return ied_duration_peaks: torch.tensor, double, shape(n*n_channel*2)
                ieds周围peaks的峰持续时间
        """

        # 判断输入是否存在
        if (samples_of_picked_peak is None) or (ieds_peak_closest_idx is None):
            return None, None, None, None
        if (peak_amplitude is None) and (half_peak_slope is None) and (peak_duration is None) and \
                (peak_sharpness is None):
            return None, None, None, None

        # 复制输入变量，防止改变
        ieds_peak_closest_idx, samples_of_picked_peak = ieds_peak_closest_idx.clone(), samples_of_picked_peak.clone()

        # 计算IED的特征
        n_channel = samples_of_picked_peak.shape[0]
        # 使用pad_sequence将list补零转换成数组，并在每个n_sample维度补0。
        # 因为samples_of_picked_peak中没有peaks的采样点值为-1，使用-1去索引，会返回最后补上的0。
        if peak_amplitude is not None:
            peak_amplitude = pad_sequence([torch.cat([torch.tensor(x).to(self.device),
                                                      torch.tensor([0]).to(self.device)])
                                           for x in peak_amplitude]).t()
            max_peaks = peak_amplitude.shape[-1] - 1
        if half_peak_slope is not None:
            half_peak_slope = pad_sequence([torch.cat([torch.tensor(x).to(self.device),
                                                       torch.tensor([0]).to(self.device)])
                                            for x in half_peak_slope]).t()
            max_peaks = half_peak_slope.shape[-1] - 1
        if peak_duration is not None:
            peak_duration = pad_sequence([torch.cat([torch.tensor(x).to(self.device),
                                                     torch.tensor([0]).to(self.device)])
                                          for x in peak_duration]).t()
            max_peaks = peak_duration.shape[-1] - 1
        if peak_sharpness is not None:
            peak_sharpness = pad_sequence([torch.cat([torch.tensor(x).to(self.device),
                                                      torch.tensor([0]).to(self.device)])
                                           for x in peak_sharpness]).t()
            max_peaks = peak_sharpness.shape[-1] - 1
        # 获取samples_of_picked_peak每个采样点对应的IED peak索引。
        samples_of_picked_peak = samples_of_picked_peak.to(self.device)
        samples_of_picked_peak[torch.where(samples_of_picked_peak == -1)] = max_peaks

        # 获取每个通道中，ied peak两侧的peaks在所有peaks中的index
        ieds_peak_closest_idx[ieds_peak_closest_idx == -1] = samples_of_picked_peak.shape[1]
        samples_of_picked_peak_temp = torch.cat([samples_of_picked_peak,
                                                 torch.tensor([[max_peaks]]).repeat(n_channel, 1).to(self.device)],
                                                dim=1)
        closest_peak_index = samples_of_picked_peak_temp.unsqueeze(0).take_along_dim(ieds_peak_closest_idx, dim=-1)

        # 对于每个IEDs event提取前chan_threshold[0]个feature的平均值，作为该IED event的特征
        if peak_amplitude is not None:
            ied_amplitude_peaks = peak_amplitude.unsqueeze(0).take_along_dim(closest_peak_index, dim=-1)
        else:
            ied_amplitude_peaks = None
        if half_peak_slope is not None:
            ied_half_slope_peaks = half_peak_slope.unsqueeze(0).take_along_dim(closest_peak_index, dim=-1)
        else:
            ied_half_slope_peaks = None
        if peak_sharpness is not None:
            ied_sharpness_peaks = peak_sharpness.unsqueeze(0).take_along_dim(closest_peak_index, dim=-1)
        else:
            ied_sharpness_peaks = None
        if peak_duration is not None:
            ied_duration_peaks = peak_duration.unsqueeze(0).take_along_dim(closest_peak_index, dim=-1)
        else:
            ied_duration_peaks = None

        self.del_var()
        return ied_amplitude_peaks, ied_half_slope_peaks, ied_sharpness_peaks, ied_duration_peaks

    def _cal_ieds_features(self, averaged_chan_num=5, ieds_peak=None, data_gfp=None, ied_amplitude_peaks=None,
                           ied_half_slope_peaks=None, ied_sharpness_peaks=None, ied_duration_peaks=None):
        """
        Description:
            对于每个ied event中每个通道，计算ied peak周围两侧peak的feature
            如果peak_amplitude, half_peak_slope, peak_duration, peak_sharpness, gfp_amp_slope为None, 对应特征返回为None

        Input:
            :param averaged_chan_num: number, long
                前averaged_chan_num个channel的feature，平均后得到IED feature
            :param ieds_peak: torch.tensor, long, shape(n)
                ieds的peak位置
            :param data_gfp: torch.tensor, double, shape(n_sample)
                GFP
            :param ied_amplitude_peaks: torch.tensor, double, shape(n*n_channel*2)
                ieds周围peaks的幅度
            :param ied_half_slope_peaks: torch.tensor, double, shape(n*n_channel*2)
                ieds周围peaks的斜率
            :param ied_sharpness_peaks: torch.tensor, double, shape(n*n_channel*2)
                ieds周围peaks的锐度
            :param ied_duration_peaks: torch.tensor, double, shape(n*n_channel*2)
                ieds周围peaks的峰持续时间

        Return:
            :return ied_amplitude: torch.tensor, double, shape(n)
                ieds的幅度
            :return ied_half_slope: torch.tensor, double, shape(n)
                ieds的斜率
            :return ied_sharpness: torch.tensor, double, shape(n)
                ieds的锐度
            :return ied_duration: torch.tensor, double, shape(n)
                ieds的峰持续时间
            :return ied_gfp_amp: torch.tensor, double, shape(n)
                ieds的gfp值
        """

        # 对于每个IED获得feature值
        if ied_amplitude_peaks is not None:
            ied_amplitude = ied_amplitude_peaks.amax(dim=-1).topk(averaged_chan_num, dim=-1)[0].mean(dim=-1)
        else:
            ied_amplitude = None
        if ied_half_slope_peaks is not None:
            ied_half_slope = ied_half_slope_peaks.amax(dim=-1).topk(averaged_chan_num, dim=-1)[0].mean(dim=-1)
        else:
            ied_half_slope = None
        if ied_sharpness_peaks is not None:
            ied_sharpness = ied_sharpness_peaks.amax(dim=-1).topk(averaged_chan_num, dim=-1)[0].mean(dim=-1)
        else:
            ied_sharpness = None
        if (ied_amplitude_peaks is not None) and (ied_duration_peaks is not None):
            # 根据amplitude选出peak，将其duration进行平均
            ied_duration_temp = ied_duration_peaks.take_along_dim(
                ied_amplitude_peaks.argmax(dim=-1).unsqueeze(-1), dim=-1)[:, :, 0].float()
            ied_duration = ied_duration_temp.take_along_dim(
                ied_amplitude_peaks.amax(dim=-1).topk(averaged_chan_num, dim=-1)[1], dim=-1).mean(dim=-1).long()
        else:
            ied_duration = None
        if (data_gfp is not None) and (ieds_peak is not None):
            ied_gfp_amp = data_gfp[ieds_peak]
        else:
            ied_gfp_amp = None

        self.del_var()
        return ied_amplitude, ied_half_slope, ied_sharpness, ied_duration, ied_gfp_amp

    def cal_data_segments_with_shift(self, data_segment_window=0.1, ieds_peak_closest_idx=None):
        """
        Description:
            根据ieds_peak_closest_idx截取数据片段：ied个数*通道数目*shift个数

        Input:
            :param data_segment_window: number, double
                输出片段的长度，单位s
            :param ieds_peak_closest_idx: torch.tensor, double, shape(n*n_channel*shift个数)
                每个通道中，ied peak两侧最近的峰，在raw meg中的采样点位置

        Return:
            :return data_segment: torch.tensor, double, shape(n*n_channel*shift个数)
                数据片段
        """

        # 判断输入是否存在
        if ieds_peak_closest_idx is None:
            return None

        # 复制输入变量，防止改变
        ieds_peak_closest_idx = ieds_peak_closest_idx.clone()

        # 获取每个IEDs的数据片段，包含每个shift的数据片段
        half_win = int(self.data_info['sfreq'] * data_segment_window / 2)
        ieds_peak_closest_idx[torch.where(ieds_peak_closest_idx == -1)] = self.raw_data.shape[1] + half_win * 2
        ieds_peak_closest_idx = ieds_peak_closest_idx + half_win
        # 获取数据切片的windows
        ieds_peak_closest_idx_win = \
            torch.stack([ieds_peak_closest_idx[:, :, :1] +
                         torch.arange(-half_win, half_win).reshape(1, 1, -1).to(self.device),
                         ieds_peak_closest_idx[:, :, -1:] +
                         torch.arange(-half_win, half_win).reshape(1, 1, -1).to(self.device)], dim=2)
        # 扩展raw_date: (1). 防止因为window超出数据范围。(2). 将没有peak的数据片段赋值为0
        raw_data_ex = torch.cat([self.raw_data[:, :1000].mean(dim=-1, keepdim=True).repeat(1, half_win), self.raw_data,
                                 self.raw_data[:, -1000:].mean(dim=-1, keepdim=True).repeat(1, half_win),
                                 torch.zeros(self.raw_data.shape[0], half_win * 2)], dim=1).to(self.device)
        data_segment = raw_data_ex.unsqueeze(0).unsqueeze(2).take_along_dim(ieds_peak_closest_idx_win, dim=-1)

        self.del_var()
        return data_segment

    def cal_data_with_large_feature(self, ied_amplitude_peaks=None, large_peak_amplitude_threshold=None,
                                    ied_half_slope_peaks=None, large_peak_half_slope_threshold=None,
                                    ied_sharpness_peaks=None, large_peak_sharpness_threshold=None,
                                    ied_duration_peaks=None, large_peak_duration_threshold=None):
        """
        Description:
            根据特征和阈值，选取输出片段中的高特征值片段

        Input:
            :param ied_amplitude_peaks: torch.tensor, double, shape(n*n_channel*shift个数)
                ieds周围peaks的幅度
            :param large_peak_amplitude_threshold: number, double
                幅度阈值, 范围(0, 1)之间，数据中ied_amplitude_peaks_threshold的位置
            :param ied_half_slope_peaks: torch.tensor, double, shape(n*n_channel*shift个数)
                ieds周围peaks的斜率
            :param large_peak_half_slope_threshold: number, double
                斜率阈值, 范围(0, 1)之间，数据中ied_half_slope_peaks_threshold的位置
            :param ied_sharpness_peaks: torch.tensor, double, shape(n*n_channel*shift个数)
                ieds周围peaks的锐度
            :param large_peak_sharpness_threshold: number, double
                锐度阈值, 范围(0, 1)之间，数据中ied_sharpness_peaks_threshold的位置
            :param ied_duration_peaks: torch.tensor, double, shape(n*n_channel*shift个数)
                ieds周围peaks的峰持续时间
            :param large_peak_duration_threshold: number, long
                duration阈值，单位s

        Return:
            :return ieds_peaks_large_idx: torch.tensor, double, shape(n*n_channel)
                每个片段是否满足阈值要求
        """
        # 复制输入变量，防止改变
        ied_amplitude_peaks, ied_half_slope_peaks, ied_sharpness_peaks, ied_duration_peaks = \
            ied_amplitude_peaks.clone(), ied_half_slope_peaks.clone(), \
            ied_sharpness_peaks.clone(), ied_duration_peaks.clone()

        # 判断输入是否存在
        if ied_amplitude_peaks is None:
            return None, None
        if (ied_half_slope_peaks is None) and (ied_sharpness_peaks is None) and (ied_duration_peaks is None) and \
                (ied_amplitude_peaks is None):
            return None, None

        # 计算每个channel的feature，以及阈值.
        ieds_peaks_large_idx = torch.tensor([[True]]).repeat(ied_amplitude_peaks.shape[0],
                                                             ied_amplitude_peaks.shape[1]).to(self.device)
        max_amplitude_idx = ied_amplitude_peaks.argmax(dim=-1).unsqueeze(-1)
        # 满足amplitude阈值
        if (large_peak_amplitude_threshold is not None) and (ied_amplitude_peaks is not None):
            ied_amplitude_peaks_max = ied_amplitude_peaks.amax(dim=-1)
            ied_amplitude_peaks_max_temp = ied_amplitude_peaks_max[ied_amplitude_peaks_max > 0]
            large_peak_amplitude_threshold = ied_amplitude_peaks_max_temp.topk(
                int(ied_amplitude_peaks_max_temp.shape[0] * (1 - large_peak_amplitude_threshold)))[0][-1]
            ieds_peaks_large_idx = ieds_peaks_large_idx & (ied_amplitude_peaks_max >= large_peak_amplitude_threshold)
        # 满足half_slope阈值
        if (large_peak_half_slope_threshold is not None) and (ied_half_slope_peaks is not None):
            ied_half_slope_peaks_max = ied_half_slope_peaks.take_along_dim(max_amplitude_idx, dim=-1).squeeze()
            ied_half_slope_peaks_max_temp = ied_half_slope_peaks_max[ied_half_slope_peaks_max > 0]
            large_peak_half_slope_threshold = ied_half_slope_peaks_max_temp.topk(
                int(ied_half_slope_peaks_max_temp.shape[0] * (1 - large_peak_half_slope_threshold)))[0][-1]
            ieds_peaks_large_idx = ieds_peaks_large_idx & (ied_half_slope_peaks_max >= large_peak_half_slope_threshold)
        # 满足sharpness阈值
        if (large_peak_sharpness_threshold is not None) and (ied_sharpness_peaks is not None):
            ied_sharpness_peaks_max = ied_sharpness_peaks.take_along_dim(max_amplitude_idx, dim=-1).squeeze()
            ied_sharpness_peaks_max_temp = ied_sharpness_peaks_max[ied_sharpness_peaks_max > 0]
            large_peak_sharpness_threshold = ied_sharpness_peaks_max_temp.topk(
                int(ied_sharpness_peaks_max_temp.shape[0] * (1 - large_peak_sharpness_threshold)))[0][-1]
            ieds_peaks_large_idx = ieds_peaks_large_idx & (ied_sharpness_peaks_max >= large_peak_sharpness_threshold)
        # 满足duration阈值
        if (large_peak_duration_threshold is not None) and (ied_duration_peaks is not None):
            ied_duration_peaks_max = ied_duration_peaks.take_along_dim(max_amplitude_idx, dim=-1).squeeze()
            ieds_peaks_large_idx = ieds_peaks_large_idx & (ied_duration_peaks_max <=
                                                           int(large_peak_duration_threshold * self.data_info['sfreq']))

        self.del_var()
        return ieds_peaks_large_idx

    @staticmethod
    def ied_validate(candidate_ied_win=None, ied_channel_number=None, chan_threshold=(5, 200)):
        """
        Description:
            判断candidate_ieds_win中是否包含满足chan_threshold的ieds。

        Input:
            :param candidate_ied_win: np.array, long, shape(n_time)
                需要判断的ied时间段，在原始MEG数据中的采样点位置。
            :param ied_channel_number: torch.tensor, double, shape(n_sample)
                对于每个时间点，被检测为IED的channel个数。
            :param chan_threshold: list, long, shape(2); or number, long
                需要满足的channel个数，每个时间内检测为IED的channel个数。
                为list的时候，channel的上限和下限；为number的时候，channel下限。

        Return:
            :return is_ied: bool
                是否有ied
        """
        # 复制输入变量，防止改变
        candidate_ied_win, ied_channel_number = candidate_ied_win.copy(), ied_channel_number.clone()

        # 判断输入是否存在
        if (candidate_ied_win is None) or (ied_channel_number is None):
            return None

        # 判断candidate_ieds_win内是检测到的IEDs, 是否满足chan_threshold要求
        temp = ied_channel_number[torch.tensor(candidate_ied_win)]
        if len(chan_threshold) == 0:
            is_ied = None
        elif len(chan_threshold) == 1:
            is_ied = len(torch.where(temp >= chan_threshold)[0]) > 0
        elif len(chan_threshold) == 2:
            is_ied = len(torch.where((temp >= chan_threshold[0]) & (temp <= chan_threshold[1]))[0]) > 0

        return is_ied

    def cal_ieds_peak_in_ieds_windows(self, candidate_ied_win=None, weighted_channel_number=None,
                                      data_gfp=None, gfp_peak_index=None, gfp_search_win=0.05):
        """
        Description:
            计算ied_windows中的peak位置，并选取peak周围的平均GFP幅度最大的时间窗为最终的ied events
            peak位置选取：(1) 选取weighted_channel_number在candidate_ied_win内的最大值位置。
                        (2) 最大值位置附近的gfp amp最大位置为最终ied的peak(gfp_search_win内，gfp_amp_slope值最大的)

        Input:
            :param candidate_ied_win: np.array, long, shape(n_time)
                需要判断的ied时间段，在原始MEG数据中的采样点位置。
            :param weighted_channel_number: torch.tensor, double, shape(n_sample)
                幅度加权的IED激活channel数目。
            :param data_gfp: torch.tensor, double, shape(n_sample)
                GFP
            :param gfp_peak_index: torch.tensor, long, shape(n_sample)
                GFP的波峰位置
            :param gfp_search_win: number, double
                GFP的波峰位置

        Return:
            :return ieds_peak: number, long
                ieds的peak位置
            :return ieds_peak_win: torch.tensor, long, shape(1*candidate_ied_win.shape[0])
                ieds events的duration
        """
        # 复制输入变量，防止改变
        candidate_ied_win, weighted_channel_number, data_gfp, gfp_peak_index =\
            candidate_ied_win.copy(), weighted_channel_number.clone(), data_gfp.clone(), gfp_peak_index.clone()

        # 判断输入是否存在
        if (candidate_ied_win is None) or (weighted_channel_number is None) or \
                (data_gfp is None) or (gfp_peak_index is None):
            return None, None

        # 选取weighted_channel_number在candidate_ied_win内的最大值位置
        half_ieds_window = int(candidate_ied_win.shape[0] / 2)
        weighted_chan_num_win = weighted_channel_number[candidate_ied_win]
        highest_weighted_chan_time = weighted_chan_num_win.argmax() + candidate_ied_win[0]

        # 最大值位置附近的gfp peak最大值为最终ied的peak
        gfp_search_win = int(gfp_search_win * self.data_info['sfreq'])
        for dist in range(0, gfp_search_win, 10):
            ieds_candidate_peak = gfp_peak_index[
                torch.where((gfp_peak_index >= highest_weighted_chan_time - gfp_search_win) &
                            (gfp_peak_index <= highest_weighted_chan_time + gfp_search_win))]
            if len(ieds_candidate_peak) > 0:
                break
        if len(ieds_candidate_peak) < 1:
            return None, None
        ieds_peak = ieds_candidate_peak[data_gfp[ieds_candidate_peak].argmax()]
        # ieds_candidate_peak = gfp_peak_index[(-(gfp_peak_index - highest_weighted_chan_time).abs()).topk(2)[1]]
        # ieds_peak = ieds_candidate_peak[gfp_amp_slope[ieds_candidate_peak].argmax()]
        # if (ieds_peak < candidate_ied_win[0]) or (ieds_peak > candidate_ied_win[-1]):
        #     return None, None

        # 选取ied peak周围的平均GFP幅度最大的时间窗为最终的ied时间窗
        search_win_beg = torch.arange(ieds_peak + int(half_ieds_window / 2) - half_ieds_window * 2,
                                      ieds_peak - int(half_ieds_window / 2))
        search_win_beg = search_win_beg[torch.where((search_win_beg >= half_ieds_window * 2) &
                                                    (search_win_beg < (data_gfp.shape[0] - half_ieds_window * 2)))]
        if len(search_win_beg) == 0:
            ieds_peak_win_beg = max(0, ieds_peak - half_ieds_window) - \
                                ((ieds_peak + half_ieds_window) - min(data_gfp.shape[0],
                                                                      ieds_peak + half_ieds_window))
            ieds_peak_win_end = min(data_gfp.shape[0], ieds_peak + half_ieds_window) + \
                                max(0, ieds_peak - half_ieds_window) - (ieds_peak - half_ieds_window)
            ieds_peak_win = torch.arange(ieds_peak_win_beg, ieds_peak_win_end).to(self.device)
        else:
            search_win = search_win_beg.reshape(-1, 1) + torch.arange(0, half_ieds_window * 2).reshape(1, -1)
            ieds_peak_win = search_win[data_gfp[search_win].sum(dim=-1).argmax()].to(self.device)
        ieds_peak_win = ieds_peak_win - ieds_peak

        return ieds_peak, ieds_peak_win

    def cal_ieds_peak_in_ictal_windows(self, candidate_ictal_win=None, ieds_window=0.3,
                                       weighted_channel_number_peaks=None,
                                       data_gfp=None, gfp_peak_index=None,
                                       ied_channel_number=None, chan_threshold=(5, 200)):
        """
        Description:
            计算candidate_ictal_win中所有的ieds，并返回ieds的peak和windows
            计算流程:
                    (1). 获取满足chan_threshold阈值的peak time
                    (2). 寻找weighted_channel_number_peaks中最大值位置。
                    (3). 最大值位置附近的GFP peak为ied_peak。
                    (4). 根据GFP值选取ied时间窗。
                    (5). 将最大值位置从weighted_channel_number_peaks中删除。
                    (6). 重复(2)-(6)直至所有weighted_channel_number_peaks被删除。

        Input:
            :param candidate_ictal_win: np.array, long, shape(n_time)
                发作数据时间段。
            :param ieds_window: number, long
                输出ieds的时间窗长度，单位ms
            :param weighted_channel_number_peaks: torch.tensor, double, shape(n_sample)
                幅度加权的IED激活channel数目，仅在weighted_channel_number中的波峰位置有值
            :param data_gfp: torch.tensor, double, shape(n_sample)
                GFP
            :param gfp_peak_index: torch.tensor, long, shape(n_sample)
                GFP的波峰位置
            :param ied_channel_number: torch.tensor, double, shape(n_sample)
                对于每个时间点，被检测为IED的channel个数。
            :param chan_threshold: list, long, shape(2); or number, long
                需要满足的channel个数，每个时间内检测为IED的channel个数。
                为list的时候，channel的上限和下限；为number的时候，channel下限。

        Return:
            :return ieds_peak: number, long
                ieds的peak位置
            :return ieds_peak_win: torch.tensor, long, shape(1*candidate_ied_win.shape[0])
                ieds events的duration
        """
        # 复制输入变量，防止改变
        candidate_ictal_win, weighted_channel_number_peaks, data_gfp, gfp_peak_index, ied_channel_number =\
            candidate_ictal_win.copy(), weighted_channel_number_peaks.clone(), data_gfp.clone(), gfp_peak_index.clone(), \
            ied_channel_number.clone()

        # 判断输入是否存在
        if (candidate_ictal_win is None) or (weighted_channel_number_peaks is None) or \
                (data_gfp is None) or (gfp_peak_index is None) or (ied_channel_number is None):
            return None, None

        # 计算ieds持续时间的一半长度
        half_ieds_window = int(ieds_window * self.data_info['sfreq'] / 2)
        # 计算ied_peak与ied_windows边界的最小距离阈值
        peak_margin_samples = int(half_ieds_window / 2)

        # 获取candidate_ictal_win时间段内的weighted_channel_number, ied_channel_number, gfp_amp_slope, gfp_peak_index
        weighted_channel_number_peaks[:candidate_ictal_win[0]] = 0.
        weighted_channel_number_peaks[candidate_ictal_win[-1]:] = 0.
        ied_channel_number[:candidate_ictal_win[0]] = 0
        ied_channel_number[candidate_ictal_win[-1]:] = 0
        data_gfp[:candidate_ictal_win[0]] = 0
        data_gfp[candidate_ictal_win[-1]:] = 0
        gfp_peak_index = gfp_peak_index[torch.where((gfp_peak_index >= candidate_ictal_win[0]) &
                                                    (gfp_peak_index <= candidate_ictal_win[-1]))]

        # 保留ied_channel_number满足chan_threshold的采样点
        if len(chan_threshold) == 1:
            is_ied_idx = ied_channel_number >= chan_threshold
        elif len(chan_threshold) == 2:
            is_ied_idx = (ied_channel_number >= chan_threshold[0]) & (ied_channel_number <= chan_threshold[1])
        weighted_channel_number_peaks[~is_ied_idx] = 0

        # Algorithm!: knock down peaks that are too close to the largest peak
        all_ictal_peak_weighted_channel = weighted_channel_number_peaks.clone()
        all_ictal_peak = torch.where(weighted_channel_number_peaks > 0)[0]
        all_ictal_ieds_peak_times, all_ictal_ieds_wins = [], []
        remove_idx = torch.tensor([10000000]).long().to(self.device)
        while len(all_ictal_peak) > 0:
            # 选取在all_ictal_peak内的weighted_channel_number_peaks最大值点
            highest_peak_index = all_ictal_peak_weighted_channel[all_ictal_peak].argmax()
            highest_peak_weighted_channel_index = all_ictal_peak[highest_peak_index]
            # 获取最大值位置附近half_ieds_window*2时间窗内所有的gfp_peak位置
            for dist in range(0, half_ieds_window, 10):
                ieds_candidate_peak_idx = gfp_peak_index[
                    torch.where((gfp_peak_index - highest_peak_weighted_channel_index).abs() <= half_ieds_window)[0]]
                if len(ieds_candidate_peak_idx) > 0:
                    break
            # 如果时间窗内没有找到gfp_peak，删除all_ictal_peak中的highest_peak_index，并继续循环
            if len(ieds_candidate_peak_idx) == 0:
                all_ictal_peak = all_ictal_peak[torch.cat([torch.arange(0, highest_peak_index),
                                                           torch.arange(highest_peak_index + 1,
                                                                        all_ictal_peak.shape[0])])]
                continue
            # 获取ieds_candidate_peak_index中满足以下条件的peaks:
            # (1). ieds_candidate_peak_index与其左侧remove_idx之间最小距离大于peak_margin_samples
            # (2). ieds_candidate_peak_index与其右侧remove_idx之间最小距离大于peak_margin_samples
            # (3). ieds_candidate_peak_index与其右侧remove_idx之间最小距离，
            #      加上ieds_candidate_peak_index与其右侧remove_idx之间最小距离，大于ieds_window
            dist_remove_idx_left = [remove_idx[torch.where(remove_idx - x < 0)[0]] for x in ieds_candidate_peak_idx]
            min_dist_remove_idx_left = torch.tensor([x[(x - y).abs().argmin()]
                                                     if len(x) > 0 else torch.tensor(0).long().to(self.device)
                                                     for x, y in zip(dist_remove_idx_left, ieds_candidate_peak_idx)])
            dist_remove_idx_right = [remove_idx[torch.where(remove_idx - x >= 0)[0]] for x in ieds_candidate_peak_idx]
            min_dist_remove_idx_right = torch.tensor([x[(x - y).abs().argmin()]
                                                      if len(x) > 0 else torch.tensor(10000000).long().to(self.device)
                                                      for x, y in zip(dist_remove_idx_right, ieds_candidate_peak_idx)])
            temp = torch.tensor([(z - x) > peak_margin_samples and (y - z) > peak_margin_samples and
                                 (y - x) > half_ieds_window * 2 for x, y, z in zip(
                min_dist_remove_idx_left, min_dist_remove_idx_right, ieds_candidate_peak_idx)])
            ieds_candidate_peak_idx = ieds_candidate_peak_idx[temp]
            min_dist_remove_idx_left = min_dist_remove_idx_left[temp]
            min_dist_remove_idx_right = min_dist_remove_idx_right[temp]
            # ieds_candidate_peak_index内所有peak均不满足条件，删除all_ictal_peak中的highest_peak_index，并继续循环。
            if not len(ieds_candidate_peak_idx) > 0:
                all_ictal_peak = all_ictal_peak[torch.cat([torch.arange(0, highest_peak_index),
                                                           torch.arange(highest_peak_index + 1,
                                                                        all_ictal_peak.shape[0])])]
                continue
            # 使用ieds_candidate_peak_index中gfp_amp_slope最大的peak为ied_peak
            temp = data_gfp[ieds_candidate_peak_idx].argmax()
            ieds_peak = ieds_candidate_peak_idx[temp]
            min_dist_remove_idx_left = min_dist_remove_idx_left[temp]
            min_dist_remove_idx_right = min_dist_remove_idx_right[temp]

            # 选取ied peak周围的平均GFP幅度最大的时间窗为最终的ied时间窗
            # 计算search_win的起始点，要保证search_win内不包含remove_idx
            search_win_beg = torch.arange(max(ieds_peak.cpu() + peak_margin_samples - half_ieds_window * 2,
                                              min_dist_remove_idx_left),
                                          min(ieds_peak.cpu() - peak_margin_samples + half_ieds_window * 2,
                                              min_dist_remove_idx_right) - half_ieds_window * 2)
            search_win = search_win_beg.reshape(-1, 1) + torch.arange(0, half_ieds_window * 2).reshape(1, -1)
            ieds_peak_win = search_win[data_gfp[search_win].sum(dim=-1).argmax()].to(self.device)

            # 更新all_ictal_peak(删除all_ictal_peak中的highest_peak_index)，remove_idx(将ieds_peak_win加入remove_idx)
            all_ictal_peak = all_ictal_peak[torch.cat([torch.arange(0, highest_peak_index),
                                                       torch.arange(highest_peak_index + 1,
                                                                    all_ictal_peak.shape[0])])]
            remove_idx = torch.cat([remove_idx, ieds_peak_win])
            all_ictal_ieds_peak_times.append(ieds_peak)
            all_ictal_ieds_wins.append(ieds_peak_win - ieds_peak)

        # 输出
        if len(all_ictal_ieds_peak_times) > 0:
            all_ictal_ieds_peak_times = torch.tensor(all_ictal_ieds_peak_times).sort()
            all_ictal_ieds_wins = torch.stack(all_ictal_ieds_wins)[all_ictal_ieds_peak_times[1]]
            all_ictal_ieds_peak_times = all_ictal_ieds_peak_times[0]
        else:
            all_ictal_ieds_peak_times, all_ictal_ieds_wins = None, None

        self.del_var()
        return all_ictal_ieds_peak_times, all_ictal_ieds_wins

    def cal_ied_segments(self, ieds_peaks=None, ieds_win=None, samples_of_picked_peak=None,
                         data_segment_window=0.1):
        """
        Description:
            计算整段数据中中所有的ieds，并返回ieds的peak和windows。

        Input:
            :param ieds_peaks: list, [torch.tensor, long, shape()], shape(n_ieds)
                ieds的peak位置
            :param ieds_win: list, long, shape(n_ieds)
                ieds events的duration
            :param samples_of_picked_peak: torch.tensor, long, shape(n_channel*n_sample)
                peak点，在原始数据(n_channel*n_sample)中对应位置
            :param data_segment_window: number, double
                data segment的长度，单位s

        Return:
            :return ieds_peak_closest_idx: torch.tensor, double, shape(n*n_channel*shift个数)
                每个通道中，ied peak两侧最近的峰，在raw meg中的采样点位置
            :return data_segment: torch.tensor, double, shape(n*n_channel*shift个数*n_segment_sample)
                ieds events的duration

        """
        # 复制输入变量，防止改变
        samples_of_picked_peak = samples_of_picked_peak.clone()

        # 计算IED peak周围，每个通道的peak点位置
        ieds_peak_closest_idx = self.cal_peaks_closest_to_ieds_peaks(
            ieds_peak=ieds_peaks, ieds_win=ieds_win, samples_of_picked_peak=samples_of_picked_peak)
        # 获取数据片段
        data_segment = self.cal_data_segments_with_shift(data_segment_window=data_segment_window,
                                                         ieds_peak_closest_idx=ieds_peak_closest_idx)
        return ieds_peak_closest_idx, data_segment

    def cal_ieds_features(self, samples_of_picked_peak=None, ieds_peak_closest_idx=None,
                          peak_amplitude=None, half_peak_slope=None, peak_duration=None, peak_sharpness=None,
                          data_gfp=None, ieds_peaks=None,
                          channel_used_for_average=5):
        """
        Description:
            计算每个IED event的feature:
                (1). 首先计算每个IED event的主peak周围，每个通道最近的两个peaks点的feature
                (2). 计算每个通道中，最大的特征值。
                (3). 前channel_used_for_average个通道的均值，作为event的feature value

        Input:
            :param samples_of_picked_peak: torch.tensor, long, shape(n_channel*n_sample)
                peak点，在原始数据(n_channel*n_sample)中对应位置
            :param ieds_peak_closest_idx: torch.tensor, double, shape(n*n_channel*shift个数)
                每个通道中，ied peak两侧最近的峰，在raw meg中的采样点位置
            :param samples_of_picked_peak: torch.tensor, long, shape(n_channel*n_sample)
                peak点，在原始数据(n_channel*n_sample)中对应位置
            :param peak_amplitude: list, double, shape(n_channel*1)
                每个通道peak的峰峰值。
            :param half_peak_slope: list, double, shape(n_channel*1)
                每个通道peak的半高宽斜率。
            :param peak_duration: list, long, shape(n_channel*1)
                每个通道peak的持续时间。
            :param peak_sharpness: list, double, shape(n_channel*1)
                每个通道peak的锐度。
            :param channel_used_for_average: number, long
                用于平均的通道个数
            :param data_gfp: torch.tensor, double, shape(n_sample)
                GFP
            :param ieds_peaks: list, long, shape(n_ieds)
                ieds的peak位置, 用于计算每个event的gfp值

        Return:
            :return ied_amplitude_peaks: torch.tensor, double, shape(n*n_channel*2)
                每个通道，ieds主峰值两侧peaks的幅度
            :return ied_half_slope_peaks: torch.tensor, double, shape(n*n_channel*2)
                每个通道，ieds主峰值两侧peaks的斜率
            :return ied_sharpness_peaks: torch.tensor, double, shape(n*n_channel*2)
                每个通道，ieds主峰值两侧peaks的锐度
            :return ied_duration_peaks: torch.tensor, double, shape(n*n_channel*2)
                每个通道，ieds主峰值两侧peaks的峰持续时间
            :return ied_amplitude: torch.tensor, double, shape(n)
                每个ieds的幅度
            :return ied_half_slope: torch.tensor, double, shape(n)
                每个ieds的斜率
            :return ied_sharpness: torch.tensor, double, shape(n)
                每个ieds的锐度
            :return ied_duration: torch.tensor, double, shape(n)
                每个ieds的峰持续时间
            :return ied_gfp_amp: torch.tensor, double, shape(n)
                每个ieds的gfp值
        """

        # 计算IED的feature
        # 计算ied event的主peak周围，每个通道最近的两个peaks点的feature
        ied_amplitude_peaks, ied_half_slope_peaks, ied_sharpness_peaks, ied_duration_peaks = \
            self.cal_ieds_peaks_features(samples_of_picked_peak=samples_of_picked_peak,
                                         ieds_peak_closest_idx=ieds_peak_closest_idx,
                                         peak_amplitude=peak_amplitude, half_peak_slope=half_peak_slope,
                                         peak_duration=peak_duration, peak_sharpness=peak_sharpness)
        # 计算IED的feature
        ied_amplitude, ied_half_slope, ied_sharpness, ied_duration, ied_gfp_amp = \
            self._cal_ieds_features(ieds_peak=ieds_peaks, averaged_chan_num=channel_used_for_average, data_gfp=data_gfp,
                                    ied_amplitude_peaks=ied_amplitude_peaks, ied_half_slope_peaks=ied_half_slope_peaks,
                                    ied_sharpness_peaks=ied_sharpness_peaks, ied_duration_peaks=ied_duration_peaks)

        return ied_amplitude_peaks, ied_half_slope_peaks, ied_sharpness_peaks, ied_duration_peaks, \
               ied_amplitude, ied_half_slope, ied_sharpness, ied_duration, ied_gfp_amp

    def get_ieds_in_candidate_ieds_windows(self, mag_grad_ratio=0.056, chan_threshold=(5, 200), candidate_ieds_win=None,
                                           peak_amplitude=None, half_peak_slope=None,
                                           win_samples_of_picked_peak=None, all_samples_of_picked_peak=None):
        """
        Description:
            计算ied candidate是否为ied，以及ieds的peak和windows

        Input:
            :param mag_grad_ratio: number, double
                mag和grad的比值，用于rescale mag和grad到同一scale
            :param chan_threshold: list, long, shape(2); or number, long
                需要满足的channel个数，每个时间内检测为IED的channel个数。
                为list的时候，channel的上限和下限；为number的时候，channel下限。
            :param candidate_ieds_win: np.array, long, shape(n*n_time)
                需要判断的ied时间段，在原始MEG数据中的采样点位置。
            :param peak_amplitude: list, double, shape(n_channel*1)
                每个通道peak的峰峰值。
            :param half_peak_slope: list, double, shape(n_channel*1)
                每个通道peak的半高宽斜率。
            :param win_samples_of_picked_peak: np.array, bool, shape(n_channel*n_samples)
                peak点周围peak_win内的采样点，在原始数据(n_channel*n_sample)中对应位置
            :param all_samples_of_picked_peak: np.array, bool, shape(n_channel*n_samples)
                peak点内所有点(包括上身边和下降边)，在原始数据(n_channel*n_sample)中对应位置

        Return:
            :return is_ieds: torch.tensor, bool, shape(candidate_ieds_win.shape[0])
                是否是ied
            :return ieds_peak: torch.tensor, long, shape(n)
                ieds的peak位置
            :return ieds_win: torch.tensor, long, shape(n*ied_win_length)
                ieds events的duration
        """
        # 复制输入变量，防止改变
        candidate_ieds_win, peak_amplitude, half_peak_slope, win_samples_of_picked_peak, all_samples_of_picked_peak = \
            candidate_ieds_win.copy(), peak_amplitude.copy(), half_peak_slope.copy(), \
            win_samples_of_picked_peak.copy(), all_samples_of_picked_peak.copy()

        # 计算GFP
        data_gfp, gfp_amp_slope, gfp_peak_amp_slope, gfp_peak_index = self.cal_dat_gfp(mag_grad_ratio=mag_grad_ratio)
        self.set_data_gfp(data_gfp=data_gfp, gfp_amp_slope=gfp_amp_slope,
                          gfp_peak_amp_slope=gfp_peak_amp_slope, gfp_peak_index=gfp_peak_index)

        # 计算weighted_channel_number
        weighted_channel_number, weighted_channel_number_peaks = \
            self.cal_weighted_channel_number(peak_amplitude=peak_amplitude, half_peak_slope=half_peak_slope,
                                             win_samples_of_picked_peak=win_samples_of_picked_peak)
        self.set_weighted_channel_number(weighted_channel_number=weighted_channel_number,
                                         weighted_channel_number_peaks=weighted_channel_number_peaks)

        # 计算ied_channel_number
        ied_channel_number = self.cal_channel_number(all_samples_of_picked_peak=all_samples_of_picked_peak)
        self.set_channel_number(ied_channel_number)

        # 计算is_ieds, ieds_peaks和ieds_win
        is_ieds, ieds_peaks, ieds_win = [], [], []
        for candidate_ied_win in candidate_ieds_win:
            is_ied = self.ied_validate(candidate_ied_win=candidate_ied_win,
                                       ied_channel_number=ied_channel_number, chan_threshold=chan_threshold)
            if is_ied is True:
                ied_peak, ied_win = \
                    self.cal_ieds_peak_in_ieds_windows(candidate_ied_win=candidate_ied_win,
                                                       weighted_channel_number=weighted_channel_number.clone(),
                                                       data_gfp=data_gfp.clone(),
                                                       gfp_peak_index=gfp_peak_index.clone())
                if (ied_peak is not None) and (ied_win is not None):
                    is_ieds.append(True)
                    ieds_peaks.append(ied_peak)
                    ieds_win.append(ied_win)
                else:
                    is_ieds.append(False)
            else:
                is_ieds.append(False)
        is_ieds = torch.tensor(is_ieds)
        if len(ieds_peaks) > 0:
            ieds_peaks, ieds_win = torch.tensor(ieds_peaks), torch.stack(ieds_win)
        else:
            ieds_peaks, ieds_win = None, None

        return is_ieds, ieds_peaks, ieds_win

    def get_ieds_in_ictal_windows(self, mag_grad_ratio=0.056, chan_threshold=(5, 200),
                                  candidate_ictal_wins=None, ieds_window=0.1,
                                  peak_amplitude=None, half_peak_slope=None,
                                  win_samples_of_picked_peak=None, all_samples_of_picked_peak=None):
        """
        Description:
            计算candidate_ictal_win中所有的ieds，并返回ieds的peak和windows

        Input:
            :param mag_grad_ratio: number, double
                mag和grad的比值，用于rescale mag和grad到同一scale
            :param chan_threshold: list, long, shape(2); or number, long
                需要满足的channel个数，每个时间内检测为IED的channel个数。
                为list的时候，channel的上限和下限；为number的时候，channel下限。
            :param candidate_ictal_wins: np.array, long, shape(n_time)
                需要判断的ied时间段，在原始MEG数据中的采样点位置。
            :param ieds_window: number, long
                输出ieds的时间窗长度，单位ms
            :param peak_amplitude: list, double, shape(n_channel*1)
                每个通道peak的峰峰值。
            :param half_peak_slope: list, double, shape(n_channel*1)
                每个通道peak的半高宽斜率。
            :param win_samples_of_picked_peak: np.array, bool, shape(n_channel*n_samples)
                peak点周围peak_win内的采样点，在原始数据(n_channel*n_sample)中对应位置
            :param all_samples_of_picked_peak: np.array, bool, shape(n_channel*n_samples)
                peak点内所有点(包括上身边和下降边)，在原始数据(n_channel*n_sample)中对应位置

        Return:
            :return ieds_peaks: torch.tensor, long, shape(n)
                ieds的peak位置
            :return ieds_win: torch.tensor, long, shape(n*n_ieds_window)
                ieds events的duration
        """
        # 复制输入变量，防止改变
        candidate_ictal_wins, peak_amplitude, half_peak_slope, win_samples_of_picked_peak, all_samples_of_picked_peak = \
            candidate_ictal_wins.copy(), peak_amplitude.copy(), half_peak_slope.copy(), \
            win_samples_of_picked_peak.copy(), all_samples_of_picked_peak.copy()

        # 计算gfp
        data_gfp, gfp_amp_slope, gfp_peak_amp_slope, gfp_peak_index = self.cal_dat_gfp(mag_grad_ratio=mag_grad_ratio)
        self.set_data_gfp(data_gfp=data_gfp, gfp_amp_slope=gfp_amp_slope,
                          gfp_peak_amp_slope=gfp_peak_amp_slope, gfp_peak_index=gfp_peak_index)

        # 计算weighted_channel_number
        weighted_channel_number, weighted_channel_number_peaks = \
            self.cal_weighted_channel_number(peak_amplitude=peak_amplitude, half_peak_slope=half_peak_slope,
                                             win_samples_of_picked_peak=win_samples_of_picked_peak)
        self.set_weighted_channel_number(weighted_channel_number=weighted_channel_number,
                                         weighted_channel_number_peaks=weighted_channel_number_peaks)

        # 计算channel_number
        ied_channel_number = self.cal_channel_number(all_samples_of_picked_peak=all_samples_of_picked_peak)
        self.set_channel_number(ied_channel_number)

        # 计算ieds_peaks和ieds_win
        ieds_peaks, ieds_win = [], []
        for candidate_ictal_win in candidate_ictal_wins:
            ied_peak, ied_win = \
                self.cal_ieds_peak_in_ictal_windows(candidate_ictal_win=candidate_ictal_win,
                                                    ieds_window=ieds_window,
                                                    weighted_channel_number_peaks=weighted_channel_number_peaks.clone(),
                                                    data_gfp=data_gfp.clone(),
                                                    gfp_peak_index=gfp_peak_index.clone(),
                                                    ied_channel_number=ied_channel_number.clone(),
                                                    chan_threshold=chan_threshold)
            if (ied_peak is not None) and (ied_win is not None):
                ieds_peaks.append(ied_peak)
                ieds_win.append(ied_win)
        if len(ieds_peaks) > 0:
            ieds_peaks, ieds_win = torch.cat(ieds_peaks), torch.cat(ieds_win)
        else:
            ieds_peaks, ieds_win = None, None

        return ieds_peaks, ieds_win

    def get_ieds_in_all_meg(self, mag_grad_ratio=0.056, chan_threshold=(5, 200), ieds_window=0.3,
                            peak_amplitude=None, half_peak_slope=None,
                            win_samples_of_picked_peak=None, all_samples_of_picked_peak=None):
        """
        Description:
            计算整段数据中中所有的ieds，并返回ieds的peak和windows。

        Input:
            :param mag_grad_ratio: number, double
                mag和grad的比值，用于rescale mag和grad到同一scale
            :param chan_threshold: list, long, shape(2); or number, long
                需要满足的channel个数，每个时间内检测为IED的channel个数。
                为list的时候，channel的上限和下限；为number的时候，channel下限。
            :param ieds_window: number, double
                输出ieds的时间窗长度，单位s
            :param peak_amplitude: list, double, shape(n_channel*1)
                每个通道peak的峰峰值。
            :param half_peak_slope: list, double, shape(n_channel*1)
                每个通道peak的半高宽斜率。
            :param win_samples_of_picked_peak: np.array, bool, shape(n_channel*n_samples)
                peak点周围peak_win内的采样点，在原始数据(n_channel*n_sample)中对应位置
            :param all_samples_of_picked_peak: np.array, bool, shape(n_channel*n_samples)
                peak点内所有点(包括上身边和下降边)，在原始数据(n_channel*n_sample)中对应位置

        Return:
            :return ieds_peak: torch.tensor, long, shape(n)
                ieds的peak位置
            :return ieds_win: torch.tensor, long, shape(n*ied_win_length)
                ieds events的duration
        """
        # 复制输入变量，防止改变
        peak_amplitude, half_peak_slope, win_samples_of_picked_peak, all_samples_of_picked_peak = \
            peak_amplitude.copy(), half_peak_slope.copy(), \
            win_samples_of_picked_peak.copy(), all_samples_of_picked_peak.copy()


        # 计算gfp
        data_gfp, gfp_amp_slope, gfp_peak_amp_slope, gfp_peak_index = self.cal_dat_gfp(mag_grad_ratio=mag_grad_ratio)
        self.set_data_gfp(data_gfp=data_gfp, gfp_amp_slope=gfp_amp_slope,
                          gfp_peak_amp_slope=gfp_peak_amp_slope, gfp_peak_index=gfp_peak_index)

        # 计算weighted_channel_number
        weighted_channel_number, weighted_channel_number_peaks = \
            self.cal_weighted_channel_number(peak_amplitude=peak_amplitude, half_peak_slope=half_peak_slope,
                                             win_samples_of_picked_peak=win_samples_of_picked_peak)
        self.set_weighted_channel_number(weighted_channel_number=weighted_channel_number,
                                         weighted_channel_number_peaks=weighted_channel_number_peaks)

        # 计算channel_number
        ied_channel_number = self.cal_channel_number(all_samples_of_picked_peak=all_samples_of_picked_peak)
        self.set_channel_number(ied_channel_number)

        # 计算IED，以及对应的peak和windows
        ieds_peaks, ieds_win = \
            self.cal_ieds_peak_in_all_data(ied_win_length=ieds_window, chan_threshold=chan_threshold,
                                           ied_channel_number=ied_channel_number,
                                           weighted_channel_number_peaks=weighted_channel_number_peaks,
                                           data_gfp=data_gfp, gfp_peak_index=gfp_peak_index)

        return ieds_peaks, ieds_win

    def plot_meg_data_in_ied_event(self, ied_peak=None, half_win=0.5, samples_of_picked_peak=None,
                                   figsize=(10, 25.6), dpi=75):
        """
        Description:
            plot event周围MEG数据

        Input:
            :param ied_peak: number, double
                ied peak 位置
            :param half_win: number, double
            :param samples_of_picked_peak: number, double
                peak点，在原始数据(n_channel*n_sample)中对应位置
            :param figsize: tuple, double
                figure的大小
            :param dpi: number, long
                figure的dpi
        """
        if ied_peak is not None:
            half_win = int(half_win * self.data_info['sfreq'])
            meg_data = self.raw_data[:,
                       max(0, ied_peak - half_win):min(ied_peak + half_win, self.raw_data.shape[1])].cpu()
            if samples_of_picked_peak is not None:
                channel_ieds = np.where(samples_of_picked_peak[:,
                                        max(0, ied_peak - half_win):min(ied_peak + half_win,
                                                                        self.raw_data.shape[1])] != -1)[0]
            fig = plt.figure(figsize=figsize, dpi=dpi)
            # plot raw
            for i in range(3):
                ax = plt.Axes(fig, [1 / 3 * i, 0.0, 0.95 / 3, 0.96])
                ax.set_axis_off()
                fig.add_axes(ax)
                X = meg_data[torch.arange(i, 306, 3)]
                X = (X - X.amin(dim=-1, keepdim=True)) / (X.amax(dim=-1, keepdim=True) - X.amin(dim=-1, keepdim=True)) \
                    + torch.arange(X.shape[0]).reshape(-1, 1)
                ax.plot(X.t(), 'b')
                if samples_of_picked_peak is not None and \
                        len([j for j, x in enumerate(np.arange(i, 306, 3)) if x in channel_ieds]) > 0:
                    ax.plot(X[[j for j, x in enumerate(np.arange(i, 306, 3)) if x in channel_ieds]].t(), 'r')
                ax.plot([ied_peak - max(0, ied_peak - half_win), ied_peak - max(0, ied_peak - half_win)],
                        [0, X.shape[0]], 'r', linewidth=0.5)
            plt.show()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(2)

    raw = mne.io.read_raw_fif('/sanbo_dataset/4k_Project/Data_Full/MEG_IED/MEG1781/MEG1781_EP_1_tsss.fif',
                              verbose='error', preload=True)
    raw.pick(mne.pick_types(raw.info, meg=True, ref_meg=False))
    Info = raw.info
    Raw_data = raw.get_data()

    # 计算发作间期IEDs
    FindPeaks = ied_peak_feature.cal_peak_feature(raw_data=Raw_data, data_info=Info, device=2)
    Peaks = FindPeaks.get_peaks(smooth_windows=0.02, smooth_iterations=2, z_win=100, z_region=True, z_mag_grad=False,
                                exclude_duration=0.02,
                                peak_amplitude_threshold=[2, None], peak_slope_threshold=[None, None],
                                half_peak_slope_threshold=[2, None], peak_sharpness_threshold=[None, None])
    IEDs = ied_detection_threshold(raw_data=Raw_data, data_info=Info, device=0)
    IEDs_peaks, IEDs_win = \
            IEDs.get_ieds_in_all_meg(mag_grad_ratio=0.56, chan_threshold=(5, 100),
                                     ieds_window=0.3,
                                     peak_amplitude=Peaks['peak_amplitude'], half_peak_slope=Peaks['half_peak_slope'],
                                     win_samples_of_picked_peak=Peaks['win_samples_of_picked_peak'],
                                     all_samples_of_picked_peak=Peaks['all_samples_of_picked_peak'])

    # 计算发作ieds
    candidate_Ictal_wins = np.concatenate((np.arange(0, 200000).reshape(1, -1),
                                           np.arange(200000, 400000).reshape(1, -1)))
    FindPeaks = ied_peak_feature.cal_peak_feature(raw_data=Raw_data, data_info=Info, device=2)
    Peaks = FindPeaks.get_peaks(smooth_windows=0.02, smooth_iterations=2, z_win=100, z_region=True, z_mag_grad=False,
                                exclude_duration=0.02,
                                peak_amplitude_threshold=[2, None], peak_slope_threshold=[None, None],
                                half_peak_slope_threshold=[2, None], peak_sharpness_threshold=[None, None])
    IEDs = ied_detection_threshold(raw_data=Raw_data, data_info=Info, device=0)
    IEDs_peaks, IEDs_win = \
        IEDs.get_ieds_in_ictal_windows(mag_grad_ratio=0.056, chan_threshold=(5, 200),
                                       candidate_ictal_wins=candidate_Ictal_wins, ieds_window=100,
                                       peak_amplitude=Peaks['peak_amplitude'],
                                       half_peak_slope=Peaks['half_peak_slope'],
                                       win_samples_of_picked_peak=Peaks['win_samples_of_picked_peak'],
                                       all_samples_of_picked_peak=Peaks['all_samples_of_picked_peak'])

    # 根据ied candidate计算发作间期IEDs
    candidate_IEDs_win = (np.random.randint(200, 10000, 100) + np.arange(-150, 150).reshape(300, -1)).T
    FindPeaks = ied_peak_feature.cal_peak_feature(raw_data=Raw_data, data_info=Info, device=2)
    Peaks = FindPeaks.get_peaks(smooth_windows=0.02, smooth_iterations=2, z_win=100, z_region=True, z_mag_grad=False,
                                exclude_duration=0.02,
                                peak_amplitude_threshold=[2, None], peak_slope_threshold=[None, None],
                                half_peak_slope_threshold=[2, None], peak_sharpness_threshold=[None, None])
    IEDs = ied_detection_threshold(raw_data=Raw_data, data_info=Info, device=0)
    is_IEDs, IEDs_peaks, IEDs_win = \
        IEDs.get_ieds_in_candidate_ieds_windows(mag_grad_ratio=0.056, chan_threshold=(5, 200),
                                                candidate_ieds_win=candidate_IEDs_win,
                                                peak_amplitude=Peaks['peak_amplitude'],
                                                half_peak_slope=Peaks['half_peak_slope'],
                                                win_samples_of_picked_peak=Peaks['win_samples_of_picked_peak'],
                                                all_samples_of_picked_peak=Peaks['all_samples_of_picked_peak'])
