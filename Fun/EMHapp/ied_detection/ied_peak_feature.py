# -*- coding:utf-8 -*-
# @Time    : 2022/2/4
# @Author  : cuiwei
# @File    : ied_peak_feature.py
# @Software: PyCharm
# @Script to:
#   - 对MEG信号重建出所有的峰值位置，并计算峰值特征。

import os
import mne
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


class cal_peak_feature:
    """
    Description:
        1. 对于输入MEG的每一个通道信号，重建出波峰波谷的采样点位置以及对应信号强度。
        2. 对重建出的每一个波峰或者波谷，计算特征：
            1). 峰值持续时间
            2). 峰值半高宽持续时间
            3). 平均峰峰值
            4). 平均斜率
            5). 平均半峰值斜率
            6). 锐度
        3. (optional) 计算每个波峰或者波谷，相对于周围基线的相对特征值。
        4. 删除特征值小于设定阈值的波峰或者波谷。
    """

    def __init__(self, raw_data=None, data_info=None, bad_segment=None, device=-1, n_jobs=10):
        """
        Description:
            初始化变量，使用GPU或者CPU。

        Input:
            :param raw_data: ndarray, double, shape(channel, samples)
                MEG滤波后数据
            :param data_info: dict
                MEG数据的信息, MNE读取的raw.info
            :param bad_segment: ndarray, bool, shape(samples)
                MEG数据中的坏片段，True代表对应时间点属于坏片段
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
        self.raw_data, self.data_info, self.bad_segment = raw_data, data_info, bad_segment
        self.signal_peaks = None
        self.peak_amplitude, self.peak_slope, self.half_peak_slope, self.peak_sharpness, self.peak_duration, \
        self.peak_half_duration, self.peak_idx, self.peak_index = None, None, None, None, None, None, None, None
        self.peak_amplitude_zscore, self.peak_slope_zscore, self.half_peak_slope_zscore, self.peak_sharpness_zscore = \
            None, None, None, None
        self.picked_peak = None
        self.set_raw_data(raw_data)
        self.set_data_info(data_info)
        self.set_bad_segment(bad_segment)
        self.set_signal_peaks()
        # 获取MAG和GRAD通道的index
        self.chan_type = {'mag': torch.tensor(mne.pick_types(self.data_info, meg='mag')).to(self.device),
                          'grad': torch.tensor(mne.pick_types(self.data_info, meg='grad')).to(self.device)}
        # 获取8脑区通道的index
        self.chan_region = [torch.tensor([self.data_info.ch_names.index(y)
                                          for y in mne.read_vectorview_selection(x, info=self.data_info)])
                            for x in ['Left-temporal', 'Right-temporal', 'Left-parietal', 'Right-parietal',
                                      'Left-occipital', 'Right-occipital', 'Left-frontal', 'Right-frontal']]

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

    def set_bad_segment(self, bad_segment=None):
        if bad_segment is None:
            bad_segment = np.ones(self.get_raw_data().shape[1]) < 0
        self.bad_segment = torch.tensor(bad_segment)

    def get_bad_segment(self):
        return self.bad_segment

    def set_signal_peaks(self, signal_peaks=None):
        if signal_peaks is None:
            signal_peaks = {'pos_peak_idx': [], 'pos_peak_data': [],
                            'neg_peak_idx': [], 'neg_peak_data': [],
                            'pos_half_peak_idx_num': [], 'neg_half_peak_idx_num': []}
        self.signal_peaks = signal_peaks

    def get_signal_peaks(self):
        return self.signal_peaks

    def cal_signal_peaks(self, smooth_windows=0.02, smooth_iterations=2, segments_gpu=(0, 0.25, 0.5, 1.1)):
        """
        Description:
            对于输入MEG的每一个通道信号，重建出波峰波谷的采样点位置以及对应信号强度。
            并对每个波峰波谷，重建出左右两边的半高宽采样点个数。
            输出数据的波峰波谷排列形式：
               .   .   .   .   .   .     波峰
              /\  /\  /\  /\  /\  /\
             /  \/  \/  \/  \/  \/  \
            0    .   .   .   .   .   n   波谷
            波峰检测：
            MEG data in IED frequency band was first smoothed to eliminate influence of noise signals in peak detection.

            Then, sample with amplitude of smoothed MEG higher/lower than those of adjacent samples (one sample before
            and one sample after selected sample) was detected as positive/negative candidate peak.

            Considering that smoothing operation may lead a time shift in peak positions, we defined samples with
            minimum value of filtered MEG data between adjacent positive candidate peaks and samples with maximum value
            of filtered MEG data between adjacent negative as final detected peaks.

        Input:
            :param smooth_windows: number, float
                平滑窗长度(毫秒)。略高于需要平滑掉的噪声峰持续时间。
            :param smooth_iterations: number, int
                平滑次数。
            :param segments_gpu: list, float
                用于矩阵计算时，将矩阵进行切片处理，从而减少显存的使用。以0开始，以大于1数字结尾。至少是（0， 1.01）
                segments_gpu = (0, 1.01)不进行切片处理

        Return
            :return signal_peaks: dict:
                pos_peak_idx_raw_2d: list, long, shape(n_channel*1)
                    每个通道的波峰所对应的采样点位置。
                pos_peak_raw_2d: list, float, shape(n_channel*1)
                    每个通道的波峰所对应的采样点幅度。
                neg_peak_idx_raw_2d: list, long, shape(n_channel*1)
                    每个通道的波谷所对应的采样点位置。
                neg_peak_raw_2d: list, float, shape(n_channel*1)
                    每个通道的波谷所对应的采样点幅度。
                pos_peak_half_idx_num_2d: list, long, shape(n_channel*1)
                    每个通道的波峰，左右两边的半高宽到该波峰位置的采样点个数。
                neg_peak_half_idx_num_2d: list, long, shape(n_channel*1)
                    每个通道的波谷，左右两边的半高宽到该波谷位置的采样点个数。
        """

        # -------------------------------- Step 0 --------------------------------
        #                            初始化，获得需要的变量
        # ------------------------------------------------------------------------
        raw_data = self.get_raw_data().float().to(self.device)
        n_channel, n_sample = raw_data.shape[0], raw_data.shape[1]
        info = self.get_data_info()
        bad_segment = self.get_bad_segment().to(self.device)
        sfreq = info['sfreq']

        # -------------------------------- Step 1 --------------------------------
        #                          平滑原始数据，去除噪声干扰
        #               将2-d的raw_data和smooth_data，转换成1-d，实现矩阵计算
        # ------------------------------------------------------------------------
        smooth_conv = torch.nn.Conv2d(1, 1, [1, np.floor(sfreq * smooth_windows).astype('int16') + 1],
                                      padding=(0, np.floor(np.floor(sfreq * smooth_windows) / 2).astype('int16')),
                                      padding_mode='replicate', bias=False).to(self.device)
        torch.nn.init.constant_(smooth_conv.weight, 1 / (np.floor(sfreq * smooth_windows).astype('int16') + 1))
        smooth_conv.requires_grad_(False)
        smooth_data = raw_data.clone().unsqueeze(0).unsqueeze(0).to(self.device)
        for i in range(smooth_iterations):
            smooth_data = smooth_conv(smooth_data)
        smooth_data = smooth_data.squeeze()
        self.del_var()
        # 将2-d的raw_data和smooth_data，转换成1-d
        raw_data, smooth_data = raw_data.reshape(-1), smooth_data.reshape(-1)

        # -------------------------------- Step 2 --------------------------------
        #                         在平滑数据中，找到波峰波谷位置
        #             一阶导数二值化后（大于0），再次求导。-1为波峰位置，1为波谷位置
        #                           输出数据波峰波谷的排列形式：
        #                          .   .   .   .   .   .     波峰
        #                         /\  /\  /\  /\  /\  /\
        #                        /  \/  \/  \/  \/  \/  \
        #                       0    .   .   .   .   .   n   波谷
        # ------------------------------------------------------------------------
        data_diff = (smooth_data.diff() > 0).float().diff()
        pos_peak_idx_smooth = torch.where(data_diff == -1)[0] + 1
        neg_peak_idx_smooth = torch.where(data_diff == 1)[0] + 1
        # 强制信号的第一个和最后一个样本点为波谷点
        if neg_peak_idx_smooth[-1] < pos_peak_idx_smooth[-1]:
            neg_peak_idx_smooth = torch.cat((neg_peak_idx_smooth,
                                             torch.tensor([smooth_data.shape[0] - 1], device=self.device)))
        else:
            neg_peak_idx_smooth[-1] = smooth_data.shape[0] - 1
        if neg_peak_idx_smooth[0] > pos_peak_idx_smooth[0]:
            neg_peak_idx_smooth = torch.cat((torch.tensor([0], device=self.device), neg_peak_idx_smooth))
        else:
            neg_peak_idx_smooth[0] = 0
        self.del_var('smooth_data')

        # -------------------------------- Step 3 --------------------------------
        #                        在原始数据中，找到波峰波谷点位置
        #                相邻波峰之间(一个完整波谷)，原始数据最小值位置为波谷
        #                相邻波谷之间(一个完整波峰)，原始数据最小值位置为波峰
        #                           输入数据波峰波谷的排列形式：
        #                          .   .   .   .   .   .     波峰
        #                         /\  /\  /\  /\  /\  /\
        #                        /  \/  \/  \/  \/  \/  \
        #                       0    .   .   .   .   .   n   波谷
        #                          输出数据波峰波谷的排列形式：
        #                         .    .   .   .   .   .     波峰
        #                          \  /\  /\  /\  /\  /
        #                           \/  \/  \/  \/  \/
        #                            .   .   .   .   .       波谷
        # ------------------------------------------------------------------------
        # Step3.1 计算波谷点位置
        # 使用数据索引的方式，将原始数据转换成矩阵：[波谷个数, 时间最长波谷的样本数]（没有原始数据的位置，赋值为100，避免对min操作的影响）。
        # 对该矩阵使用argmin(axis=-1)操作，得到最小值在每个波谷中的位置，作为波谷点位置。
        # 将波谷点加上对应波谷起始点在整个数据中的位置，得到该波谷点在整个数据中位置。
        # 生成维度为[1，波谷个数*时间最长波谷的样本数]的矩阵，全部赋值为100（100远大于MEG原始数据最大值，避免对min操作的影响）
        pos_idx_diff = pos_peak_idx_smooth.diff().to(self.device)
        if pos_idx_diff.shape[0] * pos_idx_diff.max() * 4 / 1024 / 1024 / 1024 < 12:
            neg_peak_idx_raw = torch.full([pos_idx_diff.shape[0], pos_idx_diff.max()], 100,
                                          device=self.device, dtype=torch.float)
            # 将原始数据放入生成矩阵对应位置
            neg_peak_idx_raw[
                (  # 波谷起始点在原始数据中的位置
                    torch.arange(pos_idx_diff.shape[0]).to(self.device).repeat_interleave(pos_idx_diff),
                    # 每个采样点在原始数据中的位置 -
                    torch.arange(pos_peak_idx_smooth[-1] - pos_peak_idx_smooth[0], device=self.device) -
                    # 波谷起始点在原始数据中的位置,
                    torch.cat((torch.tensor([0]).to(self.device),
                               pos_idx_diff[:-1])).cumsum(dim=0).repeat_interleave(pos_idx_diff)
                )] = raw_data[pos_peak_idx_smooth[0]:pos_peak_idx_smooth[-1]]
            # 对于每个波谷，获得最小值（波谷点）位置。将该位置加上波谷起始点在原始数据中位置，获得波谷点在原始数据中位置
            neg_peak_idx_raw = neg_peak_idx_raw[:, 1:].argmin(dim=-1) + pos_peak_idx_smooth[:-1] + 1
            self.del_var()
        else:
            # 当显存溢出, 使用chunk的方法，进行减少GPU显存使用
            pos_idx_diff_sample = pos_idx_diff.repeat_interleave(pos_idx_diff)
            neg_peak_idx_raw = torch.zeros(pos_idx_diff.shape[0], device=self.device).long()
            segments_gpu_index = (torch.tensor(segments_gpu).to(self.device) * pos_idx_diff.max()).long()
            for i in range(len(segments_gpu) - 1):
                # 计算segment内的边索引
                segments_peak_idx = torch.where((pos_idx_diff >= segments_gpu_index[i]) &
                                                (pos_idx_diff < segments_gpu_index[i + 1]))[0]
                segments_sample_idx = torch.where((pos_idx_diff_sample >= segments_gpu_index[i]) &
                                                  (pos_idx_diff_sample < segments_gpu_index[i + 1]))[0]
                if any(segments_peak_idx):
                    pos_idx_diff_temp = pos_idx_diff[segments_peak_idx]
                    # 计算当前segment对应的矩阵
                    matrix_segment = torch.full([pos_idx_diff_temp.shape[0], pos_idx_diff_temp.max()],
                                                100, device=self.device, dtype=torch.float)
                    matrix_segment[
                        (  # 波谷起始点在原始数据中的位置,
                            torch.arange(pos_idx_diff_temp.shape[0],
                                         device=self.device).repeat_interleave(pos_idx_diff_temp),
                            # 每个采样点在原始数据中的位置 -
                            torch.arange(segments_sample_idx.shape[0], device=self.device) -
                            # 波谷起始点在原始数据中的位置
                            torch.cat((torch.tensor([0], device=self.device),
                                       pos_idx_diff_temp[:-1])).cumsum(dim=0).repeat_interleave(pos_idx_diff_temp)
                        )] = raw_data[pos_peak_idx_smooth[0]:pos_peak_idx_smooth[-1]][segments_sample_idx]
                    # 对于每个波谷，获得最小值（波谷点）位置。将该位置加上波谷起始点在原始数据中位置，获得波谷点在原始数据中位置
                    neg_peak_idx_raw[segments_peak_idx] = matrix_segment[:, 1:].argmin(dim=-1) + \
                                                          pos_peak_idx_smooth[:-1][segments_peak_idx] + 1
                self.del_var()
            del matrix_segment, segments_peak_idx, segments_sample_idx, pos_idx_diff_sample
            self.del_var()

        # Step3.2 计算波峰点位置
        # 使用数据索引的方式，将原始数据转换成矩阵：[波峰个数, 时间最长波峰的样本数]（没有原始数据的位置，赋值为-100，避免对max操作的影响）。
        # 对该矩阵使用argmax(axis=-1)操作，得到最大值在每个波谷中的位置，作为波峰点位置。
        # 将波峰点加上对应波峰起始点在整个数据中的位置，得到该波峰点在整个数据中位置。
        # 生成维度为[1，波峰个数*时间最长波峰的样本数]的矩阵，全部赋值为-100（-100远小于MEG原始数据最大值，避免对max操作的影响）
        neg_idx_diff = neg_peak_idx_smooth.diff().to(self.device)
        if neg_idx_diff.shape[0] * neg_idx_diff.max() * 4 / 1024 / 1024 / 1024 < 12:
            pos_peak_idx_raw = torch.full([neg_idx_diff.shape[0], neg_idx_diff.max()], -100,
                                          device=self.device, dtype=torch.float)
            # 将原始数据放入生成矩阵对应位置
            pos_peak_idx_raw[
                (  # 波谷起始点在原始数据中的位置,
                    torch.arange(neg_idx_diff.shape[0]).to(self.device).repeat_interleave(neg_idx_diff),
                    # 每个采样点在原始数据中的位置 -
                    torch.arange(neg_peak_idx_smooth[-1] - neg_peak_idx_smooth[0], device=self.device) -
                    # 波峰起始点在原始数据中的位置 +
                    torch.cat((torch.tensor([0]).to(self.device),
                               neg_idx_diff[:-1])).cumsum(dim=0).repeat_interleave(neg_idx_diff)
                )] = raw_data[neg_peak_idx_smooth[0]:neg_peak_idx_smooth[-1]]
            # 将neg_peak_idx_raw，reshape成维度为[波峰个数，时间最长波峰的样本数]的矩阵
            pos_peak_idx_raw = pos_peak_idx_raw.reshape(neg_idx_diff.shape[0], neg_idx_diff.max())
            # 对于每个波峰，获得最大值（波峰点）位置。将该位置加上波峰起始点在原始数据中位置，获得波峰点在原始数据中位置
            pos_peak_idx_raw = pos_peak_idx_raw[:, 1:].argmax(dim=-1) + neg_peak_idx_smooth[:-1] + 1
            self.del_var()
        else:
            # 当显存溢出, 使用chunk的方法，进行减少GPU显存使用
            neg_idx_diff_sample = neg_idx_diff.repeat_interleave(neg_idx_diff)
            pos_peak_idx_raw = torch.zeros(neg_idx_diff.shape[0], device=self.device).long()
            segments_gpu_index = (torch.tensor(segments_gpu).to(self.device) * neg_idx_diff.max()).long()
            for i in range(len(segments_gpu) - 1):
                # 计算segment内的边索引
                segments_peak_idx = torch.where((neg_idx_diff >= segments_gpu_index[i]) &
                                                (neg_idx_diff < segments_gpu_index[i + 1]))[0]
                segments_sample_idx = torch.where((neg_idx_diff_sample >= segments_gpu_index[i]) &
                                                  (neg_idx_diff_sample < segments_gpu_index[i + 1]))[0]
                if any(segments_peak_idx):
                    neg_idx_diff_temp = neg_idx_diff[segments_peak_idx]
                    # 计算当前segment对应的矩阵
                    matrix_segment = torch.full([neg_idx_diff_temp.shape[0], neg_idx_diff_temp.max()],
                                                -100, device=self.device, dtype=torch.float)
                    matrix_segment[
                        (  # 波谷起始点在原始数据中的位置,
                            torch.arange(neg_idx_diff_temp.shape[0],
                                         device=self.device).repeat_interleave(neg_idx_diff_temp),
                            # 每个采样点在原始数据中的位置 -
                            torch.arange(segments_sample_idx.shape[0], device=self.device) -
                            # 波谷起始点在原始数据中的位置
                            torch.cat((torch.tensor([0], device=self.device),
                                       neg_idx_diff_temp[:-1])).cumsum(dim=0).repeat_interleave(neg_idx_diff_temp)
                        )] = raw_data[neg_peak_idx_smooth[0]:neg_peak_idx_smooth[-1]][segments_sample_idx]
                    # 对于每个波谷，获得最小值（波谷点）位置。将该位置加上波谷起始点在原始数据中位置，获得波谷点在原始数据中位置
                    pos_peak_idx_raw[segments_peak_idx] = matrix_segment[:, 1:].argmax(dim=-1) + \
                                                          neg_peak_idx_smooth[:-1][segments_peak_idx] + 1
                self.del_var()
            del matrix_segment, segments_peak_idx, segments_sample_idx, neg_idx_diff_sample
            self.del_var()

        # Step3.3 矫正一些出现误差的波峰波谷
        # 由于有些采样点没有显著波峰波谷的存在，导致波峰和对应波谷前后位置翻转
        # 波峰出现在对应波谷之后
        # 将异常波峰波谷，设置为对应的平滑数据的波峰波谷
        abnormal_idx = torch.where(pos_peak_idx_raw[:-1] - neg_peak_idx_raw >= 0)[0]
        neg_peak_idx_raw[abnormal_idx] = neg_peak_idx_smooth[abnormal_idx + 1]
        pos_peak_idx_raw[abnormal_idx] = pos_peak_idx_smooth[abnormal_idx]
        # 波峰出现在前一个波谷之前
        # 将异常波峰波谷，设置为对应的平滑数据的波峰波谷
        abnormal_idx = torch.where(pos_peak_idx_raw[1:] - neg_peak_idx_raw <= 0)[0]
        neg_peak_idx_raw[abnormal_idx] = neg_peak_idx_smooth[abnormal_idx + 1]
        pos_peak_idx_raw[abnormal_idx + 1] = pos_peak_idx_smooth[abnormal_idx + 1]
        self.del_var('pos_peak_idx_smooth', 'neg_peak_idx_smooth')

        # -------------------------------- Step 4 --------------------------------
        #             对于每一个波峰波谷上升或者下降边（边），计算半高宽的采样点位置
        #                          输入数据波峰波谷的排列形式：
        #                         .    .   .   .   .   .     波峰
        #                          \  /\  /\  /\  /\  /
        #                           \/  \/  \/  \/  \/
        #                            .   .   .   .   .       波谷
        #                          输出数据波峰波谷的排列形式：
        #                         .    .   .   .   .   .     波峰
        #                          \  /\  /\  /\  /\  /
        #                           \/  \/  \/  \/  \/
        #                            .   .   .   .   .       波谷
        #                        将数据分割成片段，减少GPU显存使用
        # ------------------------------------------------------------------------
        # 对每一个数据片段，使用数据索引的方式，将原始数据转换成边数据矩阵：边个数*时间最长的边的样本数（没有原始数据的位置，赋值为1）。
        # 对于每一个边，获得边的半高宽点：将半高宽点（边数据矩阵每一行与对应半高宽幅度最近的点）,
        #                           加上对应边起始点在整个数据中的位置，得到半高宽点在整个数据中位置。
        # Step4.1 计算每个边的半高宽幅度
        half_edge_amp = torch.cat([torch.stack([raw_data[pos_peak_idx_raw[:-1]],
                                                raw_data[neg_peak_idx_raw]], dim=1).reshape(-1),
                                   raw_data[pos_peak_idx_raw[-1]].reshape(-1)])
        half_edge_amp = ((half_edge_amp[1:] - half_edge_amp[:-1]) / 2 + half_edge_amp[:-1]).reshape(-1, 1)
        self.del_var()

        # Step4.2 计算每个采样点在原始数据中的位置对于边起始点在原始数据中的相对位置
        # 获取每个波峰波谷位置，组成peak变量
        peak = torch.cat([torch.stack([pos_peak_idx_raw[:-1],
                                       neg_peak_idx_raw], dim=1).reshape(-1), pos_peak_idx_raw[-1].reshape(-1)])
        peak_diff = peak.diff()
        # 对于每个边计算，每个采样点在原始数据中的位置 - 边起始点在原始数据中的位置
        matrix_indices = torch.arange(peak[-1] - peak[0]).to(self.device) - torch.cat(
            [torch.tensor([0]).to(self.device), peak_diff[:-1]]).cumsum(dim=0).repeat_interleave(peak_diff)
        self.del_var()

        # Step4.3 计算半高宽的采样点位置
        # 计算时间最长的边的样本数
        max_edge_duration = peak_diff.max()
        if peak_diff.shape[0] * max_edge_duration * 4 / 1024 / 1024 / 1024 < 10:
            # 生成维度为[波峰个数，时间最长波峰的样本数边个数*时间最长的边的样本数]的稀疏矩阵，全部赋值为1
            half_edge_matrix_temp = torch.ones(peak_diff.shape[0] * max_edge_duration,
                                               device=self.device, dtype=torch.float)
            # 对于每个边：边中的所有采样点相对于该边起始点的相对位置 + 每个边起始点在half_edge_matrix_temp中的位置
            half_edge_matrix_temp[
                # 对于每个边
                # 每个采样点在原始数据中的位置, 相对于边起始点在原始数据中的位置
                matrix_indices +
                # 对于每个边，获取边起始点在half_edge_matrix_temp中的位置
                (torch.arange(peak_diff.shape[0]).to(self.device).repeat_interleave(peak_diff)) * max_edge_duration] \
                = raw_data[peak[0]:peak[-1]]
            # half_edge_matrix_temp，reshape成维度为[边个数，时间最长边的样本数]的矩阵
            half_edge_matrix_temp = half_edge_matrix_temp.reshape(peak_diff.shape[0], max_edge_duration)
            self.del_var()
            # 计算半高宽的采样点位置
            half_edge_idx = (half_edge_matrix_temp[:, 1:-1] - half_edge_amp).abs().argmin(dim=1) + 1
        else:
            # 当显存溢出, 使用切片的方式，减少GPU显存使用量
            half_edge_idx = torch.zeros(half_edge_amp.shape[0]).long().to(self.device)
            # 根据切片设置，计算对应的边持续时间，对边数据矩阵进行切片。
            segments_gpu_edge_duration = (torch.tensor(segments_gpu).to(self.device) * max_edge_duration).long()
            for i in range(len(segments_gpu) - 1):
                # 计算边当前切片区间内的边索引
                segments_edge_idx = (peak_diff >= segments_gpu_edge_duration[i]) & \
                                    (peak_diff < segments_gpu_edge_duration[i + 1])
                if any(segments_edge_idx):
                    # 计算边矩阵中，当前切片区间内的边内所有采样点索引
                    segments_matrix_idx = segments_edge_idx.repeat_interleave(peak_diff)
                    # 计算时间切片区间内，最长的边的样本数
                    peak_diff_temp = peak_diff[segments_edge_idx]
                    max_edge_duration_temp = peak_diff_temp.max()
                    # 生成维度为[边个数*时间最长的边的样本数]的矩阵，全部赋值为1
                    half_edge_matrix_temp = torch.ones(peak_diff_temp.shape[0] * max_edge_duration_temp).to(self.device)
                    # 对于每个边：边中的所有采样点相对于该边起始点的相对位置 + 每个边起始点在half_edge_matrix_temp中的位置
                    half_edge_matrix_temp[
                        # 对于每个边
                        # 每个采样点在原始数据中的位置, 相对于边起始点在原始数据中的位置
                        matrix_indices[segments_matrix_idx] +
                        # 对于每个边，获取边起始点在half_edge_matrix_temp中的位置
                        (torch.arange(peak_diff_temp.shape[0]).to(self.device).repeat_interleave(
                            peak_diff_temp)) * max_edge_duration_temp] = raw_data[peak[0]:peak[-1]][segments_matrix_idx]
                    # 将half_edge_matrix_temp，reshape成维度为[边个数，时间最长边的样本数]的矩阵
                    half_edge_matrix_temp = half_edge_matrix_temp.reshape(peak_diff_temp.shape[0],
                                                                          max_edge_duration_temp)
                    self.del_var()
                    # 计算半高宽的采样点位置
                    half_edge_idx[segments_edge_idx] = \
                        (half_edge_matrix_temp[:, 1:-1] - half_edge_amp[segments_edge_idx]).abs().argmin(dim=1) + 1
                    self.del_var()
        half_edge_idx = half_edge_idx + peak[:-1]

        # -------------------------------- Step 5 --------------------------------
        #         将1-d的数据，转换成与原始数据结构一样的2-d形式: n_channel*n_sample
        #                          输入数据波峰波谷的排列形式：
        #                         .    .   .   .   .   .     波峰
        #                          \  /\  /\  /\  /\  /
        #                           \/  \/  \/  \/  \/
        #                            .   .   .   .   .       波谷
        #                          输出数据波峰波谷的排列形式：
        #                          .   .   .   .   .   .     波峰
        #                         /\  /\  /\  /\  /\  /\
        #                        /  \/  \/  \/  \/  \/  \
        #                       0    .   .   .   .   .   n   波谷
        # ------------------------------------------------------------------------
        # 转换raw_data维度：shape(n_channel, n_sample)
        raw_data = raw_data.reshape(n_channel, n_sample)
        pos_peak_idx_raw_2d, pos_peak_raw_2d, neg_peak_idx_raw_2d, neg_peak_raw_2d = [], [], [], []
        half_edge_idx_2d = []
        neg_peak_half_idx_num_2d, pos_peak_half_idx_num_2d = [], []
        for i in range(n_channel):
            # Step5.1: 转换波峰波谷的采样点位置，以及对对应信号幅度，到n_channel*n_sample形式。
            #          使用原始数据样本数n_sample，对1-d数据进行剪裁。
            # 转换波峰位置
            pos_peak_idx_raw_temp = pos_peak_idx_raw[torch.where(
                (pos_peak_idx_raw >= i * n_sample) & (pos_peak_idx_raw < (i + 1) * n_sample))[0]] - i * n_sample
            # 进一步保证波峰位置被正确剪裁
            pos_peak_idx_raw_temp = pos_peak_idx_raw_temp[torch.where((pos_peak_idx_raw_temp > 0) &
                                                                      (pos_peak_idx_raw_temp < n_sample - 1))]
            # 获取波峰采样点对应幅度值
            pos_peak_raw_2d.append(raw_data[i, pos_peak_idx_raw_temp])
            pos_peak_idx_raw_2d.append(pos_peak_idx_raw_temp)
            # 转换波谷位置
            neg_peak_idx_raw_temp = neg_peak_idx_raw[torch.where(
                (neg_peak_idx_raw >= i * n_sample) & (neg_peak_idx_raw < (i + 1) * n_sample))[0]] - i * n_sample
            # 进一步保证波谷位置被正确剪裁（确保对于每个通道，波峰在数据两端）
            neg_peak_idx_raw_temp = neg_peak_idx_raw_temp[
                torch.where((neg_peak_idx_raw_temp > pos_peak_idx_raw_temp[0]) &
                            (neg_peak_idx_raw_temp < pos_peak_idx_raw_temp[-1]))]
            # 获取波谷采样点对应幅度值
            neg_peak_raw_2d.append(torch.cat((raw_data[i][neg_peak_idx_raw_temp[0]].unsqueeze(0),
                                              raw_data[i, neg_peak_idx_raw_temp],
                                              raw_data[i][neg_peak_idx_raw_temp[-1]].unsqueeze(0))))
            neg_peak_idx_raw_2d.append(torch.cat((torch.tensor([0]).to(self.device), neg_peak_idx_raw_temp,
                                                  torch.tensor([n_sample - 1]).to(self.device))))

            # Step5.2: 转换半高宽采样点位置
            #          每个通道，波峰波谷半高宽数量关系为: length(half_edge_idx_temp) = 2 * length(pos_peak_idx_raw_temp) - 2
            #                                        length(pos_peak_idx_raw_temp) = length(neg_peak_idx_raw_temp) + 1
            half_edge_idx_temp = half_edge_idx[torch.where((half_edge_idx >= i * n_sample) &
                                                           (half_edge_idx < (i + 1) * n_sample))[0]] - i * n_sample
            # 进一步保证半高宽采样点位置被正确剪裁
            half_edge_idx_temp = half_edge_idx_temp[torch.where(
                (half_edge_idx_temp > pos_peak_idx_raw_temp[0]) & (half_edge_idx_temp <= pos_peak_idx_raw_temp[-1]))[0]]
            half_edge_idx_2d.append(half_edge_idx_temp)

            # Step5.3: 对于每个波峰波谷，计算波峰或者波谷距离半高宽位置得到采样点个数。
            #          生成一个1*2的向量：[左侧半高宽采样点个数， 右侧半高宽采样点个数]。
            # 对于波谷
            neg_peak_half_idx_num_left = neg_peak_idx_raw_temp - \
                                         half_edge_idx_temp[torch.arange(0, half_edge_idx_temp.shape[0], 2)]
            neg_peak_half_idx_num_right = - neg_peak_idx_raw_temp + \
                                          half_edge_idx_temp[torch.arange(1, half_edge_idx_temp.shape[0], 2)]
            # 将小于1的值，强制设置为1
            neg_peak_half_idx_num_temp = torch.stack([neg_peak_half_idx_num_left, neg_peak_half_idx_num_right])
            neg_peak_half_idx_num_temp[torch.where(neg_peak_half_idx_num_temp < 1)] = 1
            neg_peak_half_idx_num_2d.append(torch.cat((torch.tensor([[1, 1]]).to(self.device).t(),
                                                       neg_peak_half_idx_num_temp,
                                                       torch.tensor([[1, 1]]).to(self.device).t()), dim=-1))
            # 对于波峰
            pos_peak_half_idx_num_left = torch.cat(
                [torch.tensor([0]).to(self.device),
                 pos_peak_idx_raw_temp[1:-1] - half_edge_idx_temp[torch.arange(1, half_edge_idx_temp.shape[0] - 2, 2)],
                 (pos_peak_idx_raw_temp[-1:] - half_edge_idx_temp[-1:])])
            pos_peak_half_idx_num_right = torch.cat(
                [half_edge_idx_temp[:1] - pos_peak_idx_raw_temp[:1],
                 half_edge_idx_temp[torch.arange(2, half_edge_idx_temp.shape[0] - 1, 2)] - pos_peak_idx_raw_temp[1:-1],
                 torch.tensor([0]).to(self.device)])
            # 将小于1的值，强制设置为1
            pos_peak_half_idx_num_temp = torch.stack([pos_peak_half_idx_num_left, pos_peak_half_idx_num_right])
            pos_peak_half_idx_num_temp[torch.where(pos_peak_half_idx_num_temp < 1)] = 1
            pos_peak_half_idx_num_2d.append(pos_peak_half_idx_num_temp)
        self.del_var()

        # -------------------------------- Step 6 --------------------------------
        #                         去除bad_segment中的波峰波谷
        # ------------------------------------------------------------------------
        if bad_segment.any():
            for pos_idx, pos, neg_idx, neg, pos_half, neg_half, i in \
                    zip(pos_peak_idx_raw_2d, pos_peak_raw_2d, neg_peak_idx_raw_2d, neg_peak_raw_2d,
                        pos_peak_half_idx_num_2d, neg_peak_half_idx_num_2d, range(len(neg_peak_half_idx_num_2d))):
                # 计算需要删除的波峰
                pos_idx_delete = torch.where(bad_segment[pos_idx])[0]
                # 计算需要删除的波谷
                neg_idx_delete = torch.where(bad_segment[neg_idx])[0]
                # 计算需要删除的波峰和波谷的并集
                idx_delete = torch.cat([pos_idx_delete, neg_idx_delete]).unique()
                idx_delete = idx_delete[torch.where(idx_delete < pos_idx.shape[0])[0]]
                # 计算需要保留的波峰
                pos_idx_persevere = torch.zeros(pos_idx.shape[0])
                pos_idx_persevere[idx_delete] = -1
                pos_idx_persevere = torch.where(pos_idx_persevere == 0)[0]
                # 计算需要保留的波谷
                neg_idx_persevere = torch.zeros(neg_idx.shape[0])
                neg_idx_persevere[idx_delete] = -1
                neg_idx_persevere = torch.where(neg_idx_persevere == 0)[0]
                # 得到需要保留的数据
                pos_peak_idx_raw_2d[i], pos_peak_raw_2d[i], neg_peak_idx_raw_2d[i], neg_peak_raw_2d[i], \
                pos_peak_half_idx_num_2d[i], neg_peak_half_idx_num_2d[i] = \
                    pos_peak_idx_raw_2d[i][pos_idx_persevere], pos_peak_raw_2d[i][pos_idx_persevere], \
                    neg_peak_idx_raw_2d[i][neg_idx_persevere], neg_peak_raw_2d[i][neg_idx_persevere], \
                    pos_peak_half_idx_num_2d[i][:, pos_idx_persevere], neg_peak_half_idx_num_2d[i][:, neg_idx_persevere]

        signal_peaks = {'pos_peak_idx': pos_peak_idx_raw_2d, 'pos_peak_data': pos_peak_raw_2d,
                        'neg_peak_idx': neg_peak_idx_raw_2d, 'neg_peak_data': neg_peak_raw_2d,
                        'pos_half_peak_idx_num': pos_peak_half_idx_num_2d,
                        'neg_half_peak_idx_num': neg_peak_half_idx_num_2d}
        self.del_var()
        return signal_peaks

    def get_peak_index(self):
        return self.peak_index

    def set_peak_index(self, peak_index=None):
        self.peak_index = peak_index.clone()

    def cal_peak_index(self, pos_peak_idx=None, neg_peak_idx=None):
        """
        Description:
            计算每个波峰波谷的采样点位置
            输入和输出数据波峰波谷的排列形式：
               .   .   .   .   .   .     波峰
              /\  /\  /\  /\  /\  /\
             /  \/  \/  \/  \/  \/  \
            0    .   .   .   .   .   n   波谷

        Input:
            :param pos_peak_idx: list, long, shape(n_channel * 1)
                cal_signal_peaks()的pos_peak_idx输出
            :param neg_peak_idx:  list, long, shape(n_channel * 1)
                cal_signal_peaks()的neg_peak_idx输出

        Return:
            :return peak_index: list, long, shape(n_channel * 1)
                每个波峰和波谷的采样点位置
        """

        # 判断pos_peak_idx和neg_peak_idx是否为None
        if (pos_peak_idx is None) or (neg_peak_idx is None):
            return None
        # 把波峰波谷位置重新排列
        peak_index = [torch.stack([y, torch.cat([x, torch.tensor([0]).to(self.device)])],
                                  dim=-1).reshape(-1)[:-1] for x, y in zip(pos_peak_idx, neg_peak_idx)]

        self.del_var()
        return peak_index

    def get_peak_duration(self):
        return self.peak_duration

    def set_peak_duration(self, peak_duration=None):
        self.peak_duration = peak_duration.clone()

    def cal_peak_duration(self, pos_peak_idx=None, neg_peak_idx=None):
        """
        Description:
            计算每个波峰波谷的持续时间(采样点个数)
            输入和输出数据波峰波谷的排列形式：
               .   .   .   .   .   .     波峰
              /\  /\  /\  /\  /\  /\
             /  \/  \/  \/  \/  \/  \
            0    .   .   .   .   .   n   波谷

        Input:
            :param pos_peak_idx: list, long, shape(n_channel * 1)
                cal_signal_peaks()的pos_peak_idx输出
            :param neg_peak_idx:  list, long, shape(n_channel * 1)
                cal_signal_peaks()的neg_peak_idx输出

        Return:
            :return peak_duration: list, long, shape(n_channel * 1)
                每个波峰和波谷的持续时间
        """

        # 判断pos_peak_idx和neg_peak_idx是否为None
        if (pos_peak_idx is None) or (neg_peak_idx is None):
            return None
        # 计算下降边的持续时间：波峰-右侧波谷
        peak_down_dur = [y[1:] - x for x, y in zip(pos_peak_idx, neg_peak_idx)]
        # 计算上升边的持续时间：波峰-左侧波谷
        peak_up_dur = [x - y[:-1] for x, y in zip(pos_peak_idx, neg_peak_idx)]
        # 排列上升边和下降边为：上升边，下降边，上升边，下降边，....., 上升边，下降边
        peak_up_down_dur = [torch.stack([x, y], dim=-1).reshape(-1) for x, y in zip(peak_up_dur, peak_down_dur)]
        # 计算每个波峰波谷的平均峰值， 自动补充数据两端波谷峰值
        peak_duration = [torch.cat((torch.tensor([0]).to(self.device), x[:-1] + x[1:],
                                    torch.tensor([0]).to(self.device))) for x in peak_up_down_dur]

        self.del_var()
        return peak_duration

    def get_half_peak_duration(self):
        return self.peak_half_duration

    def set_half_peak_duration(self, peak_half_duration=None):
        self.peak_half_duration = peak_half_duration.clone()

    def cal_half_peak_duration(self, pos_half_peak_idx_num=None, neg_half_peak_idx_num=None):
        """
        Description:
            计算每个波峰波谷的半高宽持续时间(采样点个数)
            输入和输出数据波峰波谷的排列形式：
               .   .   .   .   .   .     波峰
              /\  /\  /\  /\  /\  /\
             /  \/  \/  \/  \/  \/  \
            0    .   .   .   .   .   n   波谷

        Input:
            :param pos_half_peak_idx_num: list, long, shape(n_channel * 1)
                cal_signal_peaks()的pos_half_peak_idx_num输出
            :param neg_half_peak_idx_num:  list, long, shape(n_channel * 1)
                cal_signal_peaks()的neg_half_peak_idx_num输出

        Return:
            :return peak_half_duration: list, long, shape(n_channel * 1)
                每个波峰和波谷的半波峰持续时间
        """

        # 判断pos_half_peak_idx_num和neg_half_peak_idx_num是否为None
        if (pos_half_peak_idx_num is None) or (neg_half_peak_idx_num is None):
            return None
        # 计算波峰的半高宽持续时间: 波峰左侧半高宽采样点 + 波峰右侧半高宽采样点
        pos_half_peak_duration = [x.sum(dim=0) for x in pos_half_peak_idx_num]
        # 计算波谷的半高宽持续时间: 波谷左侧半高宽采样点 + 波谷右侧半高宽采样点
        neg_half_peak_duration = [x.sum(dim=0) for x in neg_half_peak_idx_num]
        # 重新排列
        peak_half_duration = [torch.stack([y, torch.cat([x, torch.tensor([0]).to(self.device)])],
                                          dim=-1).reshape(-1)[:-1]
                              for x, y in zip(pos_half_peak_duration, neg_half_peak_duration)]

        self.del_var()
        return peak_half_duration

    def get_peak_amplitude(self):
        return self.peak_amplitude

    def set_peak_amplitude(self, peak_amplitude=None):
        self.peak_amplitude = peak_amplitude.clone()

    def cal_peak_amplitude(self, pos_peak_data=None, neg_peak_data=None):
        """
        Description:
            计算每个波峰波谷的平均峰峰值，(左侧边+右侧边)/2
            输入和输出数据波峰波谷的排列形式：
               .   .   .   .   .   .     波峰
              /\  /\  /\  /\  /\  /\
             /  \/  \/  \/  \/  \/  \
            0    .   .   .   .   .   n   波谷

        Input:
            :param pos_peak_data: list, double, shape(n_channel * 1)
                cal_signal_peaks()的pos_peak_data输出
            :param neg_peak_data: list, double, shape(n_channel * 1)
                cal_signal_peaks()的neg_peak_data输出

        Return:
            :return peak_amplitude: list, double, shape(n_channel * 1)
                每个波峰和波谷的平均峰峰值
        """

        # 判断pos_peak_data和neg_peak_data是否为None
        if (pos_peak_data is None) or (neg_peak_data is None):
            return None
        # 计算下降边的幅度：波峰-右侧波谷
        peak_down_amp = [x - y[1:] for x, y in zip(pos_peak_data, neg_peak_data)]
        # 计算上升边的幅度：波峰-左侧波谷
        peak_up_amp = [x - y[:-1] for x, y in zip(pos_peak_data, neg_peak_data)]
        # 排列上升边和下降边为：上升边，下降边，上升边，下降边，....., 上升边，下降边
        peak_up_down_amp = [torch.stack([x, y], dim=-1).reshape(-1) for x, y in zip(peak_up_amp, peak_down_amp)]
        # 计算每个波峰波谷的平均峰值， 自动补充数据两端波谷峰值
        peak_amplitude = [torch.cat((torch.tensor([0]).to(self.device), (x[:-1] + x[1:]) / 2,
                                     torch.tensor([0]).to(self.device))) for x in peak_up_down_amp]

        self.del_var()
        return peak_amplitude

    def get_peak_slope(self):
        return self.peak_slope

    def set_peak_slope(self, peak_slope=None):
        self.peak_slope = peak_slope.clone()

    def cal_peak_slope(self, pos_peak_data=None, neg_peak_data=None, pos_peak_idx=None, neg_peak_idx=None):
        """
        Description:
            计算每个波峰波谷的平均斜率，(左侧边+右侧边)/2
            输入和输出数据波峰波谷的排列形式：
               .   .   .   .   .   .     波峰
              /\  /\  /\  /\  /\  /\
             /  \/  \/  \/  \/  \/  \
            0    .   .   .   .   .   n   波谷

        Input:
            :param pos_peak_idx: list, long, shape(n_channel * 1)
                cal_signal_peaks()的pos_peak_idx输出
            :param neg_peak_idx: list, long, shape(n_channel * 1)
                cal_signal_peaks()的neg_peak_idx输出
            :param pos_peak_data: list, double, shape(n_channel * 1)
                cal_signal_peaks()的pos_peak_data输出
            :param neg_peak_data: list, double, shape(n_channel * 1)
                cal_signal_peaks()的neg_peak_data输出

        Return:
            :return peak_slope: list, double, shape(n_channel * 1)
                每个波峰和波谷的平均斜率
        """

        # 判断pos_peak_data, neg_peak_data, pos_peak_idx, neg_peak_idx
        if (pos_peak_data is None) or (neg_peak_data is None) or (pos_peak_idx is None) or (neg_peak_idx is None):
            return None
        # 计算下降边的幅度：波峰-右侧波谷
        peak_down_amp = [x - y[1:] for x, y in zip(pos_peak_data, neg_peak_data)]
        # 计算上升边的幅度：波峰-左侧波谷
        peak_up_amp = [x - y[:-1] for x, y in zip(pos_peak_data, neg_peak_data)]
        # 计算下降边的持续时间：波峰-右侧波谷
        peak_down_dur = [y[1:] - x for x, y in zip(pos_peak_idx, neg_peak_idx)]
        # 计算上升边的持续时间：波峰-左侧波谷
        peak_up_dur = [x - y[:-1] for x, y in zip(pos_peak_idx, neg_peak_idx)]
        # 计算下降边的斜率
        peak_down_slope = [x / y for x, y in zip(peak_down_amp, peak_down_dur)]
        # 计算上升边的斜率
        peak_up_slope = [x / y for x, y in zip(peak_up_amp, peak_up_dur)]
        # 排列上升边和下降边为：上升边，下降边，上升边，下降边，....., 上升边，下降边
        peak_up_down_slope = [torch.stack([x, y], dim=-1).reshape(-1) for x, y in zip(peak_up_slope, peak_down_slope)]
        # 计算每个波峰波谷的平均峰值， 自动补充数据两端波谷峰值
        peak_slope = [torch.cat((torch.tensor([0]).to(self.device), (x[:-1] + x[1:]) / 2,
                                 torch.tensor([0]).to(self.device))) for x in peak_up_down_slope]

        self.del_var()
        return peak_slope

    def get_half_peak_slope(self):
        return self.half_peak_slope

    def set_half_peak_slope(self, half_peak_slope=None):
        self.half_peak_slope = half_peak_slope.clone()

    def cal_half_peak_slope(self, pos_peak_data=None, neg_peak_data=None,
                            pos_half_peak_idx_num=None, neg_half_peak_idx_num=None):
        """
        Description:
            计算每个波峰波谷的半高宽平均斜率，(左侧边+右侧边)/2
            输入和输出数据波峰波谷的排列形式：
               .   .   .   .   .   .     波峰
              /\  /\  /\  /\  /\  /\
             /  \/  \/  \/  \/  \/  \
            0    .   .   .   .   .   n   波谷

        Input:
            :param pos_half_peak_idx_num: list, long, shape(n_channel * 1)
                cal_signal_peaks()的pos_half_peak_idx_num输出
            :param neg_half_peak_idx_num: list, long, shape(n_channel * 1)
                cal_signal_peaks()的neg_half_peak_idx_num输出
            :param pos_peak_data: list, double, shape(n_channel * 1)
                cal_signal_peaks()的pos_peak_data输出
            :param neg_peak_data: list, double, shape(n_channel * 1)
                cal_signal_peaks()的neg_peak_data输出

        Return:
            :return half_peak_slope: list, double, shape(n_channel * 1)
                每个波峰和波谷的平均半高宽斜率
        """

        # 判断pos_peak_data, neg_peak_data, pos_half_peak_idx_num, neg_half_peak_idx_num
        if (pos_peak_data is None) or (neg_peak_data is None) or \
                (pos_half_peak_idx_num is None) or (neg_half_peak_idx_num is None):
            return None
        # 计算下降边和上升边的半高宽幅度：上升边，下降边，上升边，下降边，....., 上升边，下降边
        peak_up_down_amp = [torch.stack([(x - y[:-1]) / 2, (x - y[1:]) / 2], dim=1).reshape(-1)
                            for x, y in zip(pos_peak_data, neg_peak_data)]
        # 计算波峰的半高宽斜率: 波峰左侧半高宽采样点 + 波峰右侧半高宽采样点
        pos_half_peak_slope = [(y[torch.arange(0, y.shape[0] - 1, 2)] / x[0] +
                                y[torch.arange(1, y.shape[0] - 0, 2)] / x[1]) / 2
                               for x, y in zip(pos_half_peak_idx_num, peak_up_down_amp)]
        # 计算波谷的半高宽持续时间: 波谷左侧半高宽采样点 + 波谷右侧半高宽采样点
        neg_half_peak_slope = [(y[torch.arange(1, y.shape[0] - 2, 2)] / x[0, 1:-1] +
                                y[torch.arange(2, y.shape[0] - 1, 2)] / x[1, 1:-1]) / 2
                               for x, y in zip(neg_half_peak_idx_num, peak_up_down_amp)]
        # 重新排列
        half_peak_slope = [torch.cat([torch.stack([torch.cat([x[:1], y]), x], dim=-1).reshape(-1), x[-1:]])
                           for x, y in zip(pos_half_peak_slope, neg_half_peak_slope)]

        self.del_var()
        return half_peak_slope

    def get_peak_sharpness(self):
        return self.peak_sharpness

    def set_peak_sharpness(self, peak_sharpness=None):
        self.peak_sharpness = peak_sharpness.clone()

    def cal_peak_sharpness(self, pos_peak_idx=None, neg_peak_idx=None, time_n=5):
        """
        Description:
            计算每个波峰波谷的锐度(二阶倒数)
            计算公式：dd_i=(y(i) - y(i-n)) - (y(i+n) - y(i)) + (y(i) - y(i-n+1)) - (y(i+n-1) - y(i)) + ... +
                         (y(i) - y(i+1)) - (y(i-1) - y(i))
                    y是MEG信号, i是波峰波谷采样点位置, n是采样点间隔个数
            输入和输出数据波峰波谷的排列形式：
               .   .   .   .   .   .     波峰
              /\  /\  /\  /\  /\  /\
             /  \/  \/  \/  \/  \/  \
            0    .   .   .   .   .   n   波谷

        Input:
            :param pos_peak_idx: list, long, shape(n_channel * 1)
                cal_signal_peaks()的pos_peak_idx输出
            :param neg_peak_idx: list, double, shape(n_channel * 1)
                cal_signal_peaks()的neg_peak_idx输出
            :param time_n: number, double
                计算公式中的n，代表锐度的计算范围，单位为毫秒

        Return:
            :return peak_sharpness: list, double, shape(n_channel * 1)
                每个波峰和波谷的锐度
        """

        # 判断pos_peak_data, neg_peak_data, pos_half_peak_idx_num, neg_half_peak_idx_num
        if (pos_peak_idx is None) or (neg_peak_idx is None):
            return None
        sample_n = max(1, int(time_n / 1000 * self.data_info['sfreq']))
        # 用补零的方式扩展MEG数据，防止超出范围报错
        n_ex = sample_n * 2
        meg_data_ex = self.get_raw_data().clone()
        meg_data_ex = torch.cat([torch.zeros(meg_data_ex.shape[0], n_ex), meg_data_ex,
                                 torch.zeros(meg_data_ex.shape[0], n_ex)], dim=1).to(self.device)
        # 计算波峰的锐度
        pos_peak_sharpness = [(y[x + n_ex].unsqueeze(-1).repeat(1, sample_n) * 2 -
                               y[x + n_ex + torch.arange(-sample_n, 0).unsqueeze(-1).to(self.device)].t() -
                               y[x + n_ex + torch.arange(1, sample_n + 1).unsqueeze(-1).to(self.device)].t()).sum(dim=1)
                              for x, y in zip(pos_peak_idx, meg_data_ex)]
        # 计算波谷的锐度
        neg_peak_sharpness = [(-y[x + n_ex].unsqueeze(-1).repeat(1, sample_n) * 2 +
                               y[x + n_ex + torch.arange(-sample_n, 0).unsqueeze(-1).to(self.device)].t() +
                               y[x + n_ex + torch.arange(1, sample_n + 1).unsqueeze(-1).to(self.device)].t()).sum(dim=1)
                              for x, y in zip(neg_peak_idx, meg_data_ex)]
        # 重新排列
        peak_sharpness = [torch.stack([y, torch.cat([x, torch.tensor([0]).to(self.device)])],
                                      dim=-1).reshape(-1)[:-1]
                          for x, y in zip(pos_peak_sharpness, neg_peak_sharpness)]

        self.del_var()
        return peak_sharpness

    def get_all_features(self):
        return self.peak_index, self.peak_duration, self.peak_half_duration, \
               self.peak_amplitude, self.peak_slope, self.half_peak_slope, self.peak_sharpness

    def set_all_features(self, peak_index=None, peak_duration=None, peak_half_duration=None, peak_amplitude=None,
                         peak_slope=None, half_peak_slope=None, peak_sharpness=None):
        self.peak_index, self.peak_duration, self.peak_half_duration, \
        self.peak_amplitude, self.peak_slope, self.half_peak_slope, self.peak_sharpness = \
            peak_index, peak_duration, peak_half_duration, peak_amplitude, peak_slope, half_peak_slope, peak_sharpness

    def cal_all_features(self, pos_peak_idx=None, neg_peak_idx=None, pos_peak_data=None, neg_peak_data=None,
                         pos_half_peak_idx_num=None, neg_half_peak_idx_num=None, time_n=5):
        """
        Description:
            计算所有features
            输入和输出数据波峰波谷的排列形式：
               .   .   .   .   .   .     波峰
              /\  /\  /\  /\  /\  /\
             /  \/  \/  \/  \/  \/  \
            0    .   .   .   .   .   n   波谷

        Input:
            :param pos_peak_idx: list, double, shape(n_channel * 1)
                cal_signal_peaks()的pos_peak_idx输出
            :param neg_peak_idx: list, double, shape(n_channel * 1)
                cal_signal_peaks()的neg_peak_idx输出
            :param pos_peak_data: list, double, shape(n_channel * 1)
                cal_signal_peaks()的pos_peak_data输出
            :param neg_peak_data: list, double, shape(n_channel * 1)
                cal_signal_peaks()的neg_peak_data输出
            :param pos_half_peak_idx_num: list, double, shape(n_channel * 1)
                cal_signal_peaks()的pos_half_peak_idx_num输出
            :param neg_half_peak_idx_num: list, double, shape(n_channel * 1)
                cal_signal_peaks()的neg_half_peak_idx_num输出
            :param time_n: number, double
                锐度的计算范围，单位为毫秒

        Return:
            :return peak_index: list, long, shape(n_channel * 1)
                每个波峰和波谷的采样点位置
            :return peak_duration: list, double, shape(n_channel * 1)
                每个波峰和波谷的时间长度
            :return peak_half_duration: list, double, shape(n_channel * 1)
                每个波峰和波谷的半高宽时间长度
            :return peak_amplitude: list, double, shape(n_channel * 1)
                每个波峰和波谷的平均峰峰值
            :return peak_slope: list, double, shape(n_channel * 1)
                每个波峰和波谷的平均斜率
            :return half_peak_slope: list, double, shape(n_channel * 1)
                每个波峰和波谷的平均半高宽斜率
            :return peak_sharpness: list, double, shape(n_channel * 1)
                每个波峰和波谷的锐度
        """
        peak_index = self.cal_peak_index(pos_peak_idx=pos_peak_idx, neg_peak_idx=neg_peak_idx)
        peak_duration = self.cal_peak_duration(pos_peak_idx=pos_peak_idx, neg_peak_idx=neg_peak_idx)
        peak_half_duration = self.cal_half_peak_duration(pos_half_peak_idx_num=pos_half_peak_idx_num,
                                                         neg_half_peak_idx_num=neg_half_peak_idx_num)
        peak_amplitude = self.cal_peak_amplitude(pos_peak_data=pos_peak_data, neg_peak_data=neg_peak_data)
        peak_slope = self.cal_peak_slope(pos_peak_data=pos_peak_data, neg_peak_data=neg_peak_data,
                                         pos_peak_idx=pos_peak_idx, neg_peak_idx=neg_peak_idx)
        half_peak_slope = self.cal_half_peak_slope(pos_peak_data=pos_peak_data, neg_peak_data=neg_peak_data,
                                                   pos_half_peak_idx_num=pos_half_peak_idx_num,
                                                   neg_half_peak_idx_num=neg_half_peak_idx_num)
        peak_sharpness = self.cal_peak_sharpness(pos_peak_idx=pos_peak_idx, neg_peak_idx=neg_peak_idx, time_n=time_n)

        return peak_index, peak_duration, peak_half_duration, \
               peak_amplitude, peak_slope, half_peak_slope, peak_sharpness

    def zscore_feature(self, feature=None, z_win=None, z_region=True, z_mag_grad=False, exclude_sample=None):
        """
        Description:
            对输入的feature进行归一化(z-score)
            可选择的z-score方式：
                -----------------------均值和方差使用(单个通道*全部数据)内数据进行计算----------------------
                (1). z_win=None, z_region=False, z_mag_grad=False
                -----------------------均值和方差使用(单个通道*时间窗)内数据进行计算------------------------
                (2). 时间窗为z_win。z_win=大于1, z_region=False, z_mag_grad=False
                (3). 时间窗为z_win*数据总长。z_win=0到1之间, z_region=False, z_mag_grad=False
                -----------------------均值和方差使用(多个通道*全部数据)内数据进行间计算---------------------
                (4). 306通道按照MAG和GRAD分为2个Region。z_win=None, z_region=False, z_mag_grad=True
                (5). 306通道按照脑区分为8个Region。并使用z_win=None, z_region=True, z_mag_grad=False
                -----------------------均值和方差使用(多个通道*时间窗)内数据进行间计算-----------------------
                (6). 时间窗为z_win，306通道按照MAG和GRAD分为2个Region。z_win=大于1, z_region=False, z_mag_grad=True
                (7). 时间窗为z_win*数据总长，306通道按照MAG和GRAD分为2个Region。z_win=0到1之间, z_region=False, z_mag_grad=True
                (8). 时间窗为z_win，306通道按照脑区分为8个Region。z_win=大于1, z_region=True, z_mag_grad=False
                (9). 时间窗为z_win*数据总长，306通道按照脑区分为8个Region。z_win=0到1之间, z_region=True, z_mag_grad=False
            Note: 当使用多个通道计算均值和方差时，为了保证特征在(单通道*时间窗)内和(多个通道*时间窗)内都很突出，使用两者间的最小值。

        Input:
            :param feature: list, double, shape(n_channel * 1)
                需要归一化的特征
            :param z_win: number, double
                滑动窗长度：z_win=大于1，时间窗为z_win；z_win=0到1之间，时间窗为z_win*数据总长；z_win=None，时间窗为数据总长。
            :param z_region: bool
                是否通过脑区选择多通道。
            :param z_mag_grad: bool
                是否通过MAG/GRAD选择多通道。
            :param exclude_sample: list, bool, shape(n_channel * 1)
                排除的样本。输出时，将对应位置设置为-1.

        Return:
            :return zscore_feature: list, double, shape(n_channel * 1)
                归一化后的feature
        """

        # 判断feature是否存在
        if feature is None:
            return None

        # Step1: 删除需要去除的采样点
        if exclude_sample is not None:
            feature_shape = [x.shape[0] for x in feature]
            feature = [x[~y] for x, y in zip(feature, exclude_sample)]

        # Step2: 获得计算z-score的窗长度
        if (z_win is not None) and (z_win > 0) and (z_win < 1):
            # 当z_win在0到1之间，窗长使用int(z_win*feature长度)
            z_win = max(z_win, 0.1)
            z_win = torch.tensor(int(z_win * torch.tensor([x.shape[0] for x in feature]).float().mean()))
            z_win_half = (torch.floor(z_win / 2)).long().tolist()
            z_win = torch.nn.Unfold(kernel_size=(1, z_win_half * 2 + 1)).to(self.device)
        elif (z_win is not None) and (z_win > 1):
            # 当z_win大于1，窗长使用z_win
            z_win = torch.tensor(min(z_win, torch.tensor([x.shape[0] for x in feature]).float().mean()))
            z_win_half = (torch.floor(z_win / 2)).long().tolist()
            z_win = torch.nn.Unfold(kernel_size=(1, z_win_half * 2 + 1)).to(self.device)
        else:
            # 当z_win为None, 使用全段数据计算z-score
            z_win = None

        # Step3: 计算多通道间z-score, 获得计算z-score的通道范围
        if not z_region and not z_mag_grad:
            # 对每个通道计算z-score
            z_region = None
        elif not z_region and z_mag_grad:
            # 对所有mag或者所有grad同到计算z-score
            z_region = torch.tensor([[j for j, z in enumerate([self.chan_type['grad'], self.chan_type['mag']])
                                      if x in z] for x in range(len(feature))])
        else:
            # 对每个脑区的mag和grad通道计算z-score
            z_region = torch.tensor([[i + j * len(self.chan_region)
                                      for i, y in enumerate(self.chan_region) if x in y
                                      for j, z in enumerate([self.chan_type['grad'], self.chan_type['mag']])
                                      if x in z] for x in range(len(feature))])

        # Step4: 计算z-score
        if z_win is not None:
            # 使用滑动时间窗进行z-score
            # 对于每个采样点，计算对应滑动时间窗内的数据均值和方差
            win_feature_std_mean = [torch.std_mean(z_win(x.unsqueeze(0).unsqueeze(0).unsqueeze(0)).squeeze(),
                                                   dim=0) for x in feature]
            # 对滑动时间窗结果进行Padding
            win_feature_std_mean = [(torch.cat((x[0][0].repeat(z_win_half), x[0], x[0][-1].repeat(z_win_half))),
                                     torch.cat((x[1][0].repeat(z_win_half), x[1], x[1][-1].repeat(z_win_half))))
                                    for x in win_feature_std_mean]
            # 计算通道内的z-score
            feature_zscore = [(x - y[1]) / y[0] for x, y in zip(feature, win_feature_std_mean)]
            # 计算(应滑动时间窗*通道)内的z-score
            if z_region is not None:
                # 获得每个region的通道
                feature_region = [[feature[y] for y in torch.where(z_region == x)[0]] for x in z_region.unique()]
                # 对于每个region内的所有通道，使用通道内的最小数据长度，进行截取
                region_shape = [torch.tensor([y.shape[0] for y in x]).min() for x in feature_region]
                feature_region = [torch.stack([z[:y] for z in x]) for x, y in zip(feature_region, region_shape)]
                # 计算(应滑动时间窗*通道)内的数据均值和方差
                win_feature_std_mean = [torch.std_mean(z_win(x.unsqueeze(1).unsqueeze(1)).squeeze(), dim=(0, 1))
                                        for x in feature_region]
                # 对结果进行Padding（由于滑动窗产生的数据截断）
                win_feature_std_mean = [(torch.cat((x[0][0].repeat(z_win_half), x[0], x[0][-1].repeat(z_win_half))),
                                         torch.cat((x[1][0].repeat(z_win_half), x[1], x[1][-1].repeat(z_win_half))))
                                        for x in win_feature_std_mean]
                # 对结果进行Padding（由于产生的数据截断）
                win_feature_std_mean = [(torch.cat((win_feature_std_mean[x][0],
                                                    win_feature_std_mean[x][0][-1].repeat(
                                                        feature[i].shape[0] - win_feature_std_mean[x][0].shape[0]))),
                                         torch.cat((win_feature_std_mean[x][1],
                                                    win_feature_std_mean[x][1][-1].repeat(
                                                        feature[i].shape[0] - win_feature_std_mean[x][1].shape[0]))))
                                        for i, x in enumerate(z_region)]
                # 计算(应滑动时间窗*通道)内z-score
                feature_region = [(x - y[1]) / y[0] for x, y in zip(feature, win_feature_std_mean)]
                # 为了保证特征在通道内和(应滑动时间窗*通道)内，都很突出。使用两者间的最小值
                feature_zscore = [torch.stack((x, y), dim=0).amin(dim=0)
                                  for x, y in zip(feature_region, feature_zscore)]
        else:
            # 使用整段数据进行z-score
            # 计算通道内的z-score
            feature_zscore = [(x - x.mean()) / x.std() for x in feature]
            # 计算(应滑动时间窗*通道)内的z-score
            if z_region:
                # 获得每个region的通道
                feature_region = [[feature[y] for y in torch.where(z_region == x)[0]] for x in z_region.unique()]
                # 计算region内的均值和方差
                win_feature_std_mean = [torch.std_mean(torch.cat(x)) for x in feature_region]
                win_feature_std_mean = [win_feature_std_mean[x] for x in z_region]
                # 计算region内z-score
                feature_region = [(x - y[1]) / y[0] for x, y in zip(feature, win_feature_std_mean)]
                # 为了保证特征在通道内和region内，都很突出。使用两者间的最小值
                feature_zscore = [torch.stack((x, y), dim=0).amin(dim=0)
                                  for x, y in zip(feature_region, feature_zscore)]
        self.del_var()

        # Step5: 将删除采样点设置为-1
        if exclude_sample is not None:
            for x, y, z, i in zip(feature_zscore, exclude_sample, feature_shape, range(len(feature_zscore))):
                temp = torch.ones(z).to(self.device) * -1.
                temp[~y] = x.float()
                feature_zscore[i] = temp

        self.del_var()
        return feature_zscore

    def get_zscored_features(self):
        return self.peak_amplitude_zscore, self.peak_slope_zscore, \
               self.half_peak_slope_zscore, self.peak_sharpness_zscore

    def set_zscored_features(self, peak_amplitude_zscore=None, peak_slope_zscore=None,
                             half_peak_slope_zscore=None, peak_sharpness_zscore=None):
        self.peak_amplitude_zscore, self.peak_slope_zscore, self.half_peak_slope_zscore, self.peak_sharpness_zscore \
            = peak_amplitude_zscore, peak_slope_zscore, half_peak_slope_zscore, peak_sharpness_zscore

    def cal_zscored_features(self, peak_amplitude=None, peak_slope=None, half_peak_slope=None, peak_sharpness=None,
                             peak_duration=None, exclude_duration=0.02, z_win=None, z_region=True, z_mag_grad=False):
        """
        Description:
            对所有features进行z-score

        Input:
            :param peak_amplitude: list, double, shape(n_channel * 1)
                每个波峰和波谷的平均峰峰值
            :param peak_slope: list, double, shape(n_channel * 1)
                每个波峰和波谷的平均斜率
            :param half_peak_slope: list, double, shape(n_channel * 1)
                每个波峰和波谷的平均半高宽斜率
            :param peak_sharpness: list, double, shape(n_channel * 1)
                每个波峰和波谷的锐度
            :param peak_duration: list, double, shape(n_channel * 1)
                每个波峰和波谷的时间长度
            :param exclude_duration: number, double
                去除持续时间小于该阈值的峰
            :param z_win: number, double
                滑动窗长度：z_win=大于1，时间窗为z_win；z_win=0到1之间，时间窗为z_win*数据总长；z_win=None，时间窗为数据总长。
            :param z_region: bool
                是否通过脑区选择多通道。
            :param z_mag_grad: bool
                是否通过MAG/GRAD选择多通道。

        Return:
            :return peak_amplitude_zscore: list, double, shape(n_channel * 1)
                每个波峰和波谷的z-score峰值
            :return peak_slope_zscore: list, double, shape(n_channel * 1)
                每个波峰和波谷的z-score斜率
            :return half_peak_slope_zscore: list, double, shape(n_channel * 1)
                每个波峰和波谷的z-score半高宽斜率
            :return peak_sharpness_zscore: list, double, shape(n_channel * 1)
                每个波峰和波谷的z-score锐度
        """
        # 去除持续时间小于阈值的峰
        if (peak_duration is not None) and (exclude_duration is not None):
            exclude_duration = exclude_duration * self.data_info['sfreq']
            exclude_sample = [(x < exclude_duration).to(self.device) for x in peak_duration]
        else:
            exclude_sample = None

        # 计算z-score
        peak_amplitude_zscore = self.zscore_feature(feature=peak_amplitude, exclude_sample=exclude_sample,
                                                    z_win=z_win, z_region=z_region, z_mag_grad=z_mag_grad)
        peak_slope_zscore = self.zscore_feature(feature=peak_slope, exclude_sample=exclude_sample,
                                                z_win=z_win, z_region=z_region, z_mag_grad=z_mag_grad)
        half_peak_slope_zscore = self.zscore_feature(feature=half_peak_slope, exclude_sample=exclude_sample,
                                                     z_win=z_win, z_region=z_region, z_mag_grad=z_mag_grad)
        peak_sharpness_zscore = self.zscore_feature(feature=peak_sharpness, exclude_sample=exclude_sample,
                                                    z_win=z_win, z_region=z_region, z_mag_grad=z_mag_grad)

        self.del_var()
        return peak_amplitude_zscore, peak_slope_zscore, half_peak_slope_zscore, peak_sharpness_zscore

    def threshold_feature(self, feature=None, threshold=None):
        """
        Description:
            对feature进行阈值处理。

        Input:
            :param feature: list, bool, shape(n_channel * 1)
                需要进行阈值处理的特征值
            :param threshold: list, double, shape(2 * 1)
                [下阈值， 上阈值], 非None时，参与计算

        Return:
            :return picked_peak: list, bool, shape(n_channel * 1)
                feature中每个值是否满足阈值
        """

        # 判断feature和threshold是否存在
        if (feature is None) or (threshold[0] is None and threshold[1] is None):
            return None
        # 计算feature满足threshold的index
        picked_peak = None
        # 大于下阈值
        if threshold[0] is not None:
            picked_peak = [x >= threshold[0] for x in feature]
        # 小于上阈值
        if threshold[1] is not None:
            if picked_peak is None:
                picked_peak = [x <= threshold[1] for x in feature]
            else:
                picked_peak = [(x <= threshold[1]) & y for x, y in zip(feature, picked_peak)]

        self.del_var()
        return picked_peak

    def get_picked_peak(self):
        return self.picked_peak

    def set_picked_peak(self, picked_peak=None):
        self.picked_peak = picked_peak

    def cal_picked_peaks(self, peak_amplitude=None, peak_slope=None, half_peak_slope=None, peak_sharpness=None,
                         peak_amplitude_threshold=(None, None), peak_slope_threshold=(None, None),
                         half_peak_slope_threshold=(None, None), peak_sharpness_threshold=(None, None)):
        """
        Description:
            对所有feature进行阈值处理，获得满足阈值要求的peaks
            当时输入的feature或者对应阈值为None(或者使用默认值)，该feature不影响picked_peak

        Input:
            :param peak_amplitude: list, double, shape(n_channel * 1)
                每个波峰和波谷的平均峰峰值
            :param peak_slope: list, double, shape(n_channel * 1)
                每个波峰和波谷的平均斜率
            :param half_peak_slope: list, double, shape(n_channel * 1)
                每个波峰和波谷的平均半高宽斜率
            :param peak_sharpness: list, double, shape(n_channel * 1)
                每个波峰和波谷的锐度
            :param peak_amplitude_threshold: list, double, shape(2 * 1)
                峰峰值阈值：[下阈值， 上阈值]
            :param peak_slope_threshold: list, double, shape(2 * 1)
                斜率阈值：[下阈值， 上阈值]
            :param half_peak_slope_threshold: list, double, shape(2 * 1)
                半高宽斜率阈值：[下阈值， 上阈值]
            :param peak_sharpness_threshold: list, double, shape(2 * 1)
                锐度阈值：[下阈值， 上阈值]

        Return:
            :return picked_peak: list, bool, shape(n_channel * 1)
                feature中每个值是否满足阈值
        """
        feature_index_amp = self.threshold_feature(feature=peak_amplitude, threshold=peak_amplitude_threshold)
        feature_index_slope = self.threshold_feature(feature=peak_slope, threshold=peak_slope_threshold)
        feature_index_half_slope = self.threshold_feature(feature=half_peak_slope, threshold=half_peak_slope_threshold)
        feature_index_sharpness = self.threshold_feature(feature=peak_sharpness, threshold=peak_sharpness_threshold)
        # 所有非空的feature_index求交集
        picked_peak = None
        for x in [feature_index_amp, feature_index_slope, feature_index_half_slope, feature_index_sharpness]:
            # 判断feature index是否为None
            if x is None:
                continue
            # 所有feature index求交集
            if picked_peak is None:
                picked_peak = x
            else:
                picked_peak = [y & z for y, z in zip(picked_peak, x)]

        self.del_var()
        return picked_peak

    def cal_samples_of_picked_peaks(self, picked_peaks=None, peak_index=None, peak_amplitude=None, peak_win=5):
        """
        Description:
            计算被选择(通过阈值)的peak内所有点(上升边和下降边采样点)，在原始数据(n_channel*n_sample)中的位置。

        Input:
            :param picked_peaks: list, bool, shape(n_channel * 1)
                feature中每个值是否满足阈值
            :param peak_index: list, long, shape(n_channel * 1)
                每个波峰和波谷的采样点位置
            :param peak_amplitude: list, long, shape(n_channel * 1)
                每个波峰和波谷的幅度
            :param peak_win: number, long
                用于确定samples_of_picked_peak的windows长度

        Return:
            :return all_samples_of_picked_peak: torch.tensor, long, shape(n_channel*n_sample)
                peak内所有点(上升边和下降边采样点)，在原始数据(n_channel*n_sample)中对应位置，赋值为0，1，2，3，4，.....
                代表对应采样点属于通道内第几个picked_peaks。
                没有被picked的采样点赋值为-1
            :return win_samples_of_picked_peak: torch.tensor, long, shape(n_channel*n_sample)
                peak点周围peak_win内的采样点，在原始数据(n_channel*n_sample)中对应位置，赋值为0，1，2，3，4，.....
                代表对应采样点属于通道内第几个picked_peaks。
                没有被picked的采样点赋值为-1
            :return samples_of_picked_peak: torch.tensor, long, shape(n_channel*n_sample)
                peak点，在原始数据(n_channel*n_sample)中对应位置，赋值为0，1，2，3，4，.....
                代表对应采样点属于通道内第几个picked_peaks。
                没有被picked的采样点赋值为-1
        """

        # 获取原始数据维度
        raw_data = self.get_raw_data()
        n_channel, n_sample = raw_data.shape[0], raw_data.shape[1]

        # 判断feature和threshold是否存在
        if (picked_peaks is None) or (peak_index is None) or (peak_amplitude is None):
            return (torch.ones(n_channel, n_sample) * -1).long(), (torch.ones(n_channel, n_sample) * -1).long(), \
                   (torch.ones(n_channel, n_sample) * -1).long()

        # 计算每一个通道的picked_peaks在原始数据(n_channel*n_sample)中的位置
        # 使用pad_sequence将list补零转换成数组
        samples_of_picked_peak = (torch.ones(n_channel, n_sample) * -1).long()
        win_samples_of_picked_peak = samples_of_picked_peak.clone()
        all_samples_of_picked_peak = samples_of_picked_peak.clone()
        picked_peaks_temp, peak_index_temp, peak_amplitude_temp = \
            pad_sequence(picked_peaks).t(), pad_sequence(peak_index).t(), pad_sequence(peak_amplitude).t()
        # 获取picked_peaks在二维数组中的位置
        picked_peaks_index = torch.where(picked_peaks_temp)
        # 获取picked_peaks在每个channel中的位置
        peaks_index_in_each_channel = torch.cat([torch.arange(x) for x in picked_peaks_index[0].unique_consecutive(
            return_counts=True)[1]])
        # 对picked_peaks_index以Amplitude进行排列
        temp = peak_amplitude_temp[picked_peaks_index].argsort()
        picked_peaks_index = tuple([x[temp] for x in picked_peaks_index])
        peaks_index_in_each_channel = peaks_index_in_each_channel[temp]
        # 获取每个picked_peaks的channel索引
        peaks_channel_index = picked_peaks_index[0]
        # 获取picked_peaks两边相邻peaks的采样点位置
        adjacent_peaks_beg = peak_index_temp[(picked_peaks_index[0], picked_peaks_index[1] - 1)]
        adjacent_peaks_end = peak_index_temp[(picked_peaks_index[0], picked_peaks_index[1] + 1)]
        # 获取picked_peaks的采样点位置
        peaks = peak_index_temp[picked_peaks_index].cpu()
        peaks[peaks < peak_win] = peak_win
        peaks[peaks > n_sample - peak_win - 1] = n_sample - peak_win - 1
        # 根据picked_peaks的channel索引，channel内采样点索引，进行赋值
        for peak, adjacent_peak_beg, adjacent_peak_end, peak_channel_index, idx in \
                zip(peaks, adjacent_peaks_beg, adjacent_peaks_end, peaks_channel_index, peaks_index_in_each_channel):
            all_samples_of_picked_peak[peak_channel_index, torch.arange(adjacent_peak_beg, adjacent_peak_end)] = idx
            win_samples_of_picked_peak[peak_channel_index, peak + torch.arange(-peak_win, peak_win)] = idx
            samples_of_picked_peak[peak_channel_index, peak] = idx

        self.del_var()
        return all_samples_of_picked_peak, win_samples_of_picked_peak, samples_of_picked_peak

    def get_peaks(self, smooth_windows=0.16, smooth_iterations=2,
                  exclude_duration=0.02, z_win=200, z_region=True, z_mag_grad=False,
                  peak_amplitude_threshold=(None, None), peak_slope_threshold=(None, None),
                  half_peak_slope_threshold=(None, None), peak_sharpness_threshold=(None, None)):
        """
        Description:
            获取MEG信号中的peak；计算每个peak的特征；归一化特征值；对特征进行阈值处理，获得满足阈值要求的peaks。
            Note: 当时输入的应阈值为None(或者使用默认值)，不计算对应的特征值，以及不使用对应的特征值决定peaks的选取。

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

        Return:
            :return peaks: dict
                peaks_sample_index: list, long, shape(n_channel*1)
                    每个通道peak的采样点位置。
                peak_duration: list, long, shape(n_channel*1)
                    每个通道peak的持续时间。
                peak_half_duration: list, long, shape(n_channel*1)
                    每个通道peak的半高宽持续时间。
                all_samples_of_picked_peak: list, long, shape(n_channel*1)
                    peak内所有点(上升边和下降边采样点)，在原始数据(n_channel*n_sample)中对应位置。
                win_samples_of_picked_peak: list, long, shape(n_channel*1)
                    peak点周围peak_win内的采样点，在原始数据(n_channel*n_sample)中对应位置
                samples_of_picked_peak: list, long, shape(n_channel*1)
                    peak点，在原始数据(n_channel*n_sample)中对应位置
                peak_amplitude: list, float, shape(n_channel*1)
                    每个通道peak的峰峰值。
                peak_slope: list, long, shape(n_channel*1)
                    每个通道peak的斜率。
                half_peak_slope: list, long, shape(n_channel*1)
                    每个通道peak的半高宽斜率。
                peak_sharpness: list, long, shape(n_channel*1)
                    每个通道peak的锐度。
        """
        # Step1: 获取MEG信号中的peak
        signal_peaks = self.cal_signal_peaks(smooth_windows=smooth_windows, smooth_iterations=smooth_iterations)
        self.set_signal_peaks(signal_peaks)
        # Step2: 计算每个peak的特征
        peak_index = self.cal_peak_index(pos_peak_idx=signal_peaks['pos_peak_idx'],
                                         neg_peak_idx=signal_peaks['neg_peak_idx'])
        peak_duration = self.cal_peak_duration(pos_peak_idx=signal_peaks['pos_peak_idx'],
                                               neg_peak_idx=signal_peaks['neg_peak_idx'])
        peak_half_duration = self.cal_half_peak_duration(pos_half_peak_idx_num=signal_peaks['pos_half_peak_idx_num'],
                                                         neg_half_peak_idx_num=signal_peaks['neg_half_peak_idx_num'])
        peak_amplitude = self.cal_peak_amplitude(pos_peak_data=signal_peaks['pos_peak_data'],
                                                 neg_peak_data=signal_peaks['neg_peak_data'])
        if (peak_slope_threshold[0] is not None) or (peak_slope_threshold[1] is not None):
            peak_slope = self.cal_peak_slope(pos_peak_data=signal_peaks['pos_peak_data'],
                                             neg_peak_data=signal_peaks['neg_peak_data'],
                                             pos_peak_idx=signal_peaks['pos_peak_idx'],
                                             neg_peak_idx=signal_peaks['neg_peak_idx'])
        else:
            peak_slope = None
        if (half_peak_slope_threshold[0] is not None) or (half_peak_slope_threshold[1] is not None):
            half_peak_slope = self.cal_half_peak_slope(pos_peak_data=signal_peaks['pos_peak_data'],
                                                       neg_peak_data=signal_peaks['neg_peak_data'],
                                                       pos_half_peak_idx_num=signal_peaks['pos_half_peak_idx_num'],
                                                       neg_half_peak_idx_num=signal_peaks['neg_half_peak_idx_num'])
        else:
            half_peak_slope = None
        if (peak_sharpness_threshold[0] is not None) or (peak_sharpness_threshold[1] is not None):
            peak_sharpness = self.cal_peak_sharpness(pos_peak_idx=signal_peaks['pos_peak_idx'],
                                                     neg_peak_idx=signal_peaks['neg_peak_idx'], time_n=5)
        else:
            peak_sharpness = None
        self.set_all_features(peak_index=peak_index, peak_duration=peak_duration, peak_half_duration=peak_half_duration,
                              peak_amplitude=peak_amplitude, peak_slope=peak_slope, half_peak_slope=half_peak_slope,
                              peak_sharpness=peak_sharpness)
        # Step3: 归一化特征值
        peak_amplitude_zscore, peak_slope_zscore, half_peak_slope_zscore, peak_sharpness_zscore = \
            self.cal_zscored_features(peak_amplitude=peak_amplitude, peak_slope=peak_slope,
                                      half_peak_slope=half_peak_slope, peak_sharpness=peak_sharpness,
                                      peak_duration=peak_duration, exclude_duration=exclude_duration,
                                      z_win=z_win, z_region=z_region, z_mag_grad=z_mag_grad)
        self.set_zscored_features(peak_amplitude_zscore=peak_amplitude_zscore, peak_slope_zscore=peak_slope_zscore,
                                  half_peak_slope_zscore=half_peak_slope_zscore,
                                  peak_sharpness_zscore=peak_sharpness_zscore)
        # Step4: 对特征进行阈值处理，获得满足阈值要求的peaks
        picked_peaks = self.cal_picked_peaks(peak_amplitude=peak_amplitude_zscore,
                                             peak_slope=peak_slope_zscore,
                                             half_peak_slope=half_peak_slope_zscore,
                                             peak_sharpness=peak_sharpness_zscore,
                                             peak_amplitude_threshold=peak_amplitude_threshold,
                                             peak_slope_threshold=peak_slope_threshold,
                                             half_peak_slope_threshold=half_peak_slope_threshold,
                                             peak_sharpness_threshold=peak_sharpness_threshold)
        self.set_picked_peak(picked_peak=picked_peaks)

        # Step5: 输出
        all_samples_of_picked_peak, win_samples_of_picked_peak, samples_of_picked_peak = \
            self.cal_samples_of_picked_peaks(picked_peaks=picked_peaks, peak_amplitude=peak_amplitude,
                                             peak_index=peak_index,
                                             peak_win=int(self.data_info['sfreq'] * 0.05))
        peaks = {'peaks_sample_index': [x[y].cpu().numpy() for x, y in zip(peak_index, picked_peaks)],
                 'peak_duration': [x[y].cpu().long().numpy() for x, y in zip(peak_duration, picked_peaks)],
                 'peak_half_duration': [x[y].cpu().long().numpy() for x, y in zip(peak_half_duration, picked_peaks)],
                 'all_samples_of_picked_peak': all_samples_of_picked_peak.cpu().long().numpy(),
                 'win_samples_of_picked_peak': win_samples_of_picked_peak.cpu().long().numpy(),
                 'samples_of_picked_peak': samples_of_picked_peak.cpu().long().numpy(),
                 'peak_amplitude': [x[y].cpu().numpy() for x, y in zip(peak_amplitude_zscore, picked_peaks)]
                 if peak_amplitude_zscore is not None else None,
                 'peak_slope': [x[y].cpu().numpy() for x, y in zip(peak_slope_zscore, picked_peaks)]
                 if peak_slope_zscore is not None else None,
                 'half_peak_slope': [x[y].cpu().numpy() for x, y in zip(half_peak_slope_zscore, picked_peaks)]
                 if half_peak_slope_zscore is not None else None,
                 'peak_sharpness': [x[y].cpu().numpy() for x, y in zip(peak_sharpness_zscore, picked_peaks)]
                 if peak_sharpness_zscore is not None else None}

        return peaks


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(1)

    raw = mne.io.read_raw_fif('/data2/cuiwei/HFO/xxx/MNE/xxx_tsss.fif', verbose='error', preload=True)
    raw.resample(1000, n_jobs='cuda')
    raw.filter(3, 80, fir_design='firwin', pad="reflect_limited", verbose='error', n_jobs='cuda')
    raw.notch_filter(50, verbose='error', n_jobs='cuda')
    raw.pick(mne.pick_types(raw.info, meg=True, ref_meg=False))
    Info = raw.info
    Raw_data = raw.get_data()

    Peaks = cal_peak_feature(raw_data=Raw_data, data_info=Info, device=1)
    Peaks.get_peaks(smooth_windows=0.16, smooth_iterations=2, z_win=200, z_region=True, z_mag_grad=False,
                    exclude_duration=0.02,
                    peak_amplitude_threshold=[2, None], peak_slope_threshold=None,
                    half_peak_slope_threshold=[2, None], peak_sharpness_threshold=None)
