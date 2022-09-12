# -*- coding:utf-8 -*-
# @Time    : 2022/2/4
# @Author  : cuiwei
# @File    : meg_preprocessing.py
# @Software: PyCharm
# @Script to:
#   - 对meg文件进行预处理

import mne
from mne.utils import logger
import numpy as np
import scipy.io
import os
import re
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from logs import logs
import torch
import torch.fft

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class preprocess:
    def __init__(self, fif_files=None, out_dir=None, bad_channel_dir=None, is_log=True, device=-1, n_jobs=10):
        """
        Description:
            初始化，使用GPU或者CPU。

        Input:
            :param fif_files: list, str
                MEG文件名称
            :param bad_channel_files: PATH, str
                bad channel txt文件存储位置
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
        self.out_dir = out_dir
        self.fif_files = fif_files
        self.set_fif_files(fif_files)
        self.bad_channel_dir = bad_channel_dir
        self.bad_channels = self.cal_bad_channels()
        self.is_log = is_log
        if not self.is_log:
            mne.set_log_file(fname=None)
        else:
            self.log_out_dir = os.path.join(self.out_dir, 'LOG')
            mne.set_log_level(verbose='WARNING')

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

    @staticmethod
    def set_log_file(output_format, fname, overwrite=False):
        logs.set_log_file(fname=fname, output_format=output_format, overwrite=overwrite)
        mne.set_log_file(fname=fname, output_format=output_format, overwrite=overwrite)

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
        sig = torch.abs(torch.fft.ifft(sig * h, axis=-1))
        self.del_var()
        return sig

    def hilbert_h(self, n):
        h = torch.zeros(n)
        if n % 2 == 0:
            h[0] = h[n // 2] = 1
            h[1:n // 2] = 2
        else:
            h[0] = 1
            h[1:(n + 1) // 2] = 2
        return h.to(self.device)

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

    def set_fif_files(self, fif_files=None):
        assert fif_files is not None
        self.fif_files = fif_files

    def get_fif_files(self):
        return self.fif_files

    def set_bad_channels(self, bad_channels=None):
        self.bad_channels = bad_channels

    def get_bad_channels(self):
        return self.bad_channels

    def cal_bad_channels(self):
        """
        Description:
            获取与fif文件对应的bad channels

        Return:
            :return bad_files: list, path or None
                与fif文件对应的bad channels
        """
        if (self.bad_channel_dir is not None) and (os.path.exists(self.bad_channel_dir)):
            # 根据fif文件，获取对应的bad channel名称
            bad_file_names = [x.split('/')[-1].replace('.fif', '.txt') for x in self.fif_files]
            # 获取所有的bad_channel.txt，文件名称为MEGxxxx_EP_x.txt
            file_lists, file_dirs = [], [self.bad_channel_dir].copy()
            while len(file_dirs) > 0:
                file_lists = [os.path.join(x, y) for x in file_dirs for y in os.listdir(x)
                              if len(re.compile(r'MEG(.|\n)*_EP_\d*\.txt').findall(y)) != 0] + file_lists
                file_dirs = [os.path.join(x, y) for x in file_dirs for y in os.listdir(x)
                             if os.path.isdir(os.path.join(x, y))]
            file_name_list = [x.split('/')[-1] for x in file_lists]
            bad_files = [[i for i, y in enumerate(file_name_list) if x == y] for x in bad_file_names]
            bad_files = [file_lists[x[0]] if len(x) > 0 else None for x in bad_files]
            # 读取bad channel
            bad_channels = []
            for x in bad_files:
                if x is None:
                    bad_channels.append(None)
                    continue
                with open(x, 'r') as file:
                    bad_channel = file.read().split(', ')
                if len(bad_channel) == 0:
                    bad_channels.append(None)
                else:
                    bad_channels.append(bad_channel)
        else:
            bad_channels = [None for _ in self.fif_files]

        return bad_channels

    @staticmethod
    def save_ecg_eog_sample(ecg_data=None, eog_data=None, out_dir=None):
        """
        Description:
            保存ECG/EOG片段图片。

        Input:
            :param ecg_data: np.array, double, shape(n)
                ECG信号片段
            :param eog_data: np.array, double, shape(n)
                EOG信号片段
            :param out_dir: str
                保存图片位置
        """
        if out_dir is not None:
            if not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)
            if ecg_data is not None:
                fig = plt.figure(figsize=(8, 8))
                plt.plot(ecg_data)
                plt.savefig(os.path.join(out_dir, 'ECG_Samples.png'))
                plt.close(fig)

            if eog_data is not None:
                fig = plt.figure(figsize=(8, 8))
                plt.plot(eog_data)
                plt.savefig(os.path.join(out_dir, 'EOG_Samples.png'))
                plt.close(fig)

    def cal_ecg_eog_channels(self, raw, check_ecg_and_eog_method=1, out_dir=None):
        """
        Description:
            获取ECG/EOG通道。
            如果ECG/EOG通道都存在，判断两个通道是否相反。
            如果只有一个通道，根据MEN重建ECG/EOG事件输出，判断通道。

        Input:
            :param raw: mne.io.read_raw_fif返回值
                mne.io.read_raw_fif返回值
            :param check_ecg_and_eog_method: number, long, 0 or 1
                判断ECG/EOG两个通道是否相反的算法
            :param out_dir: str
                保存输出数据路径

        Return:
            :return ecg_ref_ch: str
                ECG通道名称
            :return eog_ref_ch: str
                EOG通道名称
        """
        # 获得ECG或者EOG通道
        ecg_ref_ch = [x for x in raw.info.ch_names if 'ECG' in x]
        eog_ref_ch = [x for x in raw.info.ch_names if 'EOG' in x]
        # 如果检测到了ECG和EOG通道
        if len(ecg_ref_ch) > 0 and len(eog_ref_ch) > 0:
            # 确定ECG/EOG是否相反
            eog_data = mne.filter.filter_data(raw.get_data(picks=eog_ref_ch[0]), raw.info['sfreq'],
                                              l_freq=5., h_freq=35., verbose='error',
                                              n_jobs='cuda' if self.is_cuda else self.n_jobs)
            ecg_data = mne.filter.filter_data(raw.get_data(picks=ecg_ref_ch[0]), raw.info['sfreq'],
                                              l_freq=5., h_freq=35., verbose='error',
                                              n_jobs='cuda' if self.is_cuda else self.n_jobs)
            if check_ecg_and_eog_method == 1:
                # 使用过零点数量进行判断ECG/EOG是否相反
                # 使用1%和99%对数据做归一化
                eog_data = eog_data - (np.percentile(eog_data, 99) + np.percentile(eog_data, 1)) / 2
                ecg_data = ecg_data - (np.percentile(ecg_data, 99) + np.percentile(ecg_data, 1)) / 2
                # 两个通道的过零点数量
                zc_eog = len(np.where(np.diff(np.sign(eog_data)))[0])
                zc_ecg = len(np.where(np.diff(np.sign(ecg_data)))[0])
                # eog的过零点数量要高于ecg
                (ecg_ref_ch, eog_ref_ch, eog_data, ecg_data) = \
                    (ecg_ref_ch[0], eog_ref_ch[0], eog_data, ecg_data) if zc_ecg < zc_eog \
                        else (eog_ref_ch[0], ecg_ref_ch[0], ecg_data, eog_data)
            else:
                # 根据ECG/EOG数据分布，确定ECG/EOG是否相反
                # ECG信号为周期窄波峰信号，EOG为宽波峰非周期信号。因此ECG的数据分布要更瘦高
                ecgHist, eogHist = np.histogram(ecg_data, 5000)[0], np.histogram(eog_data, 5000)[0]
                (ecg_ref_ch, eog_ref_ch) = (ecg_ref_ch[0], eog_ref_ch[0]) \
                    if len(np.where(ecgHist > np.percentile(ecgHist, 90))[0]) <= \
                       len(np.where(eogHist > np.percentile(eogHist, 90))[0]) else (eog_ref_ch[0], ecg_ref_ch[0])
            # 确定是否能找到EOG事件
            eog_events = mne.preprocessing.create_eog_epochs(raw, ch_name=eog_ref_ch,
                                                             baseline=(None, 0), verbose='error').events
            if eog_events.shape[0] > 0:
                self.save_ecg_eog_sample(out_dir=out_dir, eog_data=eog_data[0, 4000:6000])
            else:
                eog_ref_ch = None
            # 确定是否能找到ECG事件
            ecg_events, _, _, _ = \
                mne.preprocessing.find_ecg_events(raw, ch_name=ecg_ref_ch, event_id=999, l_freq=8, h_freq=16,
                                                  return_ecg=True, reject_by_annotation=True, verbose='error')
            if ecg_events.shape[0] > 0:
                self.save_ecg_eog_sample(out_dir=out_dir, ecg_data=ecg_data[0, 4000:6000])
            else:
                ecg_ref_ch = None
        elif len(ecg_ref_ch) > 0 or len(eog_ref_ch) > 0:
            # 判断为ecg还是eog通道
            ref_ch = ecg_ref_ch[0] if len(ecg_ref_ch) > 0 else eog_ref_ch[0]
            ecg_events, _, _, _ = \
                mne.preprocessing.find_ecg_events(raw, ch_name=ref_ch, event_id=999, l_freq=8, h_freq=16,
                                                  return_ecg=True, reject_by_annotation=True, verbose='error')
            eog_events = mne.preprocessing.create_eog_epochs(raw, ch_name=ref_ch,
                                                             baseline=(None, 0), verbose='error').events
            if ecg_events.shape[0] > 0:
                ecg_ref_ch = ref_ch
                eog_ref_ch = None
                self.save_ecg_eog_sample(raw=raw, out_dir=out_dir,
                                         ecg_data=raw.get_data(picks=ecg_ref_ch)[0, 4000:6000])
            elif eog_events.shape[0] > 0:
                eog_ref_ch = ref_ch
                ecg_ref_ch = None
                self.save_ecg_eog_sample(raw=raw, out_dir=out_dir,
                                         eog_data=raw.get_data(picks=eog_ref_ch)[0, 4000:6000])
            else:
                eog_ref_ch = None
                ecg_ref_ch = None
        else:
            eog_ref_ch = None
            ecg_ref_ch = None
        return ecg_ref_ch, eog_ref_ch

    @staticmethod
    def save_ecg_eog_components(raw=None, ica=None, ecg_scores=None, eog_scores=None, out_dir=None):
        """
        Description:
            保存ECG/EOG片段图片。

        Input:
            :param raw: raw结构体
                raw
            :param ica: ica结构体
                ica.fit后结果
            :param ecg_scores: np.array, double, shape(n_components)
                每个IC和ECG之间的相关性评分
            :param eog_scores: np.array, double, shape(n_components)
                每个IC和EOG之间的相关性评分
            :param out_dir: str
                保存图片位置
        """
        if (out_dir is not None) and (ica is not None):
            if not os.path.exists(out_dir):
                os.makedirs(out_dir, exist_ok=True)
            if ecg_scores is not None:
                fig = ica.plot_scores(ecg_scores, show=False).savefig(os.path.join(out_dir, 'ECG_Components.png'))
            if eog_scores is not None:
                fig = ica.plot_scores(eog_scores, show=False).savefig(os.path.join(out_dir, 'EOG_Components.png'))
            plt.close(fig)
            # 保存ICA曲线
            fig = ica.plot_sources(raw, show=False).savefig(os.path.join(out_dir, 'ICA_Sources.png'))
            plt.close(fig)

    def cal_ecg_eog_artifact_removal(self, raw, out_dir=None):
        """
        Description:
            使用ICA的方法去除ECG/EOG噪声

        Input:
            :param raw: mne.io.read_raw_fif返回值
                mne.io.read_raw_fif返回值
            :param out_dir: path
                保存输出数据路径
        Return:
            :return raw: mne.io.read_raw_fif返回值
                处理后的raw
        """
        save_dir = os.path.join(out_dir, 'ECG_EOG_FIG', os.path.split(raw.filenames[0])[-1].split('.')[-2])
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        # 对数据进行滤波
        raw.filter(l_freq=1., h_freq=None, verbose='error', n_jobs='cuda' if self.is_cuda else self.n_jobs)
        # 提取MEG信号
        picks_meg = mne.pick_types(raw.info, meg=True, exclude='bads', stim=False)
        # 设置ICA参数
        ica = mne.preprocessing.ICA(n_components=36, method='fastica', random_state=97, max_iter='auto',
                                    verbose='error')
        # 计算MEG数据的ICs
        try:
            # 去除很大的背景噪声
            ica.fit(raw, picks=picks_meg, reject=dict(mag=5e-12, grad=8000e-13), decim=3, verbose='error')
        except:
            ica.fit(raw, picks=picks_meg, decim=3, verbose='error')
        # 获得ECG/EOG通道名称
        ecg_ref_ch, eog_ref_ch = self.cal_ecg_eog_channels(raw, check_ecg_and_eog_method=1, out_dir=save_dir)
        # 去除ECG成分
        if ecg_ref_ch is not None:
            n_max_ecg = 3
            # 从ECG数据中获得ECG片段
            ecg_epochs = mne.preprocessing.create_ecg_epochs(raw, ch_name=ecg_ref_ch, baseline=(None, 0),
                                                             verbose='error')
            # 获得与ECG数据相关的IC
            ecg_inds, scores = ica.find_bads_ecg(ecg_epochs, threshold='auto', verbose='error')
            ica.exclude += ecg_inds[:n_max_ecg]
            # 保存ICs和ECG的相关性
            self.save_ecg_eog_components(ica=ica, ecg_scores=scores, out_dir=save_dir)
        # 去除EOG成分
        if eog_ref_ch is not None:
            n_max_eog = 3
            # 从EOG数据中获得EOG片段
            eog_epochs = mne.preprocessing.create_eog_epochs(raw, ch_name=eog_ref_ch, baseline=(None, 0),
                                                             verbose='error')
            # 获得与EOG数据相关的IC
            eog_inds, scores = ica.find_bads_eog(eog_epochs, verbose='error')
            ica.exclude += eog_inds[:n_max_eog]
            # 保存ICs和EOG的相关性
            self.save_ecg_eog_components(ica=ica, eog_scores=scores, out_dir=save_dir)

        # 将不包含ECG和EOG的ICs反变换回raw
        raw = ica.apply(raw, exclude=ica.exclude)

        return raw

    def cal_maxwell_filter(self, raw, tsss_lowpass=200., thre=7., fine_cal_file=None, crosstalk_file=None, out_dir=None,
                           manual_bad=None):
        """
        Description:
            计算tsss，去除MEG头盔内和头盔外的噪声
            (1). 自动检测bad channel，并插值出bad channel数据
            (2). 计算tsss

        Input:
            :param raw: mne.io.read_raw_fif返回值
                mne.io.read_raw_fif返回值
            :param tsss_lowpass: number, double
                自动检测bad channel时的数据低通滤波
            :param thre: number, double
                自动检测bad channel时的阈值
            :param fine_cal_file: str
                计算maxwell_filter时用到的fine_cal_file
            :param crosstalk_file: str
                计算maxwell_filter时用到的crosstalk_file
            :param out_dir: path
                log文件输出路径
            :param manual_bad: list, str
                人工标记的bad channel名称
        Return:
            :return raw: mne.io.read_raw_fif返回值
                tsss处理后的raw
        """

        # 对数据进行陷波
        raw_check = raw.copy().pick_types(meg=True, exclude='bads', stim=False)
        raw_check = raw_check.notch_filter(np.arange(50, tsss_lowpass, 50), verbose='error',
                                           n_jobs='cuda' if self.is_cuda else self.n_jobs)
        # 自动检测bad channel
        auto_noisy_chs, auto_flat_chs = \
            mne.preprocessing.find_bad_channels_maxwell(raw=raw_check, limit=thre, h_freq=tsss_lowpass, verbose='error',
                                                        cross_talk=crosstalk_file, calibration=fine_cal_file)
        auto_bad = auto_noisy_chs + auto_flat_chs
        raw.info['bads'].extend(auto_bad)
        # 加入自动标记的bad channel
        if manual_bad is not None:
            raw.info['bads'].extend([x for x in manual_bad if (x not in raw.info['bads']) and (x in raw.ch_names)])
        else:
            manual_bad = []
        # 计算maxwell_filter
        raw = mne.preprocessing.maxwell_filter(raw=raw, cross_talk=crosstalk_file, calibration=fine_cal_file,
                                               verbose='error')
        # 保存bad channel名称
        if out_dir is not None:
            if not os.path.exists(os.path.join(out_dir, 'BadChannel')):
                os.makedirs(os.path.join(out_dir, 'BadChannel'), exist_ok=True)
            both_auto_manual = [x for x in manual_bad if x in auto_bad]
            only_auto = [x for x in auto_bad if x not in manual_bad]
            only_manual = [x for x in manual_bad if x not in auto_bad]
            with open(os.path.join(out_dir, 'BadChannel', os.path.split(raw.filenames[0])[-1].split('.')[-2] + '.txt'),
                      'w') as file:
                file.write('Both detected: ' + ', '.join(both_auto_manual))
                file.write('\nOnly auto detected: ' + ', '.join(only_auto))
                file.write('\nOnly manual detected: ' + ', '.join(only_manual))
        return raw

    def cal_resample(self, raw=None, resample_freq=1000):
        """
        Description:
            对数据进行重采样

        Input:
            :param raw: mne.io.read_raw_fif返回值
                mne.io.read_raw_fif返回值
            :param resample_freq: number, long
                重采样后的频率
        Return:
            :return raw: mne.io.read_raw_fif返回值
                处理后的raw
        """
        raw = raw.resample(resample_freq, n_jobs='cuda' if self.is_cuda else self.n_jobs)
        return raw

    def cal_jump_segments(self, raw, jump_win=0.2, jump_threshold=10, min_length_good=0.2, use_percentile=True):
        """
        Description:
            计算数据中有跳变(jump)引起的坏片段。
            考虑到jump为短时间巨大数据跳变，我们使用短时间窗峰峰值作为特征，进行阈值判断。

        Input:
            :param raw: mne.io.read_raw_fif返回值
                mne.io.read_raw_fif返回值
            :param jump_win: number, double
                jump坏片段的时间窗长度，单位秒
            :param jump_threshold: number, double
                jump坏片段的阈值
            :param min_length_good: number, double
                相邻jump坏片段的最小间隔，单位秒
            :param use_percentile: number, bool
                True: 使用80%位点值进行normalization; False: 使用z-score
        Return:
            :return jump_segments: np.array, bool, shape(n_sample)
                坏数据位置
        """
        # 获取原始数据并进行高通滤波
        data = raw.get_data().copy()[mne.pick_types(raw.info, meg=True)]
        data = mne.filter.filter_data(data, raw.info['sfreq'], 1, None, fir_design='firwin', pad="reflect_limited",
                                      n_jobs='cuda' if self.is_cuda else self.n_jobs, verbose='error')
        data = mne.filter.notch_filter(data, raw.info['sfreq'], np.arange(50, raw.info['sfreq'] / 3, 50),
                                       n_jobs='cuda' if self.is_cuda else self.n_jobs, verbose='error')
        data = torch.tensor(data).to(self.device)
        jump_segments = torch.ones(data.shape[-1]) < 0
        # 对于每个采样点，获取其周围5ms数据
        half_seg_win = int(raw.info['sfreq'] * 0.005 / 2)
        data_temp = torch.cat([torch.zeros(data.shape[0], half_seg_win).to(self.device),
                               data, torch.zeros(data.shape[0], half_seg_win).to(self.device)], dim=-1)
        half_seg_win = torch.arange(-half_seg_win, half_seg_win+1).reshape(-1, 1) + torch.arange(data.shape[1])
        half_seg_win = half_seg_win - half_seg_win.min()
        seg_data = data_temp.unsqueeze(-1).take_along_dim(half_seg_win.T.unsqueeze(0).to(self.device), dim=1)
        # 获取每个segment的峰峰值
        seg_data_peak2peak = (seg_data.amax(dim=-1) - seg_data.amin(dim=-1)).abs()
        if use_percentile:
            # 对于每个时间点，获得峰峰值前五个通道均值,　并使用80位点作为normalization
            seg_data_peak2peak = seg_data_peak2peak.topk(5, dim=0)[0].mean(dim=0)
            seg_data_peak2peak = seg_data_peak2peak / np.percentile(seg_data_peak2peak.cpu().numpy(), 80)
        else:
            # 计算每个通道z-score值(相对baseline的值)
            # 对于每个时间点，获得峰峰值前五个通道均值。
            seg_data_peak2peak = (seg_data_peak2peak - seg_data_peak2peak.mean(dim=1, keepdim=True)) / \
                                 seg_data_peak2peak.std(dim=1, keepdim=True)
            seg_data_peak2peak = seg_data_peak2peak.topk(5, dim=0)[0].mean(dim=0)
        # 将峰峰值大于阈值的时间点作为jump坏片段
        jump_seg = torch.where(seg_data_peak2peak > jump_threshold)[0].cpu()
        if len(jump_seg) > 0:
            # 将坏时间点左右扩展jump_win长度
            jump_win = int(jump_win * raw.info['sfreq'])
            jump_seg = (jump_seg.reshape(-1) + torch.arange(-jump_win, jump_win).unsqueeze(-1)).unique().long()
            # 将相距小于min_length_good的坏片段之间的采样点，也认为是坏片段
            min_length_good = min_length_good * raw.info['sfreq']
            jump_seg = torch.cat([torch.tensor(range(jump_seg[x], jump_seg[x + 1]))
                                  for x in torch.where((jump_seg[1:] - jump_seg[:-1] <= min_length_good) &
                                                       (jump_seg[1:] - jump_seg[:-1] > 1))[0]] + [jump_seg])
        jump_segments[jump_seg[jump_seg < jump_segments.shape[0]]] = True
        self.del_var()
        return jump_segments.cpu().numpy()

    def cal_noisy_segments(self, raw, freq_range=(40, 240), noisy_min_win=0.2, noisy_threshold=5, min_length_good=0.2):
        """
        Description:
            计算数据中noisy的坏片段。

        Input:
            :param raw: mne.io.read_raw_fif返回值
                mne.io.read_raw_fif返回值
            :param freq_range: list/tuple, double, shape(2)
                noisy的频率范围：肌电40-240Hz
            :param noisy_min_win: number, double
                坏片段的最小持续时间长度，单位秒
            :param noisy_threshold: number, double
                坏片段的阈值
            :param min_length_good: number, double
                相邻jump坏片段的最小间隔，单位秒
        Return:
            :return noisy_segments: np.array, bool, shape(n_sample)
                坏数据位置
        """

        # 获得数据并进行滤波
        data = raw.get_data().copy()
        data = mne.filter.filter_data(data, raw.info['sfreq'], freq_range[0], freq_range[1], fir_design='firwin',
                                      pad="reflect_limited", verbose='error',
                                      n_jobs='cuda' if self.is_cuda else self.n_jobs)
        if freq_range[1] >= 50:
            data = mne.filter.notch_filter(data, raw.info['sfreq'], np.arange(50, freq_range[1], 50),
                                           n_jobs='cuda' if self.is_cuda else self.n_jobs, verbose='error')
        data = torch.tensor(data).to(self.device).float()
        self.del_var()
        # 对数据进行hilbert变换，求取包络，并在通道维度叠加
        data_hilbert = self.hilbert(data)
        data_mag = data_hilbert[torch.tensor(mne.pick_types(raw.info, meg='mag')).to(self.device)]
        data_grad = data_hilbert[torch.tensor(mne.pick_types(raw.info, meg='grad')).to(self.device)]
        data_mag = (data_mag - data_mag.mean(dim=1, keepdim=True)) / data_mag.std(dim=1, keepdim=True)
        data_grad = (data_grad - data_grad.mean(dim=1, keepdim=True)) / data_grad.std(dim=1, keepdim=True)
        # 计算artifact参数
        art_scores_mag = data_mag.sum(axis=0) / np.sqrt(data_mag.shape[0])
        art_scores_mag = torch.tensor(
            mne.filter.filter_data(art_scores_mag.cpu().double(), raw.info['sfreq'], None, 4, verbose=False,
                                   n_jobs='cuda' if self.is_cuda else self.n_jobs)).to(self.device)
        art_scores_grad = data_grad.sum(axis=0) / np.sqrt(data_grad.shape[0])
        art_scores_grad = torch.tensor(
            mne.filter.filter_data(art_scores_grad.cpu().double(), raw.info['sfreq'], None, 4, verbose=False,
                                   n_jobs='cuda' if self.is_cuda else self.n_jobs)).to(self.device)
        # 获取大于阈值的片段
        min_length_good = min_length_good * raw.info['sfreq']
        noisy_segments = (art_scores_mag > noisy_threshold) | (art_scores_grad > noisy_threshold)
        # 将相距小于min_length_good的坏片段之间的采样点，也认为是坏片段
        temp = torch.where(noisy_segments)[0]
        temp = [torch.tensor(range(temp[x], temp[x + 1]))
                for x in torch.where((temp[1:] - temp[:-1] <= min_length_good) & (temp[1:] - temp[:-1] > 1))[0]]
        if len(temp) > 0:
            noisy_segments[torch.cat(temp)] = True
        # 将持续时间小于noisy_min_win的片段删除
        temp = torch.tensor([0.] + noisy_segments.float().tolist() + [0.]).diff()
        temp = torch.stack([torch.where(temp == 1)[0], torch.where(temp == -1)[0]])
        temp = [torch.arange(temp[0, x], temp[1, x])
                for x in torch.where(temp.diff(dim=0) <= noisy_min_win * raw.info['sfreq'])[1]]
        if len(temp) > 0:
            noisy_segments[torch.cat(temp)] = False
        self.del_var()
        return noisy_segments.cpu().numpy()

    def cal_filter(self, raw, freq_range=(40, 240), notch=None):
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
                          verbose='error', n_jobs='cuda' if self.is_cuda else self.n_jobs)
        if (notch is not None) and (freq_range[1] is not None) and (freq_range[1] >= notch):
            raw_filter.notch_filter(np.arange(notch, freq_range[1], notch), verbose='error',
                                    n_jobs='cuda' if self.is_cuda else self.n_jobs)

        return raw_filter

    @staticmethod
    def save_fif(raw, out_dir=None):
        """
        Description:
            存储fif文件

        Input:
            :param raw: mne.io.read_raw_fif返回值
                mne.io.read_raw_fif返回值
            :param out_dir: path
                fif文件存储位置
        """
        raw.save(os.path.join(out_dir,
                              os.path.split(raw.filenames[0])[-1].replace('_tsss', '').split('.')[-2] + '_tsss.fif'),
                 overwrite=True)

    def run_process(self, fif_file, out_dir=None,
                    do_tsss=False, tsss_auto_bad_lowpass=200., tsss_auto_bad_thre=7.,
                    tsss_manual_bad=None, fine_cal_file=None, crosstalk_file=None,
                    do_ica=False,
                    do_resample=False, resample_freq=1000,
                    do_jump_detection=False, jump_win=0.2, jump_threshold=30, min_length_good=0.2,
                    do_noisy_detection=False, noisy_freq_range=(50, 240), noisy_threshold=7.5, noisy_min_win=0.2,
                    do_filter=False, filter_band=('IED', 'HFO')):
        """
        Description:
            对数据进行处理

        Input:
            :param fif_file: path
                原始数据位置
            :param out_dir: path
                输出数据路径，不需要包含fif_file名称
            ----------------------------------TSSS处理----------------------------------
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
            :param tsss_manual_bad: list, str
                人工标记的bad channel名称
            ----------------------------------ICA处理-----------------------------------
            :param do_ica: bool
                是否做ICA
            ----------------------------------重采样处理----------------------------------
            :param do_resample: bool
                是否做重采样
            :param resample_freq: number, long
                重采样后的频率
            ---------------------------------jump坏段检测---------------------------------
            :param do_jump_detection: bool
                是否做jump坏段检测
            :param jump_win: number, double
                jump坏片段的时间窗长度，单位秒
            :param jump_threshold: number, double
                jump坏片段的阈值
            :param min_length_good: number, double
                相邻jump坏片段的最小间隔，单位秒
            ---------------------------------noisy坏段检测--------------------------------
            :param do_noisy_detection: bool
                是否做noisy坏段检测
            :param noisy_freq_range: list/tuple, double, shape(2)
                noisy的频率范围：肌电40-240Hz
            :param noisy_min_win: number, double
                坏片段的最小持续时间长度，单位秒
            :param noisy_threshold: number, double
                坏片段的阈值
            -----------------------------------滤波处理----------------------------------
            :param do_filter: bool
                是否做滤波处理
            :param filter_band: tuple/list
                滤波范围：'HFO'->80-200Hz; 'IED'->3-80Hz
        """
        fif_name = os.path.split(fif_file)[-1].replace('_tsss', '')
        try:
            # 新建文件夹
            if out_dir is not None:
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir, exist_ok=True)
            else:
                return
            # 读取原始数据
            raw = mne.io.read_raw_fif(fif_file, verbose='error', preload=True)
            # 新建log文件
            if self.is_log:
                subj_name = raw.info['subject_info']
                subj_name = subj_name['last_name'] + subj_name['first_name']
                log_out_dir_temp = os.path.join(self.log_out_dir, subj_name)
                if not os.path.exists(log_out_dir_temp):
                    os.makedirs(log_out_dir_temp, exist_ok=True)
                log_out_dir = os.path.join(self.log_out_dir, subj_name, fif_name.replace('.fif', '_logs.txt'))
            # 进行tsss处理
            if do_tsss:
                if self.is_log:
                    self.set_log_file(fname=log_out_dir,
                                      output_format='[%(levelname)s][TSSS][' + fif_name.replace('.fif', '') +
                                                    ']: <line:%(lineno)d> %(message)s', overwrite=False)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir, exist_ok=True)
                raw = self.cal_maxwell_filter(raw, tsss_lowpass=tsss_auto_bad_lowpass, thre=tsss_auto_bad_thre,
                                              fine_cal_file=fine_cal_file, crosstalk_file=crosstalk_file,
                                              manual_bad=tsss_manual_bad, out_dir=out_dir)
                self.save_fif(raw, out_dir)
            # 进行ICA处理
            if do_ica:
                if self.is_log:
                    self.set_log_file(fname=log_out_dir,
                                      output_format='[%(levelname)s][ICA][' + fif_name.replace('.fif', '') +
                                                    ']: <line:%(lineno)d> %(message)s', overwrite=False)
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir, exist_ok=True)
                raw = self.cal_ecg_eog_artifact_removal(raw, out_dir=out_dir)
            # 进行降采样
            if do_resample and resample_freq is not None and raw.info['sfreq'] != resample_freq:
                do_resample = True
                raw = self.cal_resample(raw, resample_freq=resample_freq)
            else:
                do_resample = False
            # 保存数据
            if (do_resample and resample_freq is not None) or do_ica:
                self.save_fif(raw, out_dir)
            # 进行jump segment detection处理
            if do_jump_detection:
                if self.is_log:
                    self.set_log_file(fname=log_out_dir,
                                      output_format='[%(levelname)s][Jump][' + fif_name.replace('.fif', '') +
                                                    ']: <line:%(lineno)d> %(message)s', overwrite=False)
                out_dir_segment = os.path.join(out_dir, 'MEG_BAD')
                if not os.path.exists(out_dir_segment):
                    os.makedirs(out_dir_segment, exist_ok=True)
                jump_segment = self.cal_jump_segments(raw, jump_win=jump_win, jump_threshold=jump_threshold,
                                                      min_length_good=min_length_good)
                np.save(os.path.join(out_dir_segment, fif_name[:-4] +
                                     '_jump_threshold_' + str(jump_threshold) + '.npy'), jump_segment)
            # 进行noisy segment detection处理
            if do_noisy_detection:
                if self.is_log:
                    self.set_log_file(fname=log_out_dir,
                                      output_format='[%(levelname)s][Noisy][' + fif_name.replace('.fif', '') +
                                                    ']: <line:%(lineno)d> %(message)s', overwrite=False)
                out_dir_segment = os.path.join(out_dir, 'MEG_BAD')
                if not os.path.exists(out_dir_segment):
                    os.makedirs(out_dir_segment, exist_ok=True)
                noisy_segments = self.cal_noisy_segments(raw, freq_range=noisy_freq_range, noisy_min_win=noisy_min_win,
                                                         noisy_threshold=noisy_threshold,
                                                         min_length_good=min_length_good)
                np.save(os.path.join(out_dir_segment, fif_name[:-4] +
                                     '_noisy_threshold_' + str(noisy_threshold) + '.npy'), noisy_segments)
            # 进行滤波处理
            if do_filter:
                if self.is_log:
                    self.set_log_file(fname=log_out_dir,
                                      output_format='[%(levelname)s][Filter][' + fif_name.replace('.fif', '') +
                                                    ']: <line:%(lineno)d> %(message)s', overwrite=False)
                if 'IED' in filter_band:
                    out_dir_ied = os.path.join(out_dir, 'MEG_IED')
                    if not os.path.exists(out_dir_ied):
                        os.makedirs(out_dir_ied, exist_ok=True)
                    raw_ied_filter = self.cal_filter(raw, freq_range=(3, 80), notch=50)
                    self.save_fif(raw_ied_filter, out_dir_ied)
                if 'HFO' in filter_band:
                    out_dir_hfo = os.path.join(out_dir, 'MEG_HFO')
                    if not os.path.exists(out_dir_hfo):
                        os.makedirs(out_dir_hfo, exist_ok=True)
                    raw_hfo_filter = self.cal_filter(raw, freq_range=(80, 200), notch=None)
                    self.save_fif(raw_hfo_filter, out_dir_hfo)
        except Exception as e:
            print('Preprocessing error: ', fif_name.split('.')[-2])
            if self.is_log:
                self.set_log_file(fname=log_out_dir,
                                  output_format='[%(levelname)s][ALL][' + fif_name.replace('.fif', '') +
                                                ']: <line:%(lineno)d> %(message)s', overwrite=False)
                logger.error(e)

    def run_multi_fifs_parallel(self, do_tsss=False, tsss_auto_bad_lowpass=200., tsss_auto_bad_thre=7.,
                                fine_cal_file=None, crosstalk_file=None,
                                do_ica=False,
                                do_resample=False, resample_freq=1000,
                                do_jump_detection=False, jump_win=0.2, jump_threshold=30, min_length_good=0.2,
                                do_noisy_detection=False, noisy_freq_range=(40, 240), noisy_threshold=30,
                                noisy_min_win=0.2,
                                do_filter=False, filter_band=('IED', 'HFO'),
                                workers=5):

        self._check_cuda(device=-1)
        with ProcessPoolExecutor(max_workers=min(workers, len(self.get_fif_files()))) as executor:
            _ = \
                [executor.submit(self.run_process, fif, out_dir=self.out_dir,
                                 do_tsss=do_tsss, tsss_auto_bad_lowpass=tsss_auto_bad_lowpass,
                                 tsss_auto_bad_thre=tsss_auto_bad_thre, tsss_manual_bad=bad,
                                 fine_cal_file=fine_cal_file, crosstalk_file=crosstalk_file,
                                 do_ica=do_ica,
                                 do_resample=do_resample, resample_freq=resample_freq,
                                 do_jump_detection=do_jump_detection, jump_win=jump_win, jump_threshold=jump_threshold,
                                 min_length_good=min_length_good,
                                 do_noisy_detection=do_noisy_detection, noisy_freq_range=noisy_freq_range,
                                 noisy_threshold=noisy_threshold, noisy_min_win=noisy_min_win,
                                 do_filter=do_filter, filter_band=filter_band)
                 for fif, bad in zip(self.get_fif_files(), self.get_bad_channels())]

    def run_multi_fifs(self, do_tsss=False, tsss_auto_bad_lowpass=200., tsss_auto_bad_thre=7.,
                       fine_cal_file=None, crosstalk_file=None,
                       do_ica=False,
                       do_resample=False, resample_freq=1000,
                       do_jump_detection=False, jump_win=0.2, jump_threshold=30, min_length_good=0.2,
                       do_noisy_detection=False, noisy_freq_range=(40, 240), noisy_threshold=30, noisy_min_win=0.2,
                       do_filter=False, filter_band=('IED', 'HFO')):

        for fif, bad in zip(self.get_fif_files(), self.get_bad_channels()):
            self.run_process(fif, out_dir=self.out_dir,
                             do_tsss=do_tsss, tsss_auto_bad_lowpass=tsss_auto_bad_lowpass,
                             tsss_auto_bad_thre=tsss_auto_bad_thre, tsss_manual_bad=bad,
                             fine_cal_file=fine_cal_file, crosstalk_file=crosstalk_file,
                             do_ica=do_ica,
                             do_resample=do_resample, resample_freq=resample_freq,
                             do_jump_detection=do_jump_detection, jump_win=jump_win, jump_threshold=jump_threshold,
                             min_length_good=min_length_good,
                             do_noisy_detection=do_noisy_detection, noisy_freq_range=noisy_freq_range,
                             noisy_threshold=noisy_threshold, noisy_min_win=noisy_min_win,
                             do_filter=do_filter, filter_band=filter_band)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

    Fif_files = ['/sanbo_dataset/4k_Project/Data_Full/MEG_RAW/MEG2402/MEG2402_EP_9.fif']
    BadChannel_Dir = '/sanbo_dataset/4k_Project/Data_Full/MEG_BadChannel/txt_file/IED'

    MEG = preprocess(fif_files=Fif_files, out_dir='/data2/cuiwei/HFO_IED_Ictal/TEST', bad_channel_dir=BadChannel_Dir,
                        device=0, n_jobs=10)
    MEG.run_multi_fifs(do_tsss=True, do_ica=True, do_resample=False,
                       do_jump_detection=True, jump_threshold=30,
                       do_noisy_detection=True, noisy_freq_range=(40, 240), noisy_threshold=10,
                       do_filter=True, filter_band=('IEDs', 'Ripples'))
