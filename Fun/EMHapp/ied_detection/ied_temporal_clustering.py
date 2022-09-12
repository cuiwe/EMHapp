# -*- coding:utf-8 -*-
# @Time    : 2022/2/4
# @Author  : cuiwei
# @File    : ied_temporal_clustering.py
# @Software: PyCharm
# @Script to:
#   - 使用ied片段，聚类出ied templates, 并使用template matching减少假阳性

import os
import mne
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt


class temporal_cluster:
    def __init__(self, device=-1, n_jobs=10):
        """
        Description:
            初始化，使用GPU或者CPU。

        Input:
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

    def similarity_between_segments_templates(self, data, templates, chunk=5000):
        """
        Description:
            计算输入data之间的相似度矩阵。相似度矩阵=欧氏距离矩阵+Pearson相关性矩阵。计算过程：
                (1). 数据的shift通过data的第是三维度引入，即data第三维代表每个shift产生的数据。
                     如果对应位置没有shift，则赋值为0. 0的相似性输出为nan
                (2). 计算相似性最大的shift

        Input:
            :param data: torch.tensor, double, shape(n*n_channel*n_shift*n_time)
                需要计算时间相似度的数据，有n个数据，每个数据有n_shift个平移，数据有n_time个采样点
            :param templates: torch.tensor, double, shape(n_templates*n_time)
                IED template
            :param chunk: number, long
                矩阵分成chunk计算，减少gpu显存使用。

        Return:
            :return sim_raw_format: torch.tensor, float, shape(n*n_channel*n_templates)
                相似度矩阵
            :return corr_raw_format: torch.tensor, float, shape(n*n_channel*n_templates)
                Pearson相关性矩阵
            :return dist_raw_format: torch.tensor, float, shape(n*n_channel*n_templates)
                欧氏距离矩阵
            :return sim_max_shift: torch.tensor, float, shape(n*n_channel*n_templates)
                shift矩阵
        """
        # 提取非零segment，减少计算
        none_zeros_index = torch.where(data.sum(dim=-1) != 0)
        data_x = data[none_zeros_index].clone().float().to(self.device)
        data_y = templates.float().clone().to(self.device)
        # 使用chunk方法计算相似度矩阵
        corr, dist, sim = self.cal_similarity_matrix_with_pearson_euler_chunk(data_x, data_y, chunk=chunk)
        # 转换为和输入data相同维度
        sim_raw_format = torch.zeros(data.shape[0], data.shape[1], data.shape[2], templates.shape[0])
        corr_raw_format, dist_raw_format = sim_raw_format.clone(), sim_raw_format.clone()
        sim_raw_format[none_zeros_index] = sim.float()
        corr_raw_format[none_zeros_index] = corr.float()
        dist_raw_format[none_zeros_index] = dist.float()
        # 在shift维度提取最大的sim
        sim_max_shift = sim_raw_format.argmax(dim=2).unsqueeze(-2)
        sim_raw_format = sim_raw_format.take_along_dim(sim_max_shift, dim=-2)[:, :, 0]
        corr_raw_format = corr_raw_format.take_along_dim(sim_max_shift, dim=-2)[:, :, 0]
        dist_raw_format = dist_raw_format.take_along_dim(sim_max_shift, dim=-2)[:, :, 0]

        self.del_var()
        return sim_raw_format, corr_raw_format, dist_raw_format, sim_max_shift

    def cal_similarity_matrix_with_pearson_euler_chunk(self, data_x, data_y, chunk=1000):
        """
        Description:
            将输出的相关矩阵(data_x.shape(0)*data_y.shape(0))划分成若干个chunk(chunk*chunk)，减少GPU显存的使用
            计算输入data_x和data_y之间的相似度矩阵。相似度矩阵=欧氏距离矩阵+Pearson相关性矩阵

        Input:
            :param data_x: torch.tensor, double, shape(n*n_time)
                需要计算时间相似度的数据，有n个数据，数据有n_time个采样点
            :param data_y: torch.tensor, double, shape(m*n_time)
                需要计算时间相似度的数据，有m个数据，数据有n_time个采样点
            :param chunk: number, long
                矩阵分成chunk计算，减少gpu显存使用。

        Return:
            :return sim: torch.tensor, double, shape(n*m)
                相似度矩阵
            :return corr: torch.tensor, double, shape(n*m)
                Pearson相关性矩阵
            :return dist: torch.tensor, double, shape(n*m)
                欧氏距离矩阵
        """

        # 保证data_x.shape[0] >= data_y.shape[1]
        if data_x.shape[0] < data_y.shape[1]:
            is_transpose = True
            temp = data_x.clone()
            data_x, data_y = data_y.clone(), temp.clone()
            del temp
            self.del_var()
        else:
            is_transpose = False
        with torch.no_grad():
            # 将矩阵分成chunk计算，减少gpu显存使用
            corr = torch.zeros(data_x.shape[0], data_y.shape[0]).float()
            dist, sim = corr.clone(), corr.clone()
            # 将相关矩阵划分成若干个chunk*chunk，并计算每个chunk在原始相关矩阵中的index
            chunk_index = [[i * chunk, min(data_x.shape[0], (i + 1) * chunk)]
                           for i in range(int(np.ceil(data_x.shape[0] / chunk)))]
            chunk_index_pair = torch.stack([torch.arange(len(chunk_index)).repeat_interleave(len(chunk_index)),
                                            torch.arange(len(chunk_index)).repeat(len(chunk_index))], dim=-1)
            for index in chunk_index_pair:
                # 对于corr_temp, dist_temp, sim_temp进行切片赋值
                corr[
                chunk_index[index[0]][0]:chunk_index[index[0]][1], chunk_index[index[1]][0]:chunk_index[index[1]][1]], \
                dist[
                chunk_index[index[0]][0]:chunk_index[index[0]][1], chunk_index[index[1]][0]:chunk_index[index[1]][1]], \
                sim[
                chunk_index[index[0]][0]:chunk_index[index[0]][1], chunk_index[index[1]][0]:chunk_index[index[1]][1]] \
                    = self.cal_similarity_matrix_with_pearson_euler(
                    data_x[chunk_index[index[0]][0]:chunk_index[index[0]][1]],
                    data_y[chunk_index[index[1]][0]:chunk_index[index[1]][1]])
            if is_transpose:
                corr, dist, sim = corr.t(), dist.t(), sim.t()

        return corr, dist, sim

    def cal_similarity_matrix_with_pearson_euler(self, data_x, data_y):
        """
        Description:
            计算输入data_x和data_y之间的相似度矩阵。相似度矩阵=欧氏距离矩阵+Pearson相关性矩阵

        Input:
            :param data_x: torch.tensor, double, shape(n*n_time)
                需要计算时间相似度的数据，有n个数据，数据有n_time个采样点
            :param data_y: torch.tensor, double, shape(m*n_time)
                需要计算时间相似度的数据，有m个数据，数据有n_time个采样点

        Return:
            :return sim: torch.tensor, double, shape(n*m)
                相似度矩阵
            :return corr: torch.tensor, double, shape(n*m)
                Pearson相关性矩阵
            :return dist: torch.tensor, double, shape(n*m)
                欧氏距离矩阵
        """
        with torch.no_grad():
            # 归一化数据到[-0.5, 0.5]
            data_x_temp = (data_x - data_x.amin(dim=-1, keepdim=True)) / \
                          (data_x.amax(dim=-1, keepdim=True) - data_x.amin(dim=-1, keepdim=True)) - 0.5
            data_y_temp = (data_y - data_y.amin(dim=-1, keepdim=True)) / \
                          (data_y.amax(dim=-1, keepdim=True) - data_y.amin(dim=-1, keepdim=True)) - 0.5
            data_x_temp, data_y_temp = data_x_temp.float(), data_y_temp.float()
            data_x_temp = data_x_temp.to(self.device).repeat_interleave(data_y.shape[0], dim=0)
            data_y_temp = data_y_temp.to(self.device).repeat(data_x.shape[0], 1)
            # 计算Pearson相关系数
            centered_x = data_x_temp - data_x_temp.mean(dim=1, keepdim=True)
            centered_y = data_y_temp - data_y_temp.mean(dim=1, keepdim=True)
            covariance_x = (centered_x * centered_x).sum(dim=1, keepdim=True)
            covariance_y = (centered_y * centered_y).sum(dim=1, keepdim=True)
            covariance = (centered_x * centered_y).sum(dim=1, keepdim=True)
            corr_temp = (covariance / (covariance_x * covariance_y).sqrt()).reshape(-1)
            # 计算欧氏距离，如果对应相关系数为负数，则先取反再计算欧氏距离
            # 并将欧氏距离re-scale到[0, 1]
            dist_temp = (data_x_temp - corr_temp.reshape(-1, 1).sign() * data_y_temp).pow(2).sum(dim=1).sqrt()
            dist_temp = 1 - (dist_temp / np.sqrt(data_x.shape[-1]))
            # 使用欧氏距离加相关系数，作为相似性
            sim_temp = (corr_temp.abs() + dist_temp) / 2
            # 转换为矩阵格式
            corr = corr_temp.nan_to_num_(0).reshape(data_x.shape[0], data_y.shape[0]).float()
            dist = dist_temp.nan_to_num_(0).reshape(data_x.shape[0], data_y.shape[0]).float()
            sim = sim_temp.nan_to_num_(0).reshape(data_x.shape[0], data_y.shape[0]).float()

            del corr_temp, dist_temp, sim_temp, data_x_temp, data_y_temp
            self.del_var()
        return corr.cpu(), dist.cpu(), sim.cpu()

    def similarity_matrix_with_pearson_euler(self, data, chunk=1000):
        """
        Description:
            计算输入data之间的相似度矩阵。相似度矩阵=欧氏距离矩阵+Pearson相关性矩阵。计算过程：
                (1). 数据的shift通过data的第二维度引入，即data低维代表每个shift产生的数据。
                     如果对应位置没有shift，则赋值为0. 0的相似性输出为nan
                (2). shift维度计算相似性：生成shape(n_shift^2*2)的index，两两进行计算。
                (3). 根据sim值选取，shift的结果

        Input:
            :param data: torch.tensor, double, shape(n*n_shift*n_time)
                需要计算时间相似度的数据，有n个数据，每个数据有n_shift个平移，数据有n_time个采样点
            :param chunk: number, long
                矩阵分成chunk计算，减少gpu显存使用。

        Return:
            :return sim_shift: torch.tensor, double, shape(n*n*n_shift^2)
                相似度矩阵
            :return corr_shift: torch.tensor, double, shape(n*n*n_shift^2)
                Pearson相关性矩阵
            :return dist_shift: torch.tensor, double, shape(n*n*n_shift^2)
                欧氏距离矩阵
            :return sim: torch.tensor, double, shape(n*n)
                信号对齐后，相似度矩阵
            :return corr: torch.tensor, double, shape(n*n)
                信号对齐后，Pearson相关性矩阵
            :return dist: torch.tensor, double, shape(n*n)
                信号对齐后，欧氏距离矩阵
            :return index_with_max_sim: torch.tensor, double, shape(n*n)
                用来计算相似度矩阵的shift
            :return shift_idx: torch.tensor, double, shape(n_shift^2*2)
                shift的index
        """

        # 获取需要shift的index，data的第二维为shift
        data = data.clone() if len(data.shape) == 3 else data.clone().unsqueeze(1)
        data_x, data_y = data.clone().to(self.device), data.clone().to(self.device)
        shift_idx = torch.stack([torch.arange(0, data_x.shape[1]).repeat_interleave(data_x.shape[1], dim=0),
                                 torch.arange(0, data_y.shape[1]).repeat(data_y.shape[1])], dim=-1).to(self.device)
        corr, dist, sim = [], [], []
        # 对每一个shift，计算相似度距离矩阵
        for idx in shift_idx:
            data_x_temp, data_y_temp = data_x[:, idx[0]], data_y[:, idx[1]]
            # 将矩阵分成chunk计算相关矩阵，减少gpu显存使用
            corr_temp, dist_temp, sim_temp = self.cal_similarity_matrix_with_pearson_euler_chunk(
                data_x=data_x_temp, data_y=data_y_temp, chunk=chunk)
            # 将对角线的相似度设置为0
            corr_temp[(range(corr_temp.shape[0]), range(corr_temp.shape[0]))] = 0
            corr.append(corr_temp)
            dist_temp[(range(dist_temp.shape[0]), range(dist_temp.shape[0]))] = 0
            dist.append(dist_temp)
            sim_temp[(range(sim_temp.shape[0]), range(sim_temp.shape[0]))] = 0
            sim.append(sim_temp)
        # 对于不同shift产生的相似性矩阵，取相似性最大值最大的shift
        sim_shift = torch.stack(sim, dim=-1).half().to(self.device)
        index_with_max_sim = sim_shift.argmax(dim=-1).unsqueeze(-1)
        sim = sim_shift.take_along_dim(index_with_max_sim, dim=-1).squeeze()
        corr_shift = torch.stack(corr, dim=-1).half().to(self.device)
        corr = corr_shift.take_along_dim(index_with_max_sim, dim=-1).squeeze()
        dist_shift = torch.stack(dist, dim=-1).half().to(self.device)
        dist = dist_shift.take_along_dim(index_with_max_sim, dim=-1).squeeze()

        self.del_var()
        return sim, corr, dist, sim_shift, corr_shift, dist_shift, index_with_max_sim.squeeze(), shift_idx

    def sequential_cluster_with_pearson_euler(self, data, sim_threshold=0.75, dist_threshold=0.85, corr_threshold=0.85,
                                              number_threshold=5, zero_cross_threshold=5):
        """
        Description:
            计算输入data之间的相似度矩阵。相似度矩阵=欧氏距离矩阵+Pearson相关性矩阵。计算过程：
                (1). 计算输入数据的相似度矩阵。
                (2). 对于每个输入数据，以及所有的shift，获取和剩余数据之间的sim大于阈值sim_threshold的个数。
                     选取最大值对应的数据以及shift为新的聚类中心。
                (3). 根据聚类中心和剩余数据之间的sim值，选取每个剩余数据的shift。
                (4). 根据shift

        Input:
            :param data: torch.tensor, double, shape(n*n_shift*n_time)
                需要计算时间相似度的数据，有n个数据，每个数据有n_shift个平移，数据有n_time个采样点
            :param sim_threshold: number, double
                相似度阈值，用于每一个类别中心的选择
            :param dist_threshold: number, double
                欧氏距离阈值，用于确定类内的数据
            :param corr_threshold: number, double
                Pearson相关性阈值，用于确定类内的数据
            :param number_threshold: number, double
                类内个数最小数量阈值，保留类内个数最大于number_threshold的类别
            :param zero_cross_threshold: number, double
                过零点阈值，去除振荡较多的template

        Return:
            :return clusters: list, long, shape(n*-)
                每个类别的索引
            :return clusters_data: torch.tensor, double, shape(n*n)
                每个类内的曲线
            :return clusters_center_data: torch.tensor, double, shape(n*n)
                每个类别的平均曲线
        """

        # 获取相似度矩阵
        sim, corr, dist, sim_shift, corr_shift, dist_shift, index_with_max_sim, shift_idx = \
            self.similarity_matrix_with_pearson_euler(data)
        data_unprocessed = torch.tensor([True]).repeat(corr.shape[0])
        # 计算每个满足阈值的cluster
        clusters, clusters_data = [], []
        while data_unprocessed.any():
            # 更新相似度矩阵
            corr_shift_temp = corr_shift[data_unprocessed][:, data_unprocessed]
            dist_shift_temp = dist_shift[data_unprocessed][:, data_unprocessed]
            sim_shift_temp = sim_shift[data_unprocessed][:, data_unprocessed]
            # 获取新类别的聚类中心点，并返回在剩余数据中的index
            # 聚类中心需要距离剩下数据的距离最短。
            shift_index = shift_idx[:, 0].unique()
            new_cluster_center_temp = []
            # 计算不同shift下的，聚类中心
            for shift in shift_index:
                temp = (sim_shift_temp[:, :,
                        torch.where(shift_idx[:, 0] == shift)[0]].amax(dim=-1) > sim_threshold).float().sum(dim=-1)
                new_cluster_center_temp.append(torch.tensor([temp.argmax(), temp.max()]).to(self.device))
            # 选取距离最近的shift为新的聚类中，并且固定聚类中心的shift
            new_cluster_center_shift_index = torch.stack(new_cluster_center_temp)[:, 1].argmax()
            new_cluster_center = torch.stack(new_cluster_center_temp)[:, 0][new_cluster_center_shift_index].long()
            # 将满足欧氏距离和相关系数阈值的数据加入新类别
            # 根据sim的值，确定使用的shift
            cluster_center_shift = shift_idx[:, 0] == new_cluster_center_shift_index
            sim_shift_temp = sim_shift_temp[new_cluster_center][:, cluster_center_shift]
            index_with_max_sim_temp = sim_shift_temp.argmax(dim=-1).unsqueeze(-1)
            # 根据shift，获取对应的相关性和欧式距离
            corr_shift_temp = corr_shift_temp[new_cluster_center][:, cluster_center_shift].take_along_dim(
                index_with_max_sim_temp, dim=-1).squeeze()
            dist_shift_temp = dist_shift_temp[new_cluster_center][:, cluster_center_shift].take_along_dim(
                index_with_max_sim_temp, dim=-1).squeeze()
            # 获取满足阈值的类内元素，并返回在剩余数据中的index
            new_cluster_members_temp = torch.where((corr_shift_temp.abs() > corr_threshold) &
                                                   (dist_shift_temp > dist_threshold))[0]
            if len(new_cluster_members_temp) == 0:
                # 没有满足要求的数据
                new_cluster_members = torch.where(data_unprocessed)[0][new_cluster_center].reshape(-1)
                new_cluster_data = data[new_cluster_members][:, new_cluster_center_shift_index]
                new_cluster_data = \
                    (new_cluster_data - new_cluster_data.amin(dim=-1, keepdim=True)) / \
                    (new_cluster_data.amax(dim=-1, keepdim=True) - new_cluster_data.amin(dim=-1, keepdim=True)) - 0.5
            else:
                # 获取新类别，类内数据在全部数据中的index
                new_cluster_members = torch.where(data_unprocessed)[0][torch.cat([new_cluster_center.reshape(-1),
                                                                                  new_cluster_members_temp])]
                # 获取新类别的shift，并获取对应的数据，切根据corr进行数据的取反
                new_cluster_shift_index = torch.cat([new_cluster_center_shift_index.reshape(-1),
                                                     index_with_max_sim_temp.squeeze()[new_cluster_members_temp]])
                new_cluster_data = data[new_cluster_members].take_along_dim(
                    new_cluster_shift_index.reshape(-1, 1, 1).to(self.device), dim=1).squeeze()
                new_cluster_data = \
                    (new_cluster_data - new_cluster_data.amin(dim=-1, keepdim=True)) / \
                    (new_cluster_data.amax(dim=-1, keepdim=True) - new_cluster_data.amin(dim=-1, keepdim=True))
                new_cluster_data = (new_cluster_data - 0.5) * torch.cat([
                    torch.tensor([1.]).to(self.device), corr_shift_temp[new_cluster_members_temp].sign()]).unsqueeze(-1)
            # 更新data_unprocessed
            data_unprocessed[new_cluster_members] = False
            clusters.append(new_cluster_members)
            clusters_data.append(new_cluster_data)
        # 获取满足类内个数最小数量的cluster
        # number_threshold大于最大类内个数的时候，将number_threshold设置为最大类内个数，保证有template产生
        number_threshold = min(torch.tensor([len(x) for x in clusters]).max(), number_threshold)
        clusters_data = [x for x, y in zip(clusters_data, clusters) if len(y) >= number_threshold]
        clusters = [x for x in clusters if len(x) >= number_threshold]
        clusters_center_data = torch.stack([x.mean(dim=0) for x in clusters_data])
        # 获取满足类内个数最小数量的cluster
        clusters_center_data_temp = [(x - x.min()) / (x.max() - x.min()) - 0.5 for x in clusters_center_data]
        zero_cross = torch.tensor([torch.where((x > 0).diff())[0].shape[0] for x in clusters_center_data_temp])
        clusters_data = [x for x, y in zip(clusters_data, zero_cross) if y < zero_cross_threshold]
        clusters = [x for x, y in zip(clusters, zero_cross) if y < zero_cross_threshold]
        clusters_center_data = torch.stack([x.mean(dim=0) for x in clusters_data])

        self.del_var()
        return clusters, clusters_data, clusters_center_data

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
                duration阈值，单位我为采样点

        Return:
            :return ieds_peaks_large_idx: torch.tensor, double, shape(n*n_channel)
                每个片段是否满足阈值要求
        """

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
            ieds_peaks_large_idx = ieds_peaks_large_idx & (ied_duration_peaks_max <= large_peak_duration_threshold)

        self.del_var()
        return ieds_peaks_large_idx

    def template_matching(self, ied_amplitude_peaks=None, large_peak_amplitude_threshold=0.80,
                          ied_half_slope_peaks=None, large_peak_half_slope_threshold=0.6,
                          ied_sharpness_peaks=None, large_peak_sharpness_threshold=0.6,
                          ied_duration_peaks=None, large_peak_duration_threshold=100,
                          data_segment_all=None, sim_template=0.85, dist_template=0.85, corr_template=0.85,
                          sim_threshold=0.8, dist_threshold=0.8, corr_threshold=0.8, channel_threshold=5):
        """
        Description:
            使用特征值较大的片段，进行时间聚类得到ied模版，然后对所有的ied进行模版匹配

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
                duration阈值，单位我为采样点
            :param data_segment_all: torch.tensor, double, shape(n*n_channel*shift个数*n_samples)
                ied数据片段
            :param sim_template: number, double
                聚类ied模版时候，相似度最小阈值, 范围(0, 1)之间
            :param dist_template: number, double
                聚类ied模版时候，欧氏距离最小阈值, 范围(0, 1)之间
            :param corr_template: number, double
                聚类ied模版时候，相关性系数最小阈值, 范围(0, 1)之间
            :param corr_template: number, double
                聚类ied模版时候，相关性系数最小阈值, 范围(0, 1)之间
            :param sim_threshold: number, double
                进行模版匹配是，相关性系数最小阈值, 范围(0, 1)之间
            :param dist_threshold: number, double
                进行模版匹配是，相关性系数最小阈值, 范围(0, 1)之间
            :param corr_threshold: number, double
                进行模版匹配是，相关性系数最小阈值, 范围(0, 1)之间
            :param channel_threshold: number, long
                通道数最小阈值

        Return:
            :return ied_index: torch.tensor, bool, shape(n)
                每个片段是否满足相似度阈值要求
            :return clusters_data: list, [torch.tensor, double, shape(n_segments*n_samples)], shape(n_cls)
                每个类内的曲线
            :return clusters_center_data: torch.tensor, double, shape(n_cls*n_samples)
                每个类别的平均曲线
        """

        # 判断输入是否存在
        if (ied_amplitude_peaks is None) or (data_segment_all is None):
            return None
        if (ied_half_slope_peaks is None) and (ied_sharpness_peaks is None) and (ied_duration_peaks is None):
            return None

        # 获取特征值较大的片段
        ied_large_peaks_index = self.cal_data_with_large_feature(
            ied_amplitude_peaks=ied_amplitude_peaks, large_peak_amplitude_threshold=large_peak_amplitude_threshold,
            ied_half_slope_peaks=ied_half_slope_peaks, large_peak_half_slope_threshold=large_peak_half_slope_threshold,
            ied_sharpness_peaks=ied_sharpness_peaks, large_peak_sharpness_threshold=large_peak_sharpness_threshold,
            ied_duration_peaks=ied_duration_peaks, large_peak_duration_threshold=large_peak_duration_threshold)
        if torch.where(ied_large_peaks_index)[0].shape[0] < 5:
            ied_large_peaks_index = self.cal_data_with_large_feature(
                ied_amplitude_peaks=ied_amplitude_peaks, large_peak_amplitude_threshold=0.5,
                ied_half_slope_peaks=ied_half_slope_peaks, large_peak_half_slope_threshold=0.5,
                ied_sharpness_peaks=ied_sharpness_peaks, large_peak_sharpness_threshold=0.5,
                ied_duration_peaks=ied_duration_peaks, large_peak_duration_threshold=large_peak_duration_threshold)
        # 如果特征值较大的片段过少则return
        if torch.where(ied_large_peaks_index)[0].shape[0] < 3:
            return torch.ones(data_segment_all.shape[0]) > 0, [], []
        segment_large_all = data_segment_all[ied_large_peaks_index]
        self.del_var()

        # 对大特征片段进行聚类，得到template
        clusters, clusters_data, clusters_center_data = self.sequential_cluster_with_pearson_euler(
            data=segment_large_all.clone(),
            sim_threshold=sim_template, dist_threshold=dist_template, corr_threshold=corr_template)
        self.del_var()
        # 将模版按照第一类为基准，进行翻转
        corr, _, _ = self.cal_similarity_matrix_with_pearson_euler(clusters_center_data, clusters_center_data)
        corr_sign = corr[0].sign().to(self.device)
        clusters_data = [x * y for x, y in zip(clusters_data, corr_sign)]
        clusters_center_data = clusters_center_data * corr_sign.reshape(-1, 1)

        # 计算所有片段和template之间的相似度
        sim, corr, dist, _ = self.similarity_between_segments_templates(data_segment_all, clusters_center_data)

        # 获取大于阈值的ied event
        ied_index = ((sim.amax(dim=-1) > sim_threshold) &
                     (corr.abs().amax(dim=-1) > corr_threshold) &
                     (dist.amax(dim=-1) > dist_threshold)).sum(dim=-1) >= channel_threshold

        return ied_index, clusters_data, clusters_center_data, corr, dist

    @staticmethod
    def export_emhapp(emhapp_save, corr, dist, ied_num_fif, ied_index, clusters_data):
        """
        Description:
            将模版匹配结果添加到emhapp_save中

        Input:
            :param emhapp_save: dict, shape(n_fifs)
                用于保存emhapp mat文件的字典
            :param corr: torch, double, shape(n_ieds, n_channels, n_templates)
                所有fif文件中检测到的IED，与模版之间的相关系数
            :param dist: torch, double, shape(n_ieds, n_channels, n_templates)
                所有fif文件中检测到的IED，与模版之间的欧氏距离
            :param ied_num_fif: list, long, shape(n_fifs)
                每个fif文件中检测到的IED个数
            :param ied_index: torch.tensor, bool, shape(n)
                每个片段是否满足相似度阈值要求
            :param clusters_data: list, [torch.tensor, double, shape(n_segments*n_samples)], shape(n_cls)
                每个类内的曲线

        Return:
            :return emhapp_save_new: dict
                用于保存emhapp mat文件的字典
        """

        # 获取每个fif中ied在全部ied中的索引
        ied_num_fif = torch.arange(corr.shape[0]).split(ied_num_fif)

        # 获取每个fif文件中，通过模版匹配的IED
        ied_index_fif = [ied_index[x].cpu().numpy() for x in ied_num_fif]

        # 获取每个IED事件中被检测为IED的通道索引
        temp = corr.cpu()[:, :, 0]
        temp[~(temp != 0).any(dim=-1), 0] = 1
        temp = torch.where(temp)
        ied_chan = torch.split(temp[1], temp[0].unique(return_counts=True)[1].tolist())

        # 获取EventCorr: 每个IED事件中被检测为IED的通道信号与template之间的相似度
        peak_corr = np.array([x[y].cpu().numpy() for x, y in zip(corr, ied_chan)], dtype=object)
        peak_corr = [peak_corr[x].tolist() for x in ied_num_fif]

        # 获取EventDist: 每个IED事件中被检测为IED的通道信号与template之间的欧氏距离
        peak_dist = np.array([x[y].cpu().numpy() for x, y in zip(dist, ied_chan)], dtype=object)
        peak_dist = [peak_dist[x].tolist() for x in ied_num_fif]

        # 写入emhapp_save
        emhapp_save_new = []
        for x, y, z in zip(emhapp_save, peak_dist, peak_corr):
            if len(x) == 0:
                x = []
            else:
                x['EventDist'] = np.array(y + [0], dtype=object)[:-1]
                x['EventCorr'] = np.array(z + [0], dtype=object)[:-1]
                x['Template'] = np.array([z.cpu().numpy() for z in clusters_data] + [0], dtype=object)[:-1]
            emhapp_save_new.append(x)

        # 仅保留模版匹配的IED
        for i in range(len(emhapp_save_new)):
            if not any(ied_index_fif[i]) or len(emhapp_save_new[i]) == 0:
                emhapp_save_new[i] = []
            else:
                emhapp_save_new[i]['SpikeChan'] = emhapp_save_new[i]['SpikeChan'][ied_index_fif[i]]
                emhapp_save_new[i]['SpikeRegion'] = emhapp_save_new[i]['SpikeRegion'][ied_index_fif[i]]
                emhapp_save_new[i]['BsTime'] = emhapp_save_new[i]['BsTime'][ied_index_fif[i]]
                emhapp_save_new[i]['EventSpikeTime'] = emhapp_save_new[i]['EventSpikeTime'][ied_index_fif[i]]
                emhapp_save_new[i]['Amp'] = emhapp_save_new[i]['Amp'][ied_index_fif[i]]
                emhapp_save_new[i]['Slope'] = emhapp_save_new[i]['Slope'][ied_index_fif[i]]
                emhapp_save_new[i]['Sharp'] = emhapp_save_new[i]['Sharp'][ied_index_fif[i]]
                emhapp_save_new[i]['EventDist'] = emhapp_save_new[i]['EventDist'][ied_index_fif[i]]
                emhapp_save_new[i]['EventCorr'] = emhapp_save_new[i]['EventCorr'][ied_index_fif[i]]

        return emhapp_save_new


    @staticmethod
    def plot_clusters(clusters_data=None, clusters_center_data=None, figsize=(10, 25.6), dpi=100, figure_col=10):
        """
        Description:
            plot event周围MEG数据

        Input:
            :param clusters_data: torch.tensor, double, shape(n*n)
                每个类内的曲线
            :param clusters_center_data: torch.tensor, double, shape(n*n)
                每个类别的平均曲线
            :param figure_col: number, long
                figure中的行数
            :param figsize: tuple, double
                figure的大小
            :param dpi: number, long
                figure的dpi
        """

        if (clusters_data is None) and (clusters_center_data is None):
            return

        if clusters_data is not None:
            if clusters_center_data is None:
                clusters_center_data = torch.stack([x.mean(dim=0) for x in clusters_data])
            figure_col = min(figure_col, len(clusters_data))
            fig = plt.figure(num=233, figsize=(figsize[0], round(len(clusters_data) / figure_col + 0.5) * 1.4),
                             clear=True, dpi=dpi)
            # 计算每个类别位置
            cls_range = range(len(clusters_data))
            figure_row = np.ceil(len(cls_range) / figure_col).astype('int64')
            figure_col_row = torch.stack(
                (torch.arange(1, figure_row + 1).unsqueeze(-1).repeat(1, figure_col).reshape(-1),
                 torch.arange(0, figure_col).unsqueeze(-1).repeat(1, figure_row).t().reshape(-1))).t()
            # 画信号
            for i in cls_range:
                ax = plt.Axes(fig, [1 / figure_col * figure_col_row[i][1], 1 - 1 / figure_row * figure_col_row[i][0],
                                    0.95 / figure_col, 0.85 / figure_row])
                ax.set_axis_off()
                fig.add_axes(ax)
                ax.plot(clusters_data[i].t().cpu(), linewidth=0.5, color=[0.5, 0.5, 0.5])
                ax.plot(clusters_center_data[i].cpu(), linewidth=2, color=[1, 0, 0])
                ax.text(ax.get_xlim()[0] + 5, ax.get_ylim()[1] - 0.01,
                        'Cls: ' + str(i + 1) + ';  ' + 'Num: ' + str(clusters_data[i].shape[0]))
            # 画类别之间的虚线
            for i in range(figure_col_row[:, 0].max() - 1):
                # Plot Line
                ax = plt.Axes(fig, [-1, 1 - 1 / figure_row * figure_col_row[:, 0].unique()[i + 1] + 0.9 / figure_row,
                                    3, 0.15 / figure_row])
                fig.add_axes(ax)
                ax.plot([0, 1], [0, 0], linestyle='--', color='k')
                ax.set_axis_off()
            for i in range(figure_col_row[:, 1].max()):
                # Plot Line
                ax = plt.Axes(fig, [1 / figure_col * figure_col_row[:, 1].unique()[i] + 0.95 / figure_col, -1,
                                    0.05 / figure_col, 3])
                fig.add_axes(ax)
                ax.plot([0, 0], [0, 1], linestyle='--', color='k')
                ax.set_axis_off()
            plt.show()

        if (clusters_data is None) and (clusters_center_data is not None):
            figure_col = min(figure_col, clusters_center_data.shape[0])
            fig = plt.figure(num=233, figsize=(figsize[0], round(clusters_center_data.shape[0] / figure_col + 0.5) * 1),
                             clear=True, dpi=dpi)
            # 计算每个类别位置
            cls_range = range(clusters_center_data.shape[0])
            figure_row = np.ceil(len(cls_range) / figure_col).astype('int64')
            figure_col_row = torch.stack(
                (torch.arange(1, figure_row + 1).unsqueeze(-1).repeat(1, figure_col).reshape(-1),
                 torch.arange(0, figure_col).unsqueeze(-1).repeat(1, figure_row).t().reshape(-1))).t()
            # 画信号
            for i in cls_range:
                ax = plt.Axes(fig, [1 / figure_col * figure_col_row[i][1], 1 - 1 / figure_row * figure_col_row[i][0],
                                    0.95 / figure_col, 0.85 / figure_row])
                ax.set_axis_off()
                fig.add_axes(ax)
                ax.plot(clusters_center_data[i].cpu(), linewidth=2, color=[1, 0, 0])
                ax.text(ax.get_xlim()[0] + 10, ax.get_ylim()[1] - 0.01, 'Cls: ' + str(i))
            # 画类别之间的虚线
            for i in range(figure_col_row[:, 0].max() - 1):
                # Plot Line
                ax = plt.Axes(fig, [-1, 1 - 1 / figure_row * figure_col_row[:, 0].unique()[i + 1] + 0.9 / figure_row,
                                    3, 0.15 / figure_row])
                fig.add_axes(ax)
                ax.plot([0, 1], [0, 0], linestyle='--', color='k')
                ax.set_axis_off()
            for i in range(figure_col_row[:, 1].max()):
                # Plot Line
                ax = plt.Axes(fig, [1 / figure_col * figure_col_row[:, 1].unique()[i] + 0.95 / figure_col, -1,
                                    0.05 / figure_col, 3])
                fig.add_axes(ax)
                ax.plot([0, 0], [0, 1], linestyle='--', color='k')
                ax.set_axis_off()
            plt.show()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(2)
    import ied_detection_multi_files

    fif_list = ['/sanbo_dataset/4k_Project/Data_Full/MEG_IED/MEG1781/MEG1781_EP_1_tsss.fif',
                '/sanbo_dataset/4k_Project/Data_Full/MEG_IED/MEG1781/MEG1781_EP_2_tsss.fif']
    Raw_data, Info = [], []
    for fif in fif_list:
        raw = mne.io.read_raw_fif(fif, verbose='error', preload=True)
        raw.pick(mne.pick_types(raw.info, meg=True, ref_meg=False))
        Info.append(raw.info)
        Raw_data.append(raw.get_data())

    # 计算发作间期IEDs
    IED_Multi = ied_detection_multi_files.ied_detection_threshold_multi(raw_data_s=Raw_data, data_info_s=Info,
                                                                        device=2, n_jobs=10)
    IEDs_peak_all, IEDs_win_all, IEDs_peak_closest_idx_all, Data_segment_all, \
    IEDs_amplitude_peaks_all, IEDs_half_slope_peaks_all, IEDs_sharpness_peaks_all, IEDs_duration_peaks_all, \
    IEDs_amplitude_all, IEDs_half_slope_all, IEDs_sharpness_all, IEDs_duration_all, IEDs_gfp_amp_all = \
        IED_Multi.get_ieds_in_whole_recording(smooth_windows=0.02, smooth_iterations=2, z_win=100,
                                              z_region=True, z_mag_grad=False, exclude_duration=0.02,
                                              peak_amplitude_threshold=(2.5, None), half_peak_slope_threshold=(2, None),
                                              mag_grad_ratio=0.056, chan_threshold=(5, 200), ieds_window=0.3,
                                              data_segment_window=0.15)

    Temporal_Cluster = temporal_cluster(device=IED_Multi.device_number)
    IED_index, Clusters_data, Clusters_center_data = Temporal_Cluster.template_matching(
        ied_amplitude_peaks=torch.cat(IEDs_amplitude_peaks_all), large_peak_amplitude_threshold=0.85,
        ied_half_slope_peaks=torch.cat(IEDs_half_slope_peaks_all), large_peak_half_slope_threshold=0.75,
        ied_duration_peaks=torch.cat(IEDs_duration_peaks_all), large_peak_duration_threshold=100,
        data_segment_all=torch.cat(Data_segment_all),
        sim_template=0.85, dist_template=0.85, corr_template=0.85,
        sim_threshold=0.8, dist_threshold=0.8, corr_threshold=0.8, channel_threshold=5)

    Temporal_Cluster.plot_clusters(clusters_data=Clusters_data, clusters_center_data=Clusters_center_data,
                                   figsize=(10, 35.6), dpi=100, figure_col=10)
