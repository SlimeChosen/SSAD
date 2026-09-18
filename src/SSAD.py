import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from src.utils import *

def get_voxel_neighbor_code(voxel, points, win_size=3):
    """
    :param voxel: [B, D, H, W]
    :param points: [B, N, 3]
    :param win_size:
    :return:
    """
    B, D, H, W = voxel.shape
    _, N, _ = points.shape
    device = points.device

    offsets = generate_offsets(win_size, device)
    pad_size = win_size // 2

    padded_voxel = F.pad(voxel,
                         (pad_size,pad_size,
                          pad_size,pad_size,
                          pad_size,pad_size),
                         mode='constant',
                         value=0)

    coords = points.long()

    points_expanded = coords.unsqueeze(2) + pad_size
    offsets_expanded = offsets.view(1, 1, win_size ** 3, 3).long()

    new_coords = points_expanded + offsets_expanded

    batch_indices = torch.arange(B, device=device, dtype=torch.long).view(B, 1, 1, 1).expand(-1, N, win_size ** 3, -1)
    indices = torch.cat([batch_indices, new_coords], dim=-1)

    neighbor_code = padded_voxel.long()[indices.unbind(-1)]

    return neighbor_code.float()

def knn_min_dist_threshold(pcd_src, pcd_tar, k, min_dist_diff, remove_closest=False):
    B, N, _ = pcd_src.shape

    distance = torch.cdist(pcd_src, pcd_tar, p=2)

    max_value = torch.max(distance) * 10.0

    distance = torch.where(distance < min_dist_diff, max_value, distance)

    if remove_closest is True:
        _, indices = torch.topk(distance, k=k + 1, dim=-1, largest=False, sorted=True)
        indices = indices[:, :, 1:]
    else:
        _, indices = torch.topk(distance, k=k, dim=-1, largest=False, sorted=True)

    return indices

def fps(points, k, if_random_start=False):
    device = points.device
    B, N, _ = points.shape
    idx = torch.zeros((B, k), dtype=torch.long).to(device)
    distance = torch.ones((B, N)).to(device) * 1e10

    if if_random_start is True:
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    else:
        farthest = torch.zeros((B,), dtype=torch.long).to(device)

    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    for i in range(k):
        idx[:, i] = farthest
        centroid = points[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((points - centroid) ** 2, dim=-1)
        distance = torch.min(distance, dist)
        farthest = torch.argmax(distance, dim=-1)

    return idx

def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    out = points[batch_indices, idx, :]
    return out


class MLP_displacement(nn.Module):
    def __init__(self, in_channels):
        super(MLP_displacement, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, features):
        """
        :param features: [B, N, C]
        :return: [B, N]
        """
        displacement = self.mlp(features)

        return displacement.squeeze(-1)


class MLP_normal(nn.Module):
    def __init__(self, in_channels=128):
        super(MLP_normal, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, features):
        """
        :param features: [B, N, C]
        :return: [B, N, 3]
        """
        normals = self.mlp(features)

        normals = F.normalize(normals, dim=-1)

        return normals


class LPE(nn.Module):
    def __init__(self, neighbor_code_r, out_channels=256):
        super(LPE, self).__init__()

        self.neighbor_code_r = neighbor_code_r

        self.multi_encode_layer = nn.ModuleList()
        for i in range(1, neighbor_code_r+1):
            in_channels = (2 * i + 1) ** 3
            self.multi_encode_layer.append(
                nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=1)
            )

        self.proj_layer = nn.Conv1d(in_channels=64*neighbor_code_r, out_channels=out_channels, kernel_size=1)
        # self.proj_layer = nn.Sequential(
        #     nn.Linear(64 * neighbor_code_r, out_channels),
        #     nn.ReLU(),
        #     nn.Linear(out_channels, out_channels),
        # )

    def forward(self, voxel, edge_points):
        features = get_voxel_neighbor_code(voxel, edge_points, 3)
        features = self.multi_encode_layer[0](features.transpose(1, 2))

        if (self.neighbor_code_r >= 2):
            for i in range(2, self.neighbor_code_r + 1):
                x = get_voxel_neighbor_code(voxel, edge_points, i * 2 + 1)
                x = self.multi_encode_layer[i - 1](x.transpose(1, 2))
                features = torch.cat([features, x], dim=1)

        features = self.proj_layer(features)

        return features.transpose(1, 2)


class SGCA(nn.Module):
    def __init__(self, neighbor_code_r, out_channels, k=8):
        super(SGCA, self).__init__()

        self.neighbor_code_r = neighbor_code_r
        self.k = k

        self.linear1 = nn.Sequential(nn.Conv1d(256, 64, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(64),
                                    nn.LeakyReLU(negative_slope=0.2))

        self.conv1 = nn.Sequential(nn.Conv2d(64 + 6, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv2 = nn.Sequential(nn.Conv2d(64 + 6, 128, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv3 = nn.Sequential(nn.Conv2d(128 + 6, 256, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(256),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear2 = nn.Sequential(nn.Conv1d(512, out_channels, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(out_channels),
                                   nn.LeakyReLU(negative_slope=0.2))

    @staticmethod
    def gather_points(points, idx):
        """
        :param points: [B, N, C]
        :param idx: [B, N, S]
        :return: [B, N, S, C]
        """
        _, _, C = points.shape
        B, N, S = idx.shape

        idx_expanded = idx.unsqueeze(-1).expand(B, N, S, C)
        points_expanded = points.unsqueeze(2).expand(B, N, S, C)
        gathered_points = torch.gather(points_expanded, dim=1, index=idx_expanded)

        return gathered_points

    @staticmethod
    def get_graph_feature(features, points, normals, neighbor_code_r, k=8):
        """
        :param features: [B, N, C]
        :param points: [B, N, 3]
        :param normals: [B, N, 3]
        :param k:
        :return: [B, N, k, C+3]
        """
        indices = knn_min_dist_threshold(points, points, k, neighbor_code_r / 2.0 + 1.0)

        neighbor_points = SGCA.gather_points(points, indices)
        coords_diff = neighbor_points - points.unsqueeze(2)

        neighbor_normals = gather_neighbor_points(normals, indices)
        normals_diff = neighbor_normals - normals.unsqueeze(2)
        normals_diff = batch_norm_d4(normals_diff)

        neighbor_features = gather_neighbor_points(features, indices)

        # new_features = torch.cat([coords_diff, neighbor_normals, neighbor_features], dim=-1)
        new_features = torch.cat([coords_diff, normals_diff, neighbor_features], dim=-1)

        return new_features

    def forward(self, features, edge_points, raw_normals):
        B, _, _ = edge_points.shape

        x1 = self.linear1(features.transpose(1, 2))

        features = self.get_graph_feature(x1.transpose(1, 2), edge_points, raw_normals, self.neighbor_code_r, self.k)
        features = self.conv1(features.permute(0, 3, 1, 2))
        x2 = features.max(dim=-1, keepdim=False)[0]
        # x2 = features.mean(dim=-1, keepdim=False)

        features = self.get_graph_feature(x2.transpose(1, 2), edge_points, raw_normals, self.neighbor_code_r, self.k)
        features = self.conv2(features.permute(0, 3, 1, 2))
        x3 = features.max(dim=-1, keepdim=False)[0]
        # x3 = features.mean(dim=-1, keepdim=False)

        features = self.get_graph_feature(x3.transpose(1, 2), edge_points, raw_normals, self.neighbor_code_r, self.k)
        features = self.conv3(features.permute(0, 3, 1, 2))
        x4 = features.max(dim=-1, keepdim=False)[0]
        # x4 = features.mean(dim=-1, keepdim=False)

        features = torch.cat([x1, x2, x3, x4], dim=1)
        # features = torch.cat([x0.transpose(1, 2), x4], dim=1)

        features = self.linear2(features)

        return features.transpose(1, 2)


class SSAD(nn.Module):
    def __init__(self, cfg):
        super(SSAD, self).__init__()

        neighbor_code_r = cfg['model']['neighbor_code_r']

        self.lpe = LPE(neighbor_code_r, 256)
        self.sgca = SGCA(neighbor_code_r, 256, cfg['model']['k'])

        self.mlp_displacement = MLP_displacement(256)
        self.mlp_normal = MLP_normal(256)

    def forward(self, voxel):
        with torch.no_grad():
            edge_voxel = get_mask_edge(voxel)
            edge_points = voxel2points(edge_voxel)
            raw_normal = generate_gradient_normals_distmap(voxel, edge_points)

        features = self.lpe(voxel, edge_points)
        features = self.sgca(features, edge_points, raw_normal)

        displacement = self.mlp_displacement(features)
        d_normal = self.mlp_normal(features)

        return edge_points, raw_normal, displacement, d_normal








