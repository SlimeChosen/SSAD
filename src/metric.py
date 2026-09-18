import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.signal import convolve2d
from scipy.stats import wasserstein_distance
import open3d as o3d


def compute_bi_chamfer_distance_L1(points1, points2):
    """
    :param points1: [N, 3]
    :param points2: [M, 3]
    :return:
    """
    dist1 = np.mean(np.min(cdist(points1, points2, 'euclidean'), axis=1))
    dist2 = np.mean(np.min(cdist(points2, points1, 'euclidean'), axis=1))

    return dist1 + dist2

def compute_bi_chamfer_distance_L2(points1, points2):
    """
    :param points1: [N, 3]
    :param points2: [M, 3]
    :return:
    """
    dist1 = np.mean(np.min(cdist(points1, points2, 'euclidean') ** 2, axis=1))
    dist2 = np.mean(np.min(cdist(points2, points1, 'euclidean') ** 2, axis=1))

    return dist1 + dist2

def compute_chamfer_distance_L1(points1, points2, batch_size=10000):
    """
    :param points1: [N, 3]
    :param points2: [M, 3]
    :return:
    """
    min_dist1 = []

    for i in range(0, len(points1), batch_size):
        p = points1[i : i + batch_size]
        batch_min_dist = np.min(cdist(p, points2, 'euclidean'), axis=1)
        min_dist1.append(batch_min_dist)

    min_dist1 = np.concatenate(min_dist1)

    dist1 = np.mean(min_dist1)

    return dist1

def compute_chamfer_distance_L2(points1, points2):
    """
    :param points1: [B, N, 3]
    :param points2: [B, M, 3]
    :return:
    """
    dist1 = np.mean(np.min(cdist(points1, points2, 'euclidean') ** 2, axis=1))

    return dist1

def compute_bi_hausdorff_distance(points1, points2):
    dist = cdist(points1, points2)

    dist1 = np.min(dist, axis=0)
    dist1 = np.max(dist1)

    dist2 = np.min(dist, axis=1)
    dist2 = np.max(dist2)

    return max(dist1, dist2)

def compute_hausdorff_distance(points1, points2):
    dist = cdist(points1, points2)

    dist = np.min(dist, axis=1)

    dist = np.max(dist)

    return dist

def compute_EMD(points1, points2):
    flat1 = points1.flatten()
    flat2 = points2.flatten()

    return wasserstein_distance(flat1, flat2)

def get_slice_edge(slice):
    kernel = np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ])

    padded = np.pad(slice, 1, mode='constant', constant_values=False)
    conv_result = convolve2d(padded.astype(int), kernel, mode='valid')

    return slice & (conv_result < 4)

def get_voxel_contours(label_voxel):
    edge_voxel = np.zeros_like(label_voxel)

    for d in range(label_voxel.shape[0]):
        edge_voxel[d] = get_slice_edge(label_voxel[d])

    contours = np.where(edge_voxel > 0)

    contours = np.stack(contours, axis=0).T

    return contours

def generate_voxel_points(voxel_shape):
    D, H, W = voxel_shape

    d = torch.arange(D, dtype=torch.int64)
    h = torch.arange(H, dtype=torch.int64)
    w = torch.arange(W, dtype=torch.int64)

    grid_d, grid_h, grid_w = torch.meshgrid(d, h, w, indexing='ij')

    grid = torch.stack([grid_d, grid_h, grid_w], dim=-1)

    points = grid.reshape(-1, 3)

    return points



















