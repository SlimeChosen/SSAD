import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import *


def compute_compact_loss(points, neighbor_points, valid_mask):
    """
    :param points: [B, N, 3]
    :param neighbor_points: [B, N, k, 3]
    :param valid_mask: [B, N, k]
    :return:
    """
    B, N, _ = points.shape

    neighbors_count = torch.sum(valid_mask, dim=-1, keepdim=True)

    dist = torch.norm(neighbor_points - points.unsqueeze(-2), p=2, dim=-1)
    dist_mean = torch.mean(dist, dim=-1, keepdim=True)

    loss_compact = ((dist - dist_mean) ** 2) * valid_mask

    loss_compact = torch.sum(loss_compact, dim=-1) / (neighbors_count.squeeze(-1) + 1.0e-8)

    loss_compact = torch.mean(loss_compact)

    return loss_compact

def compute_curvature_loss(points, neighbor_points, normals, valid_mask):
    """
    :param points: [B, N, 3]
    :param neighbor_points: [B, N, k, 3]
    :param normals: [B, N, 3]
    :param valid_mask: [B, N, k]
    :return:
    """
    neighbors_count = torch.sum(valid_mask, dim=-1, keepdim=True)

    coords_diff = (neighbor_points - points.unsqueeze(2)) * valid_mask.unsqueeze(-1)

    dot_product = torch.sum(coords_diff * normals.unsqueeze(2), dim=-1)
    projections_parallel_length = torch.sqrt(dot_product ** 2 + 1.0e-8)
    projections_parallel = dot_product.unsqueeze(-1) * normals.unsqueeze(2)
    projections_vertical = (coords_diff - projections_parallel) * valid_mask.unsqueeze(-1)  # [B, N, k, 3]

    dist = torch.norm(projections_vertical, dim=-1) * valid_mask # [B, N, k]
    max_dist = torch.max(dist, dim=-1)[0].detach() + 1.0e-8

    weights = max_dist.unsqueeze(-1) / (dist + 1.0e-8) * valid_mask
    sum_weights = torch.sum(weights, dim=-1, keepdim=True)
    weights = weights / (sum_weights + 1.0e-8)

    loss_curvature = projections_parallel_length * weights

    loss_curvature = loss_curvature * valid_mask

    loss_curvature = torch.sum(loss_curvature, dim=-1)

    loss_curvature = torch.mean(loss_curvature)

    return loss_curvature

def compute_normal_smooth_loss(normals, neighbor_normals, valid_mask):
    """
    :param neighbor_normals: [B, N, k, 3]
    :param valid_mask: [B, N, k]
    :return:
    """
    B, N, k, _ = neighbor_normals.shape

    neighbors_count = torch.sum(valid_mask, dim=-1, keepdim=True)

    normals_repeat = normals.repeat(1, 1, k).reshape(neighbor_normals.shape)
    # point_wise_loss = 1.0 - torch.abs(F.cosine_similarity(normals_repeat, neighbor_normals, dim=-1))
    point_wise_loss = 1.0 - F.cosine_similarity(normals_repeat, neighbor_normals, dim=-1)

    point_wise_loss = torch.sum(point_wise_loss * valid_mask, dim=-1, keepdim=True) / (neighbors_count + 1.0e-8)

    loss_normal_smooth = torch.mean(point_wise_loss)

    return loss_normal_smooth

def compute_gaussian_smooth_loss(points, neighbor_points, valid_mask):
    """
    :param points: [B, N, 3]
    :param neighbor_points: [B, N, k, 3]
    :param valid_mask: [B, N, k]
    :return:
    """

    dist = torch.sum((points.unsqueeze(2) - neighbor_points)**2, dim=-1)

    weights = torch.exp(-dist) * valid_mask

    sum_weights = torch.sum(weights, dim=2, keepdim=True)

    smooth_points = torch.sum(weights.unsqueeze(-1) * neighbor_points, dim=2) / (sum_weights + 1.0e-8)

    loss_gaussian_smooth = torch.mean((points - smooth_points)**2)

    return loss_gaussian_smooth


class Criterion(nn.Module):
    def __init__(self, cfg):
        super(Criterion, self).__init__()

        self.w_dil = cfg['train']['loss']['w_dil']
        self.w_normal = cfg['train']['loss']['w_normal']
        self.w_compact = cfg['train']['loss']['w_compact']
        self.w_gaussian = cfg['train']['loss']['w_gaussian']
        self.w_curv = cfg['train']['loss']['w_curv']

    def forward(self, edge_points, raw_normal, pred_displacement, pred_d_normal):
        normals = raw_normal + pred_d_normal
        normals = F.normalize(normals, dim=-1)
        normals = adjust_normals_direction(normals, raw_normal)
        edge_points_sdf = apply_displacement(edge_points, normals, pred_displacement)

        """ctrl_points"""
        ctrl_points, ctrl_indices, ctrl_mask = get_ctrl_points(edge_points, raw_normal)
        ctrl_raw_normals = torch.gather(raw_normal, dim=1, index=ctrl_indices.unsqueeze(-1).expand(-1, -1, 3))
        ctrl_points_sdf = torch.gather(edge_points_sdf, dim=1, index=ctrl_indices.unsqueeze(-1).expand(-1, -1, 3))

        """dilation"""
        loss_dilation = torch.mean(torch.exp(-pred_displacement))

        """curvature"""
        indices, valid_mask = get_ctrl_neighbors_filter_angle(
            edge_points_sdf, raw_normal, ctrl_points, ctrl_raw_normals, 64, 15, remove_closest=True)
        neighbor_points_sdf = gather_ctrl_points(ctrl_points_sdf, indices)
        loss_curvature = compute_curvature_loss(edge_points_sdf, neighbor_points_sdf, normals, valid_mask)

        """normal"""
        indices, valid_mask = get_neighbors_filter_angle(
            edge_points, raw_normal, 32, 10, remove_closest=True)
        neighbor_normals = gather_neighbor_points(normals, indices)
        loss_normal = compute_normal_smooth_loss(normals, neighbor_normals, valid_mask)

        """compact"""
        indices, valid_mask = get_neighbors_filter_angle(
            edge_points_sdf, raw_normal, 32, 10, remove_closest=True)
        neighbor_points_sdf = gather_neighbor_points(edge_points_sdf, indices)
        loss_compact = compute_compact_loss(edge_points_sdf, neighbor_points_sdf, valid_mask)

        """gaussian"""
        indices, valid_mask = get_neighbors_filter_angle(
            edge_points_sdf, raw_normal, 64, 10, remove_closest=True)
        neighbor_points_sdf = gather_neighbor_points(edge_points_sdf, indices)
        loss_gaussian = compute_gaussian_smooth_loss(edge_points_sdf, neighbor_points_sdf, valid_mask)

        loss = (
            + self.w_dil * loss_dilation
            + self.w_normal * loss_normal
            + self.w_compact * loss_compact
            + self.w_gaussian * loss_gaussian
            + self.w_curv * loss_curvature
        )

        loss_dict = {
            'total': loss.item(),

            'loss_dilation': loss_dilation.item(),
            'loss_normal': loss_normal.item(),
            'loss_compact': loss_compact.item(),
            'loss_gaussian': loss_gaussian.item(),
            'loss_curvature': loss_curvature.item(),
        }

        return loss, loss_dict






