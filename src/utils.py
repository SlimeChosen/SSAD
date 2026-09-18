import torch
import torch.nn.functional as F
import distmap


def knn(points_src, points_tar, k=8):
    """
    :param points_src: [B, N, 3]
    :param points_tar: [B, N, 3]
    :return:        [B, N, k]
    """
    distance = torch.cdist(points_src, points_tar, p=2)

    _, indices = torch.topk(distance, k=k, dim=-1, largest=False, sorted=True)

    return indices

def gather_neighbor_points(points, idx):
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

def gather_ctrl_points(points, idx):
    B, M, C = points.shape
    _, N, S = idx.shape

    points_expanded = points.unsqueeze(1).expand(B, N, M, C)
    idx_expanded = idx.unsqueeze(-1).expand(B, N, S, C)

    gathered_points = torch.gather(points_expanded, dim=2, index=idx_expanded)

    return gathered_points

def generate_offsets(r=3, device='cuda'):
    offset = torch.arange(r, device=device) - (r//2)
    dx, dy, dz = torch.meshgrid(offset, offset, offset)

    offsets = torch.stack([dx.reshape(-1), dy.reshape(-1), dz.reshape(-1)], dim=-1).view(-1, 3)

    return offsets

def batch_norm_d3(inputs):
    inp = torch.sqrt(torch.sum(inputs.mul(inputs) + 1.0e-8, dim=2)).view(inputs.shape[0], inputs.shape[1], 1)

    outp = inputs / (inp + 1.0e-8)

    return outp

def batch_norm_d4(inputs):
    inp = torch.sqrt(torch.sum(inputs.mul(inputs) + 1.0e-8, dim=3)).unsqueeze(-1)

    outp = inputs / (inp + 1.0e-8)

    return outp

def get_mask_edge(voxel):
    """
    :param voxel: [B, D, H, W]
    :return:
    """
    B, D, H, W = voxel.shape

    padded_voxel = F.pad(
        voxel.float(),
        pad=(1, 1, 1, 1, 1, 1),
        mode='constant',
        value=0
    )

    dist = distmap.euclidean_distance_transform(padded_voxel, ndim=3)

    # edge = torch.where((dist > 0) & (dist <= 1.7321), 1, 0)
    # edge = torch.where((dist > 0) & (dist <= 1.0 + 1.0e-6), 1, 0)
    edge = torch.where(dist == 1, 1, 0)

    edge = edge[:, 1:D + 1, 1:H + 1, 1:W + 1]

    return edge

def voxel2points(voxel):
    B, D, H, W = voxel.shape

    points_list = []

    for b in range(B):
        p = torch.nonzero(voxel[b], as_tuple=False)
        points_list.append(p)

    points = torch.stack(points_list, dim=0).float()

    return points

def get_ctrl_points(edge_points, raw_normal):
    """
    :param edge_points: [B, N, 3]
    :param raw_normal: [B, N, 3]
    :return:
    """
    B, N, _ = edge_points.shape
    device = edge_points.device

    abs_normals = torch.abs(raw_normal)

    cond_axis = (abs_normals > (1.0 - 1.0e-6))

    cond_others = (abs_normals < 1.0e-6)

    cond_sum = (torch.sum(abs_normals, dim=-1) < 1.0e-6)

    mask_x = cond_axis[..., 0] & cond_others[..., 1] & cond_others[..., 2]
    mask_y = cond_axis[..., 1] & cond_others[..., 0] & cond_others[..., 2]
    mask_z = cond_axis[..., 2] & cond_others[..., 0] & cond_others[..., 1]

    # mask_boundaries = (((edge_points[:,:,0]).int()==0) | ((edge_points[:,:,0]).int()==voxel.shape[1]-1))

    mask = mask_x | mask_y | mask_z | cond_sum
    mask = ~mask
    # mask = mask | mask_boundaries

    fill_value = 10.0 * torch.max(edge_points)

    M = max(torch.sum(mask.int(), dim=-1))

    ctrl_points = []
    ctrl_indices = []
    valid_mask = []

    for b in range(B):
        batch_mask = mask[b]
        batch_points = edge_points[b][batch_mask]
        batch_indices = torch.where(batch_mask)[0].long()

        full_length = M - batch_points.shape[0]

        m = torch.tensor(batch_indices >= 0, dtype=torch.bool, device=device)

        if full_length == 0:
            points = batch_points
            indices = batch_indices
        else:
            points = torch.cat([batch_points, torch.full((full_length, 3), fill_value, device=device)], dim=0)
            indices = torch.cat([batch_indices, torch.full((full_length,), 0, device=device)])
            m = torch.cat([m, torch.full((full_length,), 0, device=device)])

        ctrl_points.append(points)
        ctrl_indices.append(indices)
        valid_mask.append(m)

    ctrl_points = torch.stack(ctrl_points)
    ctrl_indices = torch.stack(ctrl_indices)
    valid_mask = torch.stack(valid_mask)

    return ctrl_points, ctrl_indices, valid_mask

def get_neighbors_filter_angle(points, normals, k=8, max_coords_diff=1.0, remove_closest=False):
    """
    :param points: [B, N, 3]
    :param normals: [B, N, 3]
    :return:
    """
    B, N, _ = points.shape

    if remove_closest is True:
        indices = knn(points, points, k+1)
        indices = indices[:, :, 1:]
    else:
        indices = knn(points, points, k)

    neighbor_points = gather_neighbor_points(points, indices)
    coords_diff = points.unsqueeze(2) - neighbor_points

    valid_mask = (coords_diff.abs() <= max_coords_diff).all(dim=-1)

    neighbor_normals = gather_neighbor_points(normals, indices)
    dot_products = torch.sum(normals.unsqueeze(2) * neighbor_normals, dim=-1, keepdim=True)

    valid_mask = valid_mask & (dot_products.squeeze(-1) >= 0)

    return indices, valid_mask

def get_ctrl_neighbors_filter_angle(points, raw_normals, ctrl_points, ctrl_raw_normals, k=8, max_coords_diff=1.0, remove_closest=False):
    B, N, _ = points.shape

    if remove_closest is True:
        indices = knn(points, ctrl_points, k+1)
        indices = indices[:, :, 1:]
    else:
        indices = knn(points, ctrl_points, k)

    neighbor_points = gather_ctrl_points(ctrl_points, indices)
    coords_diff = points.unsqueeze(2) - neighbor_points

    valid_mask = (coords_diff.abs() <= max_coords_diff).all(dim=-1)

    neighbor_normals = gather_ctrl_points(ctrl_raw_normals, indices)
    dot_products = torch.sum(raw_normals.unsqueeze(2) * neighbor_normals, dim=-1, keepdim=True)

    valid_mask = valid_mask & (dot_products.squeeze(-1) >= 0)

    return indices, valid_mask

def generate_gradient_normals_distmap(voxel, points):
    """
    :param voxel: [B, D, H, W]
    :param points: [B, N, 3]
    :return: [B, N, 3]
    """
    padded_voxel = F.pad(
        voxel.float(),
        pad=(1,1,1,1,1,1),
        mode='constant',
        value=0
    )

    dist = distmap.euclidean_distance_transform(padded_voxel, ndim=3)
    # dist = distmap.euclidean_distance_transform(voxel.float(), ndim=3)

    gradient = torch.gradient(dist, dim=[1, 2, 3])
    gradient = torch.stack(gradient, dim=-1)

    d = points[..., 0].long() + 1
    h = points[..., 1].long() + 1
    w = points[..., 2].long() + 1

    indices = torch.arange(gradient.size(0)).view(-1, 1).expand_as(d)
    normals = gradient[indices, d, h, w]

    normals = -normals / (torch.norm(normals, dim=-1, keepdim=True) + 1.0e-8)

    return normals

def adjust_normals_direction(pred_normals, gradient_normals):
    dot_products = torch.sum(pred_normals * gradient_normals, dim=-1, keepdim=True)

    adjust_normals = torch.where(dot_products < 0, -pred_normals, pred_normals)

    return adjust_normals

def apply_displacement(edge_points, normals, sdf):
    sdf_expanded = sdf.unsqueeze(-1)

    displacement = normals * sdf_expanded

    new_points = edge_points + displacement

    return new_points























