import matplotlib.pyplot as plt
import torch
import numpy as np
import open3d as o3d
import pyvista as pv
# from skimage import measure
from tqdm import tqdm


from src.utils import *

class PostProcessing():
    @staticmethod
    def get_displaced_points_normals(edge_points, raw_normal, pred_displacement, pred_d_normal):
        normals = raw_normal + pred_d_normal
        normals = batch_norm_d3(normals)
        normals = adjust_normals_direction(normals, raw_normal)

        edge_points_displacement = apply_displacement(edge_points, normals, pred_displacement)

        return edge_points_displacement.detach().cpu().numpy(), normals.detach().cpu().numpy()

    @staticmethod
    def generate_mesh(points, normals, spacing, smooth=True, laplacian_iterations=10, laplacian_lambda=0.1):
        pcd_points = np.array(points)
        pcd_points[:, 0] = pcd_points[:, 0] * spacing[2]
        pcd_points[:, 1] = pcd_points[:, 1] * spacing[1]
        pcd_points[:, 2] = pcd_points[:, 2] * spacing[0]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_points)
        pcd.normals = o3d.utility.Vector3dVector(np.array(normals))

        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)

        clusters, clusters_n, _ = mesh.cluster_connected_triangles()
        clusters = np.asarray(clusters)
        clusters_n = np.asarray(clusters_n)
        largest_idx = clusters_n.argmax()
        remove_mask = clusters != largest_idx
        mesh.remove_triangles_by_mask(remove_mask)

        if smooth is True:
            mesh = mesh.filter_smooth_laplacian(number_of_iterations=laplacian_iterations, lambda_filter=laplacian_lambda)
            mesh.remove_unreferenced_vertices()

            mesh = mesh.filter_smooth_simple(5)

        mesh = mesh.remove_duplicated_vertices()
        mesh = mesh.remove_degenerate_triangles()
        mesh = mesh.remove_non_manifold_edges()

        return mesh

    @staticmethod
    def save_mesh(save_path, mesh):
        o3d.io.write_triangle_mesh(save_path, mesh)
































