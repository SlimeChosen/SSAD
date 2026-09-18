import os
import argparse
import SimpleITK as sitk
import yaml
import torch

from src.SSAD import SSAD
from src.postprocessing import PostProcessing


def test(args):
    with open(args.config_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    save_path = cfg['test']['save_path']
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SSAD(cfg)
    state_dict = torch.load(args.model_path, weights_only=True)
    model.load_state_dict(state_dict)

    model = model.to(device).eval()

    label = sitk.ReadImage(args.data_path)
    voxel = sitk.GetArrayFromImage(label)
    spacing = label.GetSpacing()

    voxel[voxel != args.label_id] = 0
    voxel[voxel > 0] = 1
    voxel = torch.tensor(voxel).to(device)

    with torch.no_grad():
        edge_points, raw_normal, pred_displacement, pred_d_normal = model(voxel.unsqueeze(0))

        edge_points_displacement, normals = PostProcessing.get_displaced_points_normals(
            edge_points, raw_normal, pred_displacement, pred_d_normal)

        # small amount of smoothing
        mesh = PostProcessing.generate_mesh(
            edge_points_displacement[0], normals[0], spacing,
            smooth=True, laplacian_iterations=10, laplacian_lambda=0.1
        )

        PostProcessing.save_mesh(args.save_path, mesh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config/config.yaml')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True, help='path to segmentation label')
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--label_id', type=int, required=True, default=1)

    args = parser.parse_args()

    test(args)








