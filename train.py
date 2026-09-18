import os
import argparse
import yaml
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from src.SSAD import SSAD
from src.loss import Criterion
from src.dataset import SSAD_Dataset


import warnings
warnings.filterwarnings('ignore', category=UserWarning)


def train(args):
    with open(args.config_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    if not os.path.exists(cfg['train']['out_path']):
        os.makedirs(cfg['train']['out_path'])

    data_json_path = cfg['train']['json_path']

    train_loader = DataLoader(
        SSAD_Dataset(json_path=data_json_path, label_id=cfg['train']['label_id']),
        num_workers=8, batch_size=1, shuffle=True, drop_last=False)

    device = torch.device("cuda")

    model = SSAD(cfg)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg['train']['lr'], weight_decay=1e-4)

    criterion = Criterion(cfg)

    scheduler = CosineAnnealingLR(optimizer, cfg['train']['epochs'], eta_min=cfg['train']['lr'])

    best_loss = 100000

    iterator = tqdm.tqdm(range(cfg['train']['epochs']))
    for epoch in iterator:
        iters = 0.0
        train_loss_dict = {}

        model.train()

        for voxel in train_loader:
            voxel = voxel.to(device)

            optimizer.zero_grad()

            edge_points, raw_normal, pred_displacement, pred_d_normal = model(voxel)

            loss, loss_dict = criterion(edge_points, raw_normal, pred_displacement, pred_d_normal)

            loss.backward()
            optimizer.step()

            iters += 1
            for loss_name in loss_dict.keys():
                if loss_name not in train_loss_dict:
                    train_loss_dict[loss_name] = loss_dict[loss_name]
                else:
                    train_loss_dict[loss_name] += loss_dict[loss_name]

        scheduler.step()

        for loss_name in train_loss_dict.keys():
            train_loss_dict[loss_name] = f"{train_loss_dict[loss_name] / iters:.6f}"

        iterator.set_description("Train loss: " + train_loss_dict['total'])

        torch.save(model.state_dict(), cfg['train']['out_path'] + '/last.pth')

        if best_loss > float(train_loss_dict['total']):
            best_loss = float(train_loss_dict['total'])

            torch.save(model.state_dict(), cfg['train']['out_path'] + '/best_loss.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='config/config.yaml')
    args = parser.parse_args()

    args.config_path = 'config/config.yaml'
    train(args)
















