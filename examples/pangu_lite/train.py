import argparse
import os

from torch.utils.data import DataLoader
from torch import nn
import torch
from tqdm import tqdm
import pandas as pd

# vscode Relative path
import sys
sys.path.append("../../")

from weatherlearn.models import Pangu_lite
from data_utils import DatasetFromFolder, surface_inv_transform, upper_air_inv_transform


parser = argparse.ArgumentParser(description="Train Pangu_lite Models")
parser.add_argument("--num_epochs", default=50, type=int, help="train epoch number")


if __name__ == "__main__":
    opt = parser.parse_args()

    NUM_EPOCHS = opt.num_epochs

    train_set = DatasetFromFolder("data", "train")
    val_set = DatasetFromFolder("data", "valid")
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False)

    land_mask, soil_type, topography = train_set.get_constant_mask()
    surface_mask = torch.stack([land_mask, soil_type, topography], dim=0)  # C Lat Lon
    lat, lon = train_set.get_lat_lon()

    pangu_lite = Pangu_lite()
    print("# parameters: ", sum(param.numel() for param in pangu_lite.parameters()))

    surface_criterion = nn.L1Loss()
    upper_air_criterion = nn.L1Loss()

    if torch.cuda.is_available():
        pangu_lite.cuda()
        surface_criterion.cuda()
        upper_air_criterion.cuda()

        surface_mask = surface_mask.cuda()

    surface_invTrans, surface_variables = surface_inv_transform("data/surface_mean.pkl", "data/surface_std.pkl")
    upper_air_invTrans, upper_air_variables, upper_air_pLevels = upper_air_inv_transform("data/upper_air_mean.pkl", "data/upper_air_std.pkl")

    # The learning rate is 5e-4 as in the paper, while the weight decay is 3e-6
    optimizer = torch.optim.Adam(pangu_lite.parameters(), lr=5e-4, weight_decay=3e-6)

    results = {'loss': [], 'surface_mse': [], 'upper_air_mse': []}

    for epoch in range(1, NUM_EPOCHS + 1):
        train_bar = tqdm(train_loader)
        running_results = {"batch_sizes": 0, "loss": 0}

        pangu_lite.train()
        for input_surface, input_upper_air, target_surface, target_upper_air in train_bar:
            batch_size = input_surface.size(0)
            if torch.cuda.is_available():
                input_surface = input_surface.cuda()
                input_upper_air = input_upper_air.cuda()
                target_surface = target_surface.cuda()
                target_upper_air = target_upper_air.cuda()

            output_surface, output_upper_air = pangu_lite(input_surface, surface_mask, input_upper_air)

            optimizer.zero_grad()
            surface_loss = surface_criterion(output_surface, target_surface)
            upper_air_loss = upper_air_criterion(output_upper_air, target_upper_air)
            loss = upper_air_loss + surface_loss * 0.25
            loss.backward()
            optimizer.step()

            running_results["loss"] += loss.item() * batch_size
            running_results["batch_sizes"] += batch_size

            train_bar.set_description(desc="[%d/%d] Loss: %.4f" % 
                                      (epoch, NUM_EPOCHS, running_results["loss"] / running_results["batch_sizes"]))

        with torch.no_grad():
            val_bar = tqdm(val_loader)
            valing_results = {"batch_sizes": 0, "surface_mse": 0, "upper_air_mse": 0} 
            for val_input_surface, val_input_upper_air, val_target_surface, val_target_upper_air, times in val_bar:
                batch_size = val_input_surface.size(0)
                if torch.cuda.is_available():
                    val_input_surface = val_input_surface.cuda()
                    val_input_upper_air = val_input_upper_air.cuda()
                    val_target_surface = val_target_surface.cuda()
                    val_target_upper_air = val_target_upper_air.cuda()

                val_output_surface, val_output_upper_air = pangu_lite(val_input_surface, surface_mask, val_input_upper_air)

                val_output_surface = val_output_surface.squeeze(0)  # C Lat Lon
                val_output_upper_air = val_output_upper_air.squeeze(0)  # C Pl Lat Lon

                val_target_surface = val_target_surface.squeeze(0)
                val_target_upper_air = val_target_upper_air.squeeze(0)

                valing_results["batch_sizes"] += batch_size

                surface_mse = ((val_output_surface - val_target_surface) ** 2).data.mean().cpu().item()
                upper_air_mse = ((val_output_upper_air - val_target_upper_air) ** 2).data.mean().cpu().item()

                valing_results["surface_mse"] += surface_mse * batch_size
                valing_results["upper_air_mse"] += upper_air_mse * batch_size

                val_bar.set_description(desc="[validating] Surface MSE: %.4f Upper Air MSE: %.4f" % 
                                        (valing_results["surface_mse"] / valing_results["batch_sizes"], valing_results["upper_air_mse"] / valing_results["batch_sizes"]))
        
        os.makedirs("epochs", exist_ok=True)
        torch.save(pangu_lite.state_dict(), "epochs/pangu_lite_epoch_%d.pth" % (epoch))

        results["loss"].append(running_results["loss"] / running_results["batch_sizes"])
        results["surface_mse"].append(valing_results["surface_mse"] / valing_results["batch_sizes"])
        results["upper_air_mse"].append(valing_results["upper_air_mse"] / valing_results["batch_sizes"])

        data_frame = pd.DataFrame(
            data=results, 
            index=range(1, epoch + 1)
        )
        save_root = "train_logs"
        if not os.path.exists(save_root):
            os.makedirs(save_root)

        data_frame.to_csv(os.path.join(save_root, "logs.csv"), index_label="Epoch")
