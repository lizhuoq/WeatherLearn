import onnx
from onnx2torch import convert
import numpy as np
import os
import torch
from torch import optim
from torch import nn
from tqdm import tqdm
import json
import argparse
from tools import adjust_learning_rate


parser = argparse.ArgumentParser(description='finetune')
parser.add_argument('--learning_rate', type=float, default=1e-6, help='optimizer learning rate')
parser.add_argument('--lradj', type=str, default='cosine', help='adjust learning rate')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')

args = parser.parse_args()
print(args)

model_path = "pangu_weather_1.onnx"
onnx_model = onnx.load(model_path)
torch_model = convert(onnx_model, True)

input_data_dir = "input_data"
output_data_dir = "output_data"
target_data_dir = "target_data"

# Load the upper-air numpy arrays
input = np.load(os.path.join(input_data_dir, 'input_upper.npy')).astype(np.float32)
# Load the surface numpy arrays
input_surface = np.load(os.path.join(input_data_dir, 'input_surface.npy')).astype(np.float32)

target = np.load(os.path.join(target_data_dir, "target_upper.npy")).astype(np.float32)
target_surface = np.load(os.path.join(target_data_dir, 'target_surface.npy')).astype(np.float32)

target = torch.tensor(target)
target_surface = torch.tensor(target_surface)

model_optim = optim.Adam(torch_model.parameters(), lr=args.learning_rate)
criterion = nn.L1Loss()
torch_model.train()
train_loss = []
for epoch in tqdm(range(args.train_epochs)):
    # batch = 1
    # device cpu
    model_optim.zero_grad()
    output, output_surface = torch_model(input=torch.tensor(input), input_surface=torch.tensor(input_surface))
    # We use the MAE loss to train the model
    # The weight of surface loss is 0.25
    # Different weight can be applied for differen fields if needed
    loss = criterion(output, target) + criterion(output_surface, target_surface) * 0.25

    loss.backward()
    model_optim.step()

    train_loss.append(loss.item())

    adjust_learning_rate(model_optim, epoch + 1, args)

    print("Epoch: {0} | Train Loss: {1:.7f}".format(
                epoch + 1, loss.item()))
    

with open("train_results.json", "w") as f:
    json.dump(train_loss, f)
