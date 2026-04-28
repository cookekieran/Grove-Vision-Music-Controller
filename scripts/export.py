"""
Export .pth -> .onnx

This script loads a trained Convolutional Neural Network from a PyTorch 
checkpoint (.pth) and exports it to the ONNX format for cross-platform deployment.

Requirements:
    - A trained '.pth' file (generated via notebooks/CNN_v3.ipynb)
"""

import torch
import torch.onnx
import torch.nn as nn

device = 'cuda' if torch.is_cuda_avaliable() else 'cpu'

class TinyCNN(torch.nn.Module):
    def __init__(self, num_outputs):
        super(TinyCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.rl1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.rl2 = nn.ReLU()

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.rl3 = nn.ReLU()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        
        self.linear1 = nn.Linear(64, 32)
        self.rl4 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        self.linear2 = nn.Linear(32, num_outputs)

    def forward(self, x):
        x = self.rl1(self.conv1(x))
        x = self.rl2(self.conv2(x))
        x = self.rl3(self.conv3(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.rl4(self.linear1(x))
        x = self.linear2(x)
        return x

def init_weights(m):
    if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)

num_classes = 2
model = TinyCNN(num_outputs=num_classes).to(device)
model.apply(init_weights)

def export_to_onnx(model_path, output_name="model.onnx"):
    model = TinyCNN(num_outputs=num_classes)

    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    
    model.eval()

    dummy_input = torch.randn(1, 1, 96, 96)

    print(f"Exporting {model_path} to {output_name}...")
    torch.onnx.export(
        model, 
        dummy_input, 
        output_name,
        export_params=True,        # Store the trained parameter weights inside the file
        opset_version=11,          # Standard version for Edge AI compilers
        do_constant_folding=True,  # Optimisation
        input_names=['input'],    
        output_names=['output']    
    )
    print("Success!")

if __name__ == "__main__":
    export_to_onnx("best_model.pth")