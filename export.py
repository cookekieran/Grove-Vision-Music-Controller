import torch
import torch.onnx

device = 'cpu'

# 1. THE BLUEPRINT: Copy your CNN class definition from your notebook here
class CNN(torch.nn.Module):
    def __init__(self, num_outputs):
        super(CNN, self).__init__()
        self.convl = torch.nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1)
        self.rll = torch.nn.ReLU()
        self.maxl = torch.nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv2 = torch.nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2)
        self.rl2 = torch.nn.ReLU()
        self.max2 = torch.nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.conv3 = torch.nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.rl3 = torch.nn.ReLU()
        
        self.conv4 = torch.nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.rl4 = torch.nn.ReLU()
        
        self.conv5 = torch.nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.rl5 = torch.nn.ReLU()
        self.max3 = torch.nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.flatten = torch.nn.Flatten()
        self.linearl = torch.nn.Linear(256, 4096)
        self.rl6 = torch.nn.ReLU()
        self.dropoutl = torch.nn.Dropout(0.5)
        
        self.linear2 = torch.nn.Linear(4096, 4096)
        self.rl7 = torch.nn.ReLU()
        self.dropout2 = torch.nn.Dropout(0.5)
        
        self.linear3 = torch.nn.Linear(4096, num_outputs)

    def forward(self, x):
        out = self.maxl(self.rll(self.convl(x)))
        out = self.max2(self.rl2(self.conv2(out)))
        out = self.rl3(self.conv3(out))
        out = self.rl4(self.conv4(out))
        out = self.rl5(self.conv5(out))
        out = self.max3(out)
        out = self.flatten(out)
        out = self.dropoutl(self.rl6(self.linearl(out)))
        out = self.dropout2(self.rl7(self.linear2(out)))
        out = self.linear3(out)
        return out

def init_weights(m):
    if type(m) == torch.nn.Linear or type(m) == torch.nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)

num_classes = 4
model = CNN(num_outputs=num_classes).to(device)
model.apply(init_weights)

def export_to_onnx(model_path, output_name="model.onnx"):
    num_classes = 4 
    model = CNN(num_outputs=num_classes)

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    model.eval()

    dummy_input = torch.randn(1, 1, 96, 96)

    print(f"Exporting {model_path} to {output_name}...")
    torch.onnx.export(
        model, 
        dummy_input, 
        output_name,
        export_params=True,        # Store the trained parameter weights inside the file
        opset_version=11,          # Standard version for most Edge AI compilers
        do_constant_folding=True,  # Optimization: simplifies the graph
        input_names=['input'],     # Name of the input node
        output_names=['output']    # Name of the output node
    )
    print("Success!")

if __name__ == "__main__":
    export_to_onnx("best_model.pth")