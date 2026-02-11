import torch
import torch.nn as nn
import torch.nn.functional as F
from ptflops import get_model_complexity_info
import torchprofile

class RPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, dropout=0.5):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                # nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: [batch_size, input_dim]
        return self.net(x)  # logits

if __name__ == "__main__":
    input_dim = 10000
    hidden_dims = [4096,2048,1024,512,256,128]
    num_classes = 1
    model = RPClassifier(input_dim, hidden_dims, num_classes)
    
    x = torch.randn(1, input_dim)
    # y = model(x)
    macs = torchprofile.profile_macs(model, (x,))
    print(f"MACs: {macs}")
# 在训练循环中：
#  logits = model(x_proj)           # x_proj: [B,100]
#  loss   = criterion(logits, y)
#  loss.backward(); optimizer.step()

