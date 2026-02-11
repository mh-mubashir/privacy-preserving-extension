import os 
import sys
file_path = os.path.abspath(__file__)  
cur_dir = os.path.dirname(file_path)
sys.path.append(cur_dir)


from cifar_like.mlp_mixer import MLPMixer
from cifar_like.resnet import ResNet18  
from cifar_like.vit import ViT
from torch import nn




def get_model(model_name):
    if model_name == "resnet":
        model = ResNet18()
        model.linear = nn.Linear(512, 1)
        return model
    elif model_name == "vit":
        return ViT(image_size=32, patch_size=4, num_classes=1, dim=256, depth=12, heads=8, mlp_dim=512, dropout=0.1, emb_dropout=0.1)
    elif model_name == "mlp_mixer":
        return MLPMixer( image_size=32, channels=3, patch_size=4, dim=256, depth=12, num_classes=1, expansion_factor=4, expansion_factor_token=0.5, dropout=0.1)
    else:
        raise ValueError(f"Model {model_name} not available")