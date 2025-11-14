import os
import argparse
import torch
import torch.nn as nn
import torchvision.models as models
import mlflow
import mlflow.pytorch
from utils import train
from transfer import print_model
from modules import ResNet18PlusClassifier
from peft import get_peft_model, LoraConfig
from memory import set_gpu_memory_limit, print_gpu_utilization 
import yaml

def parse_yaml_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return argparse.Namespace(**config)

def launch():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    args_config = parser.parse_args()
    args = parse_yaml_config(args_config.config)  # Load actual training args from YAML
    args.device = torch.device("cuda:0" if torch.cuda.device_count() > 0 else "cpu")
    set_gpu_memory_limit(args.memory_limit)
    print_gpu_utilization()
    model = ResNet18PlusClassifier(dropout_rate=args.dropout_rate,
                                    hidden_dims = args.hidden_dims, 
                                    activation= args.activation).to(args.device)

    train(args, model, mlflow_uri = "http://localhost:8080")
    
if __name__ == "__main__":
    launch()