import os
import torch
import torchvision
from PIL import Image
import imageio
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import nvidia_smi
from data_manage import HDF5ImageGenerator
from modules import ResNet18PlusClassifier
import torchvision.models as models
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import logging
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
import mlflow
import mlflow.pytorch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from transfer import set_up_for_lora, set_up_for_traditional_transfer, print_model
import torch.nn.functional as F
import builtins

GLOBAL_SEED = 42

"""
LOSS FUNCTION
"""
def calculate_loss(z_spec, z_photo):
    """
    HSC METRIC. Returns a tensor. Loss is accuracy metric defined by HSC, meant
    to capture the effects of bias, scatter, and outlier all in one.
    z_photo: tensor
        Photometric or predicted redshifts.
    z_spec: tensor
        Spectroscopic or actual redshifts.
    """
    dz = (z_photo - z_spec)/(1 + z_spec)  # Difference between predicted and actual redshift scaled by 1 + ground truth redshift
    gamma = 0.15
    denominator = 1.0 + torch.square(dz / gamma)  # Equivalent to K.square
    L = 1 - 1.0 / denominator  # Equivalent to the custom loss formula
    return torch.mean(L)  # Return the mean loss over the batch

import torch, numpy as np, random

def set_seed(seed):
    """
    Set global seed for reproducibility of models. 
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

"""
TRAINING UTILITIES
"""
def get_dataset_name(dataset_number):
    dataset_names = {
        1: "GalaxiesML",
        2: "TransferZ", 
        3: "Combo"
    }
    return dataset_names[dataset_number]

def create_datasets(args):
    " GALAXIESML DATA PATHS "
    # " GALAXIESML DATA PATHS "
    # galaxiesml_training_path = f'/your_path/5x{args.image_size}x{args.image_size}_training.hdf5'
    # galaxiesml_validation_path =  f'/your_path/{args.image_size}x{args.image_size}/5x{args.image_size}x{args.image_size}_validation.hdf5'
    # galaxiesml_testing_path =  f'/your_path/{args.image_size}x{args.image_size}/5x{args.image_size}x{args.image_size}_testing.hdf5'

    # " TRANSFERZ DATA PATHS "
    # transferz_training_path = f'/your_path/transferz_5x{args.image_size}x{args.image_size}_TRAINING.h5'
    # transferz_validation_path = f'/your_path/transferz_5x{args.image_size}x{args.image_size}_VALIDATION.h5'
    # transferz_testing_path = f'/your_path/transferz_5x{args.image_size}x{args.image_size}_TESTING.h5'

    # " COMBO DATA PATHS "
    # combo_training_path = f'/your_path/combo_5x{args.image_size}x{args.image_size}_TRAINING.h5'
    # combo_validation_path = f'/your_path/combo_5x{args.image_size}x{args.image_size}_VALIDATION.h5'
    # combo_testing_path = f'/your_path/combo_5x{args.image_size}x{args.image_size}_TESTING.h5'
    " GALAXIESML DATA PATHS "
    galaxiesml_training_path = f'/data/HSC/HSC_v6/step2A/{args.image_size}x{args.image_size}/5x{args.image_size}x{args.image_size}_training.hdf5'
    galaxiesml_validation_path =  f'/data/HSC/HSC_v6/step2A/{args.image_size}x{args.image_size}/5x{args.image_size}x{args.image_size}_validation.hdf5'
    galaxiesml_testing_path =  f'/data/HSC/HSC_v6/step2A/{args.image_size}x{args.image_size}/5x{args.image_size}x{args.image_size}_testing.hdf5'

    " TRANSFERZ DATA PATHS "
    transferz_training_path = f'/data3/HSC/HSC_COSMOS/v7/transferz_5x{args.image_size}x{args.image_size}_TRAINING.h5'
    transferz_validation_path = f'/data3/HSC/HSC_COSMOS/v7/transferz_5x{args.image_size}x{args.image_size}_VALIDATION.h5'
    transferz_testing_path = f'/data3/HSC/HSC_COSMOS/v7/transferz_5x{args.image_size}x{args.image_size}_TESTING.h5'

    " COMBO DATA PATHS "
    combo_training_path = f'/data3/HSC/HSC_COSMOS/v7/combo_5x{args.image_size}x{args.image_size}_TRAINING.h5'
    combo_validation_path = f'/data3/HSC/HSC_COSMOS/v7/combo_5x{args.image_size}x{args.image_size}_VALIDATION.h5'
    combo_testing_path = f'/data3/HSC/HSC_COSMOS/v7/combo_5x{args.image_size}x{args.image_size}_TESTING.h5'


    dataset_paths = {
        1: (galaxiesml_training_path, galaxiesml_validation_path, "specz_redshift"),
        2: (transferz_training_path, transferz_validation_path, "lp_zPDF_CLASSIC"),
        3: (combo_training_path, combo_validation_path, "z_truth")
    }
    try:
        training_dataset_path, validation_dataset_path, ground_truth = dataset_paths[args.dataset]
    except KeyError:
        raise ValueError("Please enter 1, 2, or 3 for dataset.")
    train_dataset = HDF5ImageGenerator(
        training_dataset_path,
        X_key = 'image',
        y_key= ground_truth,
        scaler=False,
        labels_encoding=False,
        num_classes=None,
        shuffle=True,
        mode='train'
    )
    val_dataset = HDF5ImageGenerator(
        validation_dataset_path,
        X_key = 'image',
        y_key=ground_truth,
        scaler=False,
        labels_encoding=False,
        num_classes=None,
        shuffle=False,
        mode='validation'
    )
    return train_dataset, val_dataset

def subsample_dataset(dataset, fraction, seed=GLOBAL_SEED):
    """
    Subsample a fraction of the given dataset. 
    """
    if fraction >= 1.0:
        return dataset
    subset_size = int(len(dataset) * fraction)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=subset_size, replace=False)
    return torch.utils.data.Subset(dataset, indices)
 
def get_dataloaders(args):
    train_dataset, val_dataset = create_datasets(args)
    train_dataset = subsample_dataset(train_dataset, args.data_fraction, seed=GLOBAL_SEED)
    val_dataset = subsample_dataset(val_dataset, args.data_fraction, seed=GLOBAL_SEED + 1)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader
    
def setup_optimizer_and_scheduler(model, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    print("Initializing optimizer...")
    if args.scheduler:
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True) #Scheduler parameters are taken from ResNet paper
        print("Initializing scheduler...")
    else:
        scheduler = None
    return optimizer, scheduler
    
def load_checkpoint_if_available(model, args):
    best_val_loss = float('inf')
    if args.checkpoint:
        model = torch.load(args.checkpoint)
    print(f"Loading model from checkpoint: {args.checkpoint}")
    return 0, float('inf')
    
def train_one_epoch(model, loader, optimizer, args, epoch): 
    model.train()
    total_loss = 0
    criterion = calculate_loss
    device = args.device
    for i, (images, labels) in enumerate(tqdm(loader)):
        images, labels = images.to(device), torch.clamp(labels.to(device), 0, 4)
        optimizer.zero_grad()
        outputs = model(images)
        outputs = outputs.squeeze(1)
        loss = criterion(labels, outputs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    logging.info(f"Epoch {epoch+1} - Average Training Loss: {avg_loss:.4f}")
    return avg_loss
    
def validate(model, loader, args, epoch):
    model.eval()
    total_loss = 0
    criterion = calculate_loss
    device = args.device
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = outputs.squeeze(1)
            loss = criterion(labels, outputs)
            total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    logging.info(f"Epoch {epoch+1} - Average Validation Loss: {avg_loss:.4f}")
    return avg_loss

def _train_model_core(args, model, example_input=None, log_to_mlflow=True, return_ema=False):
    set_seed(GLOBAL_SEED)
    model = model.to(args.device)
    train_loader, val_loader = get_dataloaders(args)
    optimizer, scheduler = setup_optimizer_and_scheduler(model, args)
    start_epoch = 0
    best_val_loss = float('inf')
    train_losses, val_losses, ema_val_losses = [], [], []
    best_ema_val_loss = float('inf')
    best_ema_epoch = -1
    best_val_epoch = -1
    patience_counter = 0
    patience = 20
    alpha = 2/11 #to look over past 10 epochs for EMA

    for epoch in range(start_epoch, start_epoch + args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, args, epoch)
        val_loss = validate(model, val_loader, args, epoch)

        if args.scheduler:
            scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # EMA tracking
        if len(ema_val_losses) == 0:
            ema_val_loss = val_loss
        else:
            ema_val_loss = alpha * ema_val_losses[-1] + (1 - alpha) * val_loss
        ema_val_losses.append(ema_val_loss)

        # Logging
        if log_to_mlflow:
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("ema_val_loss", ema_val_loss, step=epoch)
            mlflow.log_metric("learning_rate", current_lr, step=epoch)

        # Save best raw model
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_val_epoch = epoch
            if log_to_mlflow:
                mlflow.pytorch.log_model(
                    model,
                    artifact_path="best_model",
                    registered_model_name=args.run_name if hasattr(args, "run_name") else None,
                )

        # Save best EMA model
        is_best_ema = ema_val_loss < best_ema_val_loss
        if is_best_ema:
            best_ema_val_loss = ema_val_loss
            best_ema_epoch = epoch
            patience_counter = 0
            if log_to_mlflow:
                mlflow.pytorch.log_model(
                    model,
                    artifact_path="best_ema_model",
                    registered_model_name=args.run_name if hasattr(args, "run_name") else None,
                )
        else:
            patience_counter += 1

        # Periodic checkpoint
        if (epoch + 1) % args.save_frequency == 0 and log_to_mlflow:
            mlflow.pytorch.log_model(model, artifact_path=f"checkpoint_epoch_{epoch+1}")

        if args.early_stopping and patience_counter >= patience:
            print(f"Early stopping at epoch {epoch} (patience={patience})")
            break

    if log_to_mlflow:
        mlflow.log_metric("num_epochs_trained", len(val_losses))
        mlflow.log_metric("final_raw_val_loss", val_losses[-1])
        mlflow.log_metric("best_val_loss", best_val_loss)
        mlflow.log_metric("best_val_loss_epoch", best_val_epoch)
        mlflow.log_metric("best_ema_val_loss", best_ema_val_loss)
        mlflow.log_metric("best_ema_loss_epoch", best_ema_epoch)

    return best_ema_val_loss if return_ema else best_val_loss

def train(args, model=models.resnet18(weights=None), mlflow_uri="http://localhost:8080"):
    if not args.nested:
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(args.run_name)

    with mlflow.start_run(run_name=args.run_name, nested = args.nested):
        example_input = torch.randn(1, 5, 64, 64).cpu().numpy()
        hidden_dims = args.hidden_dims if hasattr(args, 'hidden_dims') else [512, 256]
        dropout_rate = args.dropout_rate
        activation = args.activation

        mlflow.log_params({ #You can adjust what parameters to log here
            'dataset': get_dataset_name(args.dataset),
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'epochs': args.epochs,
            'scheduler': args.scheduler,
            'checkpoint': args.checkpoint or '',
            'run_name': args.run_name,
            'image_size': args.image_size,
            'model_description': args.model_description,
            'transfer_learn': args.transfer_learn,
            'activation': args.activation,
            'image_size': args.image_size,
            'model_description': args.model_description,
            'dropout_rate': args.dropout_rate,
            'hidden_dims': str(hidden_dims)  # logs like "[512, 256, 128, 64]"
        })
                
        if args.lora_flag:
            model = set_up_for_lora(model, args)

        if args.transfer_learn:
            model = set_up_for_traditional_transfer(model, args)

        return _train_model_core(args, model, example_input=example_input, log_to_mlflow=True, return_ema=True)



