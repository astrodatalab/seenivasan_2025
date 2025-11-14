import torch
import torch.nn as nn
from modules import ResNet18PlusClassifier
from data_manage import HDF5ImageGenerator
import os
from photoz_utils import *
import pandas as pd
from torch.utils.data import DataLoader
from utils import *
from memory import set_gpu_memory_limit
import argparse

def load_model(model, model_checkpoint_path, device):
    model = torch.load(model_checkpoint_path)
    return model

def make_predictions(model, testing_data_path, ground_truth_redshift, device, batch_size=512, X_key='image'):
    test_dataset = HDF5ImageGenerator(
        testing_data_path,
        X_key=X_key,
        y_key=ground_truth_redshift,
        scaler=False,
        labels_encoding=False,
        num_classes=None,
        shuffle=False,
        mode='test'
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    ground_truth, predictions = [], []
    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(device)
            ground_truth.extend(labels.cpu().numpy())
            pred = model(images)
            pred = pred.squeeze(1)
            predictions.extend(pred.cpu().numpy())
    
    return np.array(ground_truth), np.array(predictions)

def save_results_to_csv(model_name, ground_truth, predictions, filename):
    df = pd.DataFrame({
        "Ground Truth": ground_truth,
        "Predictions": predictions
    })
    save_dir = os.path.join("/data2/logs/", model_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    # Ensure `save_path` is not mistakenly a directory
    if os.path.isdir(save_path):
        raise ValueError(f"Error: {save_path} is a directory, expected a file path.")
    df.to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")

#include your model names and checkpoint paths (.pth files) as tuples here in the models list:
models = [
           ('TransferZ_Baseline', '/data2/models/mlflow/mlruns/206/075bf4ba0ff34d5d88a93022b5474e45/artifacts/best_ema_model/data/model.pth'),
           ('GalaxiesML_Baseline', '/data2/models/mlflow/mlruns/205/fac31b21d2f4486dbd99d9b8570cd43d/artifacts/best_ema_model/data/model.pth'),
           ('Combo_Model', '/data2/models/mlflow/mlruns/201/56661abc8b8c406a873427202e4323d9/artifacts/best_ema_model/data/model.pth'),
           ('TZ_to_GM_LoRA_Full', '/data2/models/mlflow/mlruns/211/e6084baacbde4e62964686087642db2d/artifacts/best_ema_model/data/model.pth'),
           ('GM_to_TZ_LoRA_Full', '/data2/models/mlflow/mlruns/215/62958f34f0874feeaea61699c8b3c126/artifacts/best_ema_model/data/model.pth'),
           ('TZ_to_GM_Standard_TL', '/data2/models/mlflow/mlruns/216/4d84d37d072344d18e6907af9e346dc1/artifacts/best_ema_model/data/model.pth'),
           ('GM_to_TZ_Standard_TL', '/data2/models/mlflow/mlruns/217/df654070acb648aa9c8ba017e8aa9fec/artifacts/best_ema_model/data/model.pth'),
           ('TZbase_GMLoraFull_frac_10', '/data2/models/mlflow/mlruns/222/e7f1b4ce0d9f414d97f019e1f71e8f2d/artifacts/best_ema_model/data/model.pth'),
           ('TZbase_GMLoraFull_frac_20', '/data2/models/mlflow/mlruns/222/190322db7684406486cef98d7d9c1983/artifacts/best_ema_model/data/model.pth'),
           ('TZbase_GMLoraFull_frac_30', '/data2/models/mlflow/mlruns/222/cddfea69f9744d5181f9d6a8dca90336/artifacts/best_ema_model/data/model.pth'),
           ('TZbase_GMLoraFull_frac_40', '/data2/models/mlflow/mlruns/222/13ad8fd3bb314f82bd39a9580b4ec8bb/artifacts/best_ema_model/data/model.pth'),
           ('TZbase_GMLoraFull_frac_50', '/data2/models/mlflow/mlruns/222/e189d457879a453d87e5f538bb5d6209/artifacts/best_ema_model/data/model.pth'),
           ('TZbase_GMLoraFull_frac_60', '/data2/models/mlflow/mlruns/222/2bfd6530f49644e1a246f454590ab3ef/artifacts/best_ema_model/data/model.pth'),
           ('TZbase_GMLoraFull_frac_70', '/data2/models/mlflow/mlruns/222/db983339a191483fa1c851ea6d331c22/artifacts/best_ema_model/data/model.pth'),
           ('TZbase_GMLoraFull_frac_80', '/data2/models/mlflow/mlruns/222/48b0f56ffe4f4d88b56a6dd07c9c5bee/artifacts/best_ema_model/data/model.pth'),
           ('TZbase_GMLoraFull_frac_90', '/data2/models/mlflow/mlruns/222/0cc4d79d813f49c29b3d18c41676737e/artifacts/best_ema_model/data/model.pth'),
           ('TZbase_GMLoraFull_frac_100', '/data2/models/mlflow/mlruns/222/8845443acac54d54813c5c01b4fa0891/artifacts/best_ema_model/data/model.pth')
          ]
if __name__ == "__main__":
    galaxiesml_testing_path =  f'/data/HSC/HSC_v6/step2A/64x64/5x64x64_testing.hdf5'
    transferz_testing_path = f'/data3/HSC/HSC_COSMOS/v7/transferz_5x64x64_TESTING.h5'
    combo_testing_path = f'/data3/HSC/HSC_COSMOS/v7/combo_5x64x64_TESTING.h5'
    TZ_GT_z = 'lp_zPDF_CLASSIC'
    GM_GT_z = 'specz_redshift'
    Combo_GT_z = 'z_truth'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for model_name, model_path in models:
        print(f"Loading model: {model_name} from {model_path}")
        model = ResNet18PlusClassifier(dropout_rate=0.5, hidden_dims=[512, 256], activation='silu').to(device)
        model = load_model(model, model_path, device)
        print(f"Model {model_name} loaded successfully.")
        
        # Make predictions
        combo_GT, combo_pred = make_predictions(model, combo_testing_path, Combo_GT_z, device)
        TZ_GT, TZ_pred = make_predictions(model, transferz_testing_path, TZ_GT_z, device)
        GM_GT, GM_pred = make_predictions(model, galaxiesml_testing_path, GM_GT_z, device, X_key = 'image')
        
        # Save results to CSV
        save_results_to_csv(model_name, combo_GT, combo_pred, 'combo_predictions.csv')
        save_results_to_csv(model_name, TZ_GT, TZ_pred , 'transferz_predictions.csv')
        save_results_to_csv(model_name, GM_GT, GM_pred, 'galaxiesml_predictions.csv')

