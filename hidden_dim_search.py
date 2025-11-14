import mlflow
import torch
import copy
from argparse import Namespace
from utils import train  

"""
Runs a search across various hidden_dimension configurations for the ResnetClassifier Model,
reporting the best model as that with the lowest EMA val loss. 
"""
# Define the configurations to test
hidden_dim_configs = [
    [512],
    [512, 256],
    [512, 256, 128],
    [512, 256, 128, 64],
    [256],
    [256, 128],
    [256, 128, 64],
    [256, 128, 64, 32],
    [256, 128, 64, 32, 16],
    [1024, 512, 256],
    [1024, 512, 256, 128],
]

# Base args 
base_args = Namespace(
    batch_size=512,
    lr=1e-3,
    epochs=80,
    scheduler=True,
    checkpoint=None,
    run_name="hidden_dim_search",  # Will be overridden for child runs
    image_size=64,
    model_description="Searching hidden_dims",
    transfer_learn=False,
    full_lora=False,
    activation='silu',
    dropout_rate=0.4,
    dataset=6,  
    device="cuda" if torch.cuda.is_available() else "cpu", 
    nested = True
)

# Setup MLflow
mlflow.set_tracking_uri("http://localhost:8080")
mlflow.set_experiment("hidden_dim_search")

best_config = None
best_loss = float("inf")

with mlflow.start_run(run_name="hidden_dim_search", nested = True) as parent_run:
    parent_run_id = parent_run.info.run_id

    for dims in hidden_dim_configs:
        # Deep copy args for isolation
        args = copy.deepcopy(base_args)
        args.hidden_dims = dims
        args.run_name = f"hidden_dim_{'_'.join(map(str, dims))}"
        args.nested = True

        print(f"\n🔍 Running with hidden_dims = {dims}")

        # Run training
        ema_val_loss = train(args)

        print(f"✅ {args.run_name} finished with EMA val loss: {ema_val_loss:.4f}")
        mlflow.log_metric("ema_val_loss", ema_val_loss)

        if ema_val_loss < best_loss:
            best_loss = ema_val_loss
            best_config = dims

    print("\n🏆 Best configuration:")
    print(f"hidden_dims = {best_config}, EMA val loss = {best_loss:.4f}")
