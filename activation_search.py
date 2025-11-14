import torch
import argparse
from types import SimpleNamespace
from modules import ResNet18PlusClassifier
from utils import train  

# Define base arguments
def get_base_args():
    return SimpleNamespace(
        run_name=None,  # will be set per activation
        nested = False, 
        dataset=3,
        batch_size=512,
        lr=0.001,
        epochs=300,
        scheduler="plateau",
        checkpoint=None,
        image_size=64,
        model_description="Activation sweep for ResNet18PlusClassifier",
        transfer_learn=False,
        full_lora=False,
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        dropout_rate=0.4,
        hidden_dims=[256, 128, 64],  # used for logging
        activation=None,  # to be set
        data_fraction = 0.25,
        save_frequency = 10,
        lora_flag = False
    )
 #Runs through the three activations and reports best model with lowest EMA val loss. 
def run_all_activations():
    results = {}
    activations = ["relu", "gelu", "silu"]

    for act in activations:
        print(f"\n🔁 Running training with activation = {act}")
        args = get_base_args()
        args.activation = act
        args.run_name = f"activation_test_{act}"

        # Train model and get final EMA loss
        ema_val_loss = train(args)

        results[act] = ema_val_loss
        print(f"✅ {act} finished with best EMA val loss: {ema_val_loss:.4f}")

    # Report best activation
    best_act = min(results, key=results.get)
    print("\n🏆 Best activation:")
    print(f"   🔹 {best_act} with EMA val loss = {results[best_act]:.4f}")

if __name__ == "__main__":
    run_all_activations()
