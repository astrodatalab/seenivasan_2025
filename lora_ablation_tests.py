import time
import csv
import os
import gc
import torch
import torch.cuda as cuda
from datetime import datetime
import argparse
import json
import torch
import mlflow
import copy
from modules import ResNet18PlusClassifier
from utils import train
from memory import set_gpu_memory_limit, print_gpu_utilization

def run_lora_nested_ablation(args, lora_configs):
    best_loss = float("inf")
    best_config = None
    results = []

    mlflow.set_tracking_uri("http://localhost:8080")
    mlflow.set_experiment("lora_ablation")

    with mlflow.start_run(run_name="lora_ablation", nested=False) as parent_run:

        for config in lora_configs:
            # Clone args and attach current LoRA config
            run_args = copy.deepcopy(args)
            run_args.run_name = config["name"]
            run_args.lora_type = config["lora_type"]
            run_args.lora_r = config["r"]
            run_args.lora_alpha = config["alpha"]
            run_args.lora_dropout = config["dropout"]
            run_args.nested = True
            run_args.model_description = f"LoRA {run_args.lora_type} r={run_args.lora_r} alpha={run_args.lora_alpha} dropout={run_args.lora_dropout}"

            # Setup model
            model = ResNet18PlusClassifier(
                dropout_rate=run_args.dropout_rate,
                hidden_dims=run_args.hidden_dims,
                activation=run_args.activation
            ).to(run_args.device)

            if run_args.silu_push_force:
                replace_relu_with_silu(model)

            # Clean up GPU and memory
            gc.collect()
            if torch.cuda.is_available():
                cuda.empty_cache()
                cuda.reset_peak_memory_stats()

            print_gpu_utilization()
            print(f"🚀 Training {run_args.run_name} with r={run_args.lora_r}, alpha={run_args.lora_alpha}, dropout={run_args.lora_dropout}")

            # Time tracking
            start_time = time.time()

            ema_val_loss = train(run_args, model)

            elapsed = time.time() - start_time
            peak_mem_mb = cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0

            print(f"✅ {run_args.run_name} finished with EMA val loss: {ema_val_loss:.4f} | Time: {elapsed:.2f}s | Mem: {peak_mem_mb:.1f}MB")
            mlflow.log_metric("ema_val_loss", ema_val_loss)
            mlflow.log_metric("train_time_sec", elapsed)
            mlflow.log_metric("peak_gpu_mem_mb", peak_mem_mb)

            # Save results for CSV
            results.append({
                "name": config["name"],
                "lora_type": config["lora_type"],
                "r": config["r"],
                "alpha": config["alpha"],
                "dropout": config["dropout"],
                "ema_val_loss": ema_val_loss,
                "train_time_sec": round(elapsed, 2),
                "peak_gpu_mem_mb": round(peak_mem_mb, 1)
            })

            if ema_val_loss < best_loss:
                best_loss = ema_val_loss
                best_config = config

    print("\n🏆 Best LoRA config:")
    print(f"{best_config} with EMA val loss = {best_loss:.4f}")

    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"lora_ablation_results_{timestamp}.csv"
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"\n📄 Results saved to: {csv_path}")


def main(args):
    with open(args.config_file, "r") as f:
        lora_configs = json.load(f)

    set_gpu_memory_limit(args.memory_limit)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    run_lora_nested_ablation(args, lora_configs)

if __name__ == "__main__":
    transferz_baseline = "/data2/models/mlflow/mlruns/206/0d77fc4e5e224017a854c7d399326dfc/artifacts/best_ema_model/data/model.pth"
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="configs/lora_ablation_configs.json")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--dataset", type=int, default=1)
    parser.add_argument("--image_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout_rate", type=float, default=0.5)
    parser.add_argument("--activation", type=str, default="silu")
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[512, 256])
    parser.add_argument("--transfer_learn", action="store_true")
    parser.add_argument("--silu_push_force", action="store_true")
    parser.add_argument("--scheduler", action="store_true")
    parser.add_argument("--spectra_mode", action="store_true")
    parser.add_argument("--memory_limit", type=int, default=30)
    parser.add_argument("--checkpoint", type=str, default=transferz_baseline)
    parser.add_argument("--nested", action="store_true")  # gets overridden internally
    parser.add_argument("--lora_flag", action="store_true")
    parser.add_argument("--data_fraction", type=float, default =0.01)
    parser.add_argument("--save_frequency", type=int, default = 10)
    parser.add_argument("--early_stopping", type=int, default = 10)



    args = parser.parse_args()
    main(args)
