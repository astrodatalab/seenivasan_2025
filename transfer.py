from peft import get_peft_model, LoraConfig, TaskType
import torch
import torch.nn as nn
import argparse
import loralib as lora


"""
LORA UTILITIES
"""
def print_model(model):
    """
    Custom function to print the model layers. If a LoRA layer is detected, it will be labeled as 'LoRA'.
    It checks both LoRA Linear and LoRA Conv2d layers and prints their details.
    """
    for name, module in model.named_modules():
        # If it's a LoRA Linear layer
        if isinstance(module, lora.Linear):  
            print(f"{name} - LoRA Linear Layer: {module}")
        
        # If it's a LoRA Conv2d layer
        elif isinstance(module, lora.Conv2d):
            print(f"{name} - LoRA Conv2d Layer: {module}")
        
        # Print regular layers as well
        else:
            print(f"{name} - {module}")

def set_up_for_lora(model, args):   
    """
    Set up the model for LoRA by applying LoRA configurations to specified layers: 
    the ResNet feature extractor, the classifier layers, or both. 
    """
    # Freeze model parameters 
    print("Setting up model for LoRA...")
    for param in model.parameters():
        param.requires_grad = False

    # Load from checkpoint if provided
    if hasattr(args, 'checkpoint') and args.checkpoint:
        try:
            model = torch.load(args.checkpoint)
        except Exception as e:
            print(f"Warning: unable to load checkpoint {args.checkpoint}: {e}")
   
    # LoRA configuration
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=[],
        bias="none",
    )
    if args.lora_type == 'full':
        # Apply LoRA to all Linear and Conv2d layers in the model
        lora_config.target_modules = [
            # ResNet conv layers
            "features.0",    
            "features.4.0.conv1", "features.4.0.conv2", "features.4.1.conv1", "features.4.1.conv2",
            "features.5.0.conv1", "features.5.0.conv2", "features.5.1.conv1", "features.5.1.conv2",
            "features.6.0.conv1", "features.6.0.conv2", "features.6.1.conv1", "features.6.1.conv2",
            "features.7.0.conv1", "features.7.0.conv2", "features.7.1.conv1", "features.7.1.conv2",
            # Downsample layers
            "features.5.0.downsample.0",
            "features.6.0.downsample.0",
            "features.7.0.downsample.0",
            # Classifier Linear layers
            "classifier.0", "classifier.3", "classifier.6"
        ]
    elif args.lora_type == 'classifier':
        lora_config.target_modules = [
            "classifier.0",  # Linear(512, 512)
            "classifier.3",  # Linear(512, 256)
            "classifier.6",  # Linear(256, 1)
        ]  # Classifier only list

    elif args.lora_type == 'resnet':
        lora_config.target_modules = [
            "features.0",    
            "features.4.0.conv1", "features.4.0.conv2", "features.4.1.conv1", "features.4.1.conv2",
            "features.5.0.conv1", "features.5.0.conv2", "features.5.1.conv1", "features.5.1.conv2",
            "features.6.0.conv1", "features.6.0.conv2", "features.6.1.conv1", "features.6.1.conv2",
            "features.7.0.conv1", "features.7.0.conv2", "features.7.1.conv1", "features.7.1.conv2",
            # Downnsample layers
            "features.5.0.downsample.0",
            "features.6.0.downsample.0",
            "features.7.0.downsample.0",
        ] # ResNet only list

    # Wrap model with PEFT LoRA  
    model = get_peft_model(model, lora_config)
    print_model(model)
    model.print_trainable_parameters()
    print("LoRA setup complete.")
    return model

def set_up_for_traditional_transfer(model, args):
    """
    Set up the model for traditional transfer learning:
    - Freeze all layers.
    - Load weights from a checkpoint.
    - Unfreeze specific layers defined in args.unfreeze_layers.
    """
    if(args.transfer_learn_mode =='full_fine_tuning'):
        print("Not freezing model weights for full fine-tuning transfer learning...")
        return model
    
    print("Freezing all layers...")
    for param in model.parameters():
        param.requires_grad = False

    # Load weights from checkpoint
    try:
        if hasattr(args, "checkpoint") and args.checkpoint:
            print(f"Loading weights from checkpoint: {args.checkpoint}")
            model = torch.load(args.checkpoint)
        else:
            raise ValueError("Checkpoint path is required for traditional transfer learning.")
    except Exception as e:
        print(f"Error: Unable to load checkpoint {args.checkpoint}: {e}")
    
    # ResNet conv layers
    "features.0",    
    "features.4.0.conv1", "features.4.0.conv2", "features.4.1.conv1", "features.4.1.conv2",
    "features.5.0.conv1", "features.5.0.conv2", "features.5.1.conv1", "features.5.1.conv2",
    "features.6.0.conv1", "features.6.0.conv2", "features.6.1.conv1", "features.6.1.conv2",
    "features.7.0.conv1", "features.7.0.conv2", "features.7.1.conv1", "features.7.1.conv2",
    # Classifier Linear layers
    "classifier.0",  # Linear(512, 512)
    "classifier.3",  # Linear(512, 256)
    "classifier.6",  # Linear(256, 1)

    # Unfreeze specific layers
    if hasattr(args, "unfreeze_layers"):
        print(f"Unfreezing layers: {args.unfreeze_layers}")
        for name, param in model.named_parameters():
            if any(unfreeze_key in name for unfreeze_key in args.unfreeze_layers):
                param.requires_grad = True
                print(f"Unfreezing {name}")
    print_model(model)
    print("Traditional transfer learning setup complete.")                
    return model


