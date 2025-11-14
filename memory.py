import torch
import nvidia_smi
	
"""
MEMORY UTILITIES
"""
def set_gpu_memory_limit(memory_limit_gb):
    """
    Set a memory limit for PyTorch GPU usage.
    Args:
        memory_limit_gb (float): Maximum GPU memory to use in gigabytes
    """
    if not torch.cuda.is_available():
        print("CUDA not available - running on CPU")
        return
        
    try:
        # Get total GPU memory
        total_memory_bytes = get_total_gpu_memory()
        total_memory_gb = total_memory_bytes / (1024 ** 3)
        
        # Check if requested memory exceeds available memory
        if memory_limit_gb > total_memory_gb:
            print(f"Warning: Requested {memory_limit_gb} GB exceeds available {total_memory_gb:.1f} GB")
            print(f"Setting to maximum available: {total_memory_gb:.1f} GB")
            memory_limit_gb = total_memory_gb * 0.95  # Use 95% to be safe
        
        # Convert the desired memory limit to a fraction of the total GPU memory
        memory_limit_bytes = memory_limit_gb * 1024 ** 3
        memory_fraction = memory_limit_bytes / total_memory_bytes
        
        # Ensure fraction is between 0 and 1 (extra safety check)
        memory_fraction = min(max(memory_fraction, 0.0), 0.95)  # Cap at 95%
        
        # Set memory limit based on fraction
        torch.cuda.set_per_process_memory_fraction(memory_fraction, 0)
        torch.cuda.empty_cache()
        actual_limit_gb = (memory_fraction * total_memory_bytes) / (1024 ** 3)
        print(f"GPU memory limit set to {actual_limit_gb:.1f} GB ({memory_fraction:.1%} of {total_memory_gb:.1f} GB total)")
    except Exception as e:
        print(f"Error setting GPU memory limit: {e}")
        # Fallback: try to get GPU info without nvidia_smi
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            total_gb = total_memory / (1024 ** 3)
            print(f"GPU total memory (fallback): {total_gb:.1f} GB")

def get_total_gpu_memory():
    """Get total memory of default GPU in bytes."""
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    return info.total

def print_gpu_utilization():
    """Print current GPU memory usage."""
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")
    
