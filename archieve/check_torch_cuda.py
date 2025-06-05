import torch


def check_cuda_availability():
    """Checks if CUDA is available for PyTorch."""
    if torch.cuda.is_available():
        print("CUDA is available!")
        print(f"Number of CUDA devices available: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        return True
    else:
        print("CUDA is NOT available. PyTorch will use the CPU.")
        return False

if __name__ == "__main__":
    is_cuda_available = check_cuda_availability()
    print(f"Torch CUDA Available: {is_cuda_available}")