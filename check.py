import torch

print("=" * 60)
print("GPU Availability Check")
print("=" * 60)

# Check if CUDA (NVIDIA GPU) is available
cuda_available = torch.cuda.is_available()
print(f"\n✓ CUDA Available: {cuda_available}")

if cuda_available:
    # Get GPU details
    gpu_count = torch.cuda.device_count()
    print(f"✓ Number of GPUs: {gpu_count}")
    
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        print(f"  GPU {i}: {gpu_name}")
        
        # GPU memory info
        total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f"  Memory: {total_memory:.2f} GB")
    
    # Current device
    current_device = torch.cuda.current_device()
    print(f"\n✓ Current Device: GPU {current_device}")
    
else:
    print("\n⚠️  No GPU detected - will use CPU")
    print("\nTo use GPU, you need:")
    print("  1. NVIDIA GPU")
    print("  2. CUDA-enabled PyTorch")
    print("  3. CUDA drivers installed")

# Test tensor creation
print("\n" + "=" * 60)
print("Testing Device Assignment")
print("=" * 60)

# Create tensor on CPU
cpu_tensor = torch.randn(3, 3)
print(f"\nCPU Tensor device: {cpu_tensor.device}")

if cuda_available:
    # Create tensor on GPU
    gpu_tensor = torch.randn(3, 3).cuda()
    print(f"GPU Tensor device: {gpu_tensor.device}")
    
    # Alternative way
    gpu_tensor2 = torch.randn(3, 3).to('cuda')
    print(f"GPU Tensor device (alternative): {gpu_tensor2.device}")

print("\n" + "=" * 60)