import torch
import torch.nn as nn
import time
import numpy as np
import polynomial_cuda  # This assumes you've built the extension using setup.py

class CUDAPolynomialActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return polynomial_cuda.polynomial_activation(x)

    @staticmethod
    def backward(ctx, grad_output):
        # Implement backward pass if needed
        raise NotImplementedError("Backward pass not implemented")

class PolynomialActivation(nn.Module):
    def __init__(self, implementation='pytorch'):
        super().__init__()
        self.implementation = implementation

    def forward(self, x):
        if self.implementation == 'pytorch':
            return x**2 + x + 1
        elif self.implementation == 'cuda':
            return CUDAPolynomialActivation.apply(x)
        else:
            raise ValueError(f"Unknown implementation: {self.implementation}")

# Benchmark function
def benchmark(func, x, name, num_runs=1000):
    start_time = time.time()
    for _ in range(num_runs):
        func(x)
    torch.cuda.synchronize()
    end_time = time.time()
    # return f"{name}: {(end_time - start_time) / num_runs * 1000:.4f} ms"
    return np.around(((end_time - start_time) / num_runs * 1000), decimals=6)

# Main function to run benchmarks
def main():
    torch.manual_seed(0)
    x = torch.randn(1000000, device='cuda')

    pytorch_activation = PolynomialActivation(implementation='pytorch').cuda()
    cuda_activation = PolynomialActivation(implementation='cuda').cuda()

    out = cuda_activation.forward(x)
    print(out)

    pytorch_time = benchmark(pytorch_activation, x, "PyTorch built-in")
    cuda_time = benchmark(cuda_activation, x, "CUDA extension")

    print(f"PyTorch built-in execution time: {pytorch_time}")
    print(f"CUDA extension execution time: {cuda_time}")
    print(f"Speep-up from built-in PyTorch to CUDA extension: {np.around((pytorch_time/cuda_time), decimals=5)}")
    
if __name__ == "__main__":
    main()