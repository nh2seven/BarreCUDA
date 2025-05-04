#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void gridInfo(int *a, int *b, int *c)
{
    // Calculate the global thread ID
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    c[i] = a[i] + b[i];

    // Print thread and block information
    // printf("Block Size: %d, Grid Size: %d\n", blockDim.x, gridDim.x);
    printf("Thread ID: %d, Block ID: %d, Operation: %d + %d = %d\n", i, blockIdx.x, a[i], b[i], c[i]);
}

__managed__ int vector_a[256], vector_b[256], vector_c[256];

// Hard-coded core counts per SM, not available from CUDA API; 
int getCudaCoresPerSM(int major, int minor)
{
    switch (major)
    {
    case 2:
        return 32; // Fermi
    case 3:
        return 192; // Kepler
    case 5:
        return 128; // Maxwell
    case 6:
        return minor == 0 ? 64 : 128; // Pascal
    case 7:
        return (minor == 0 || minor == 5) ? 64 : 128; // Volta/Turing
    case 8:
        return minor == 0 ? 64 : 128; // Ampere/Ada Lovelace
    case 9:
        return 128; // Hopper/newer
    default:
        return 0; // Unknown
    }
}

int main()
{
    int deviceCount;
    cudaDeviceProp deviceProp;

    // Kernel launch parameters
    int numBlocks = 2;
    int numThreads = 4;
    int num = 16;

    // Get count of available CUDA-capable devices
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0)
    {
        cout << "No CUDA-capable devices found." << endl;
        return 1;
    }

    // Get properties of each device
    for (int i = 0; i < deviceCount; i++)
    {
        cudaGetDeviceProperties(&deviceProp, i);
        int coresPerSM = getCudaCoresPerSM(deviceProp.major, deviceProp.minor);

        // Device properties
        cout << "\n----------------------------------------" << "\n";
        cout << "Device Properties" << "\n";
        cout << "├── Device: " << i << "\n";
        cout << "├── Name: " << deviceProp.name << "\n";
        cout << "├── Compute Capability: " << deviceProp.major << "." << deviceProp.minor << "\n";
        cout << "├── CUDA Cores: " << (coresPerSM * deviceProp.multiProcessorCount) << " (verify this manually)" << "\n";
        cout << "├── Clock Rate: " << deviceProp.clockRate / 1000 << " MHz\n";
        cout << "├── Total number of SMs: " << deviceProp.multiProcessorCount << "\n";
        cout << "└── Max Threads Per SM: " << deviceProp.maxThreadsPerMultiProcessor << "\n";

        // Memory properties
        cout << "\n----------------------------------------" << "\n";
        cout << "Memory Properties" << "\n";
        cout << "├── Total Global Memory: " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB\n";
        cout << "├── Total Constant Memory: " << deviceProp.totalConstMem / (1024) << " KB\n";
        cout << "├── Shared Memory per Block: " << deviceProp.sharedMemPerBlock / 1024 << " KB\n";
        cout << "└── Registers per Block: " << deviceProp.regsPerBlock << "\n";

        /// Grid properties
        cout << "\n----------------------------------------" << "\n";
        cout << "Grid Properties" << "\n";
        cout << "├── Max Threads per Block: " << deviceProp.maxThreadsPerBlock << "\n";
        cout << "├── Max Block Dim X: " << deviceProp.maxThreadsDim[0] << "\n";
        cout << "├── Max Block Dim Y: " << deviceProp.maxThreadsDim[1] << "\n";
        cout << "├── Max Block Dim Z: " << deviceProp.maxThreadsDim[2] << "\n";
        cout << "├── Max Grid Dim X: " << deviceProp.maxGridSize[0] << "\n";
        cout << "├── Max Grid Dim Y: " << deviceProp.maxGridSize[1] << "\n";
        cout << "└── Max Grid Dim Z: " << deviceProp.maxGridSize[2] << "\n";

        // Execution example
        cout << "\n----------------------------------------" << "\n";
        cout << "Sample Kernel Execution" << "\n";
        cout << "Launching kernel with " << numBlocks << " blocks and " << numThreads << " threads per block.\n" << endl;
    }

    for (int i = 0; i < num; i++)
    {
        vector_a[i] = i;
        vector_b[i] = num - i;
    }

    gridInfo<<<numBlocks, numThreads>>>(vector_a, vector_b, vector_c);

    cudaDeviceSynchronize();

    int result = 0;
    for (int i = 0; i < num; i++)
        result += vector_c[i];

    cout << "\nResult: " << result << endl;

    return 0;
}