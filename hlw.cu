#include <iostream>

void divide(const float* a, const float* b, float* c, int N) {
    for (int i = 0; i < N; i++) {
        c[i] = a[i] / b[i];
    }
}

__global__ void Divide(float const* a, float const* b, float* c, int N) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        c[i] = a[i] / b[i];
}

int main() {
    // function to add the elements of two arrays

    int N = 1<<30; // 1M elements

    // allocate host memory
    std::cout << "Allocating host memory" << std::endl;
    float *x = new float[N];
    float *y = new float[N];

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // allocate device memory
    std::cout << "Allocating device memory" << std::endl;
    float *dx, *dy, *dsum;
    cudaMalloc(&dx, N * sizeof(float));
    cudaMalloc(&dy, N * sizeof(float));
    cudaMalloc(&dsum, N * sizeof(float));

    // copy host memory to device
    std::cout << "Copying memory to device" << std::endl;
    cudaMemcpy(dx, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, y, N * sizeof(float), cudaMemcpyHostToDevice);

    // Run kernel on 1M elements on the device
    std::cout << "Dispatching" << std::endl;
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock; // round up to 256
    Divide<<<blocks, threadsPerBlock>>>(dx, dy, dsum, N);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }

    // Copy back to host
    std::cout << "Copying memory to host" << std::endl;
    float *sum = new float[N];
    cudaMemcpy(sum, dsum, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Check for errors (all values should be 0.5f)
    std::cout << "Error checks:" << std::endl;
    float totalError = 0.0f;
    for (int i = 0; i < N; i++)
        totalError += sum[i] - 0.5f;
    std::cout << "Total error: " << totalError << std::endl;

    // Free memory
    std::cout << "Freeing memory" << std::endl;
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dsum);
    delete [] x;
    delete [] y;
    delete [] sum;

    // Doing it on host
    std::cout << "Reallocate memory" << std::endl;
    float *x2 = new float[N];
    float *y2 = new float[N];
    float *sum2 = new float[N];
    for (int i = 0; i < N; i++) {
        x2[i] = 1.0f;
        y2[i] = 2.0f;
    }
    std::cout << "Host divide" << std::endl;
    divide(x2, y2, sum2, N);
    std::cout << "Error checks:" << std::endl;
    totalError = 0.0f;
    for (int i = 0; i < N; i++)
        totalError += sum2[i] - 0.5f;
    std::cout << "Total error: " << totalError << std::endl;
    std::cout << "Free" << std::endl;
    delete [] x;
    delete [] y;
    delete [] sum;

    return 0;
}