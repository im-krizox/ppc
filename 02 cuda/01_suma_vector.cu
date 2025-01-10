#include <stdio.h>

// Kernel CUDA para sumar dos vectores
__global__ void sumarVectores(int *a, int *b, int *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    int n = 1000;  // Tamaño del vector
    size_t size = n * sizeof(int);

    // Reservar memoria en el host (CPU)
    int *h_a = (int *)malloc(size);
    int *h_b = (int *)malloc(size);
    int *h_c = (int *)malloc(size);

    // Inicializar los vectores en el host
    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // Reservar memoria en el device (GPU)
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copiar datos desde el host a la GPU
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Configuración de bloques e hilos
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Lanzar el kernel
    sumarVectores<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);

    // Sincronizar la GPU
    cudaDeviceSynchronize();

    // Copiar el resultado desde la GPU al host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Imprimir los primeros 10 resultados
    for (int i = 0; i < 10; i++) {
        printf("c[%d] = %d\n", i, h_c[i]);
    }

    // Liberar memoria
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
