#include <stdio.h>

// Kernel CUDA para multiplicar cada elemento de un vector por un escalar
__global__ void multiplicarEscalar(int *a, int escalar, int *c, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] * escalar;
    }
}

int main() {
    int n = 1000;  // Tamaño del vector
    int escalar = 5;  // Valor escalar
    size_t size = n * sizeof(int);

    // Reservar memoria en el host (CPU)
    int *h_a = (int *)malloc(size);
    int *h_c = (int *)malloc(size);

    // Inicializar el vector en el host
    for (int i = 0; i < n; i++) {
        h_a[i] = i + 1;  // Valores del 1 al n
    }

    // Reservar memoria en el device (GPU)
    int *d_a, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_c, size);

    // Copiar datos desde la CPU a la GPU
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

    // Configuración de bloques e hilos
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Lanzar el kernel
    multiplicarEscalar<<<numBlocks, blockSize>>>(d_a, escalar, d_c, n);

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
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_c);

    return 0;
}
