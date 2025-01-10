
#include <stdio.h>

// Kernel CUDA para invertir un arreglo
__global__ void invertirArreglo(int *a, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int swapIdx = n - idx - 1;

    if (idx < n / 2) {
        int temp = a[idx];
        a[idx] = a[swapIdx];
        a[swapIdx] = temp;
    }
}

int main() {
    int n = 10;  // Tamaño del arreglo
    size_t size = n * sizeof(int);

    // Reservar memoria en el host (CPU)
    int *h_a = (int *)malloc(size);

    // Inicializar el arreglo en el host
    for (int i = 0; i < n; i++) {
        h_a[i] = i + 1;
    }

    // Mostrar el arreglo original
    printf("Arreglo original:\n");
    for (int i = 0; i < n; i++) {
        printf("%d ", h_a[i]);
    }
    printf("\n");

    // Reservar memoria en el device (GPU)
    int *d_a;
    cudaMalloc(&d_a, size);

    // Copiar datos desde la CPU a la GPU
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);

    // Configuración de bloques e hilos
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Lanzar el kernel
    invertirArreglo<<<numBlocks, blockSize>>>(d_a, n);

    // Sincronizar la GPU
    cudaDeviceSynchronize();

    // Copiar el resultado desde la GPU al host
    cudaMemcpy(h_a, d_a, size, cudaMemcpyDeviceToHost);

    // Mostrar el arreglo invertido
    printf("Arreglo invertido:\n");
    for (int i = 0; i < n; i++) {
        printf("%d ", h_a[i]);
    }
    printf("\n");

    // Liberar memoria
    free(h_a);
    cudaFree(d_a);

    return 0;
}
