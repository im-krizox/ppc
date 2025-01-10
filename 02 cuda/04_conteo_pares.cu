#include <stdio.h>

// Kernel CUDA para contar números pares en un arreglo
__global__ void contarPares(int *a, int *contador, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n && a[idx] % 2 == 0) {
        atomicAdd(contador, 1);
    }
}

int main() {
    int n = 1000;  // Tamaño del arreglo
    size_t size = n * sizeof(int);

    // Reservar memoria en el host (CPU)
    int *h_a = (int *)malloc(size);
    int h_contador = 0;

    // Inicializar el arreglo en el host con números secuenciales
    for (int i = 0; i < n; i++) {
        h_a[i] = i + 1;
    }

    // Mostrar una parte del arreglo
    printf("Primeros 10 números del arreglo:\n");
    for (int i = 0; i < 10; i++) {
        printf("%d ", h_a[i]);
    }
    printf("\n");

    // Reservar memoria en el device (GPU)
    int *d_a, *d_contador;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_contador, sizeof(int));

    // Copiar datos desde la CPU a la GPU
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_contador, &h_contador, sizeof(int), cudaMemcpyHostToDevice);

    // Configuración de bloques e hilos
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Lanzar el kernel
    contarPares<<<numBlocks, blockSize>>>(d_a, d_contador, n);

    // Sincronizar la GPU
    cudaDeviceSynchronize();

    // Copiar el resultado desde la GPU al host
    cudaMemcpy(&h_contador, d_contador, sizeof(int), cudaMemcpyDeviceToHost);

    // Mostrar el resultado
    printf("Cantidad de números pares en el arreglo: %d\n", h_contador);

    // Liberar memoria
    free(h_a);
    cudaFree(d_a);
    cudaFree(d_contador);

    return 0;
}
