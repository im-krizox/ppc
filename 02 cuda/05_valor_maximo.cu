#include <stdio.h>
#include <limits.h>

// Kernel CUDA para encontrar el valor máximo en un arreglo
__global__ void encontrarMaximo(int *a, int *maximo, int n) {
    __shared__ int maximosParciales[256];  // Memoria compartida para almacenar máximos parciales

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int tid = threadIdx.x;

    // Inicializar la memoria compartida con el valor más bajo posible
    if (idx < n) {
        maximosParciales[tid] = a[idx];
    } else {
        maximosParciales[tid] = INT_MIN;
    }
    __syncthreads();

    // Reducción paralela en memoria compartida
    for (int salto = blockDim.x / 2; salto > 0; salto /= 2) {
        if (tid < salto && idx + salto < n) {
            maximosParciales[tid] = max(maximosParciales[tid], maximosParciales[tid + salto]);
        }
        __syncthreads();
    }

    // Escribir el máximo parcial en el resultado global
    if (tid == 0) {
        atomicMax(maximo, maximosParciales[0]);
    }
}

int main() {
    int n = 1000;  // Tamaño del arreglo
    size_t size = n * sizeof(int);

    // Reservar memoria en el host (CPU)
    int *h_a = (int *)malloc(size);
    int h_maximo = INT_MIN;

    // Inicializar el arreglo en el host con valores aleatorios
    for (int i = 0; i < n; i++) {
        h_a[i] = rand() % 10000;  // Números aleatorios entre 0 y 9999
    }

    // Mostrar una parte del arreglo
    printf("Primeros 10 números del arreglo:\n");
    for (int i = 0; i < 10; i++) {
        printf("%d ", h_a[i]);
    }
    printf("\n");

    // Reservar memoria en el device (GPU)
    int *d_a, *d_maximo;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_maximo, sizeof(int));

    // Copiar datos desde la CPU a la GPU
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_maximo, &h_maximo, sizeof(int), cudaMemcpyHostToDevice);

    // Configuración de bloques e hilos
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    // Lanzar el kernel
    encontrarMaximo<<<numBlocks, blockSize>>>(d_a, d_maximo, n);

    // Sincronizar la GPU
    cudaDeviceSynchronize();

    // Copiar el resultado desde la GPU al host
    cudaMemcpy(&h_maximo, d_maximo, sizeof(int), cudaMemcpyDeviceToHost);

    // Mostrar el resultado
    printf("El valor máximo en el arreglo es: %d\n", h_maximo);

    // Liberar memoria
    free(h_a);
    cudaFree(d_a);
    cudaFree(d_maximo);

    return 0;
}
