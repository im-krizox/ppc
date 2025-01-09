from numba import cuda
import numpy as np
import time

# Tamaño de la matriz
N = 1024

# Función que suma una matriz utilizando memoria global
@cuda.jit
def suma_global(matriz, resultado):
    fila, columna = cuda.grid(2)
    if fila < matriz.shape[0] and columna < matriz.shape[1]:
        cuda.atomic.add(resultado, 0, matriz[fila, columna])

# Función que suma una matriz utilizando memoria compartida
@cuda.jit
def suma_compartida(matriz, resultado):
    # Crear memoria compartida
    memoria_compartida = cuda.shared.array((N, N), dtype=np.int32)
    fila, columna = cuda.grid(2)

    # Cargar los datos en memoria compartida
    if fila < matriz.shape[0] and columna < matriz.shape[1]:
        memoria_compartida[fila, columna] = matriz[fila, columna]

    cuda.syncthreads()  # Sincronizar los hilos

    # Realizar la suma
    if fila < matriz.shape[0] and columna < matriz.shape[1]:
        cuda.atomic.add(resultado, 0, memoria_compartida[fila, columna])

# Generar la matriz aleatoria
matriz = np.random.randint(0, 100, (N, N)).astype(np.int32)
resultado_global = np.zeros(1, dtype=np.int32)
resultado_compartida = np.zeros(1, dtype=np.int32)

# Configuración de hilos y bloques
bloques = (32, 32)
hilos = (32, 32)

# 🚀 Suma utilizando memoria global
inicio = time.time()
suma_global[bloques, hilos](matriz, resultado_global)
fin = time.time()
print(f"Suma utilizando memoria global: {resultado_global[0]}, Tiempo: {fin - inicio:.6f} segundos")

# 🚀 Suma utilizando memoria compartida
inicio = time.time()
suma_compartida[bloques, hilos](matriz, resultado_compartida)
fin = time.time()
print(f"Suma utilizando memoria compartida: {resultado_compartida[0]}, Tiempo: {fin - inicio:.6f} segundos")
