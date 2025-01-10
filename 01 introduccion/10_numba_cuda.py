from numba import cuda
import numpy as np
import time

# Tamaño de la matriz
N = 1024

# 🚫 Función que accede a la memoria global de forma no coalescente
@cuda.jit
def acceso_no_coalescente(matriz, resultado):
    fila, columna = cuda.grid(2)
    if fila < matriz.shape[0] and columna < matriz.shape[1]:
        resultado[fila] += matriz[fila, columna]

# ✅ Función que accede a la memoria global de forma coalescente
@cuda.jit
def acceso_coalescente(matriz, resultado):
    fila, columna = cuda.grid(2)
    if fila < matriz.shape[0] and columna < matriz.shape[1]:
        resultado[columna] += matriz[fila, columna]

# Generar la matriz aleatoria
matriz = np.random.randint(0, 100, (N, N)).astype(np.int32)
resultado_no_coalescente = np.zeros(N, dtype=np.int32)
resultado_coalescente = np.zeros(N, dtype=np.int32)

# Configuración de hilos y bloques
bloques = (32, 32)
hilos = (32, 32)

# 🚀 Acceso no coalescente
inicio = time.time()
acceso_no_coalescente[bloques, hilos](matriz, resultado_no_coalescente)
fin = time.time()
print(f"Acceso no coalescente completado. Tiempo: {fin - inicio:.6f} segundos")

# 🚀 Acceso coalescente
inicio = time.time()
acceso_coalescente[bloques, hilos](matriz, resultado_coalescente)
fin = time.time()
print(f"Acceso coalescente completado. Tiempo: {fin - inicio:.6f} segundos")
