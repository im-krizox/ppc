import numpy as np
from numba import cuda
import time

# 🚀 Función para preprocesar los datos en la CPU
def preprocesar_datos(datos):
    return datos * 2

# 🚀 Función para procesar los datos en la GPU
@cuda.jit
def procesar_datos_gpu(entrada, salida):
    idx = cuda.grid(1)
    if idx < entrada.size:
        salida[idx] = entrada[idx] ** 2

# 🚀 Función principal que combina CPU y GPU con bloques optimizados
def pipeline_hibrido(tamaño):
    # Crear datos aleatorios
    datos = np.random.randint(1, 100, tamaño).astype(np.float32)

    # Preprocesar los datos en la CPU
    inicio_cpu = time.time()
    datos_preprocesados = preprocesar_datos(datos)
    fin_cpu = time.time()
    print(f"Tiempo de preprocesamiento en la CPU: {fin_cpu - inicio_cpu:.6f} segundos")

    # Preparar datos para la GPU
    entrada_gpu = cuda.to_device(datos_preprocesados)
    salida_gpu = cuda.device_array_like(datos_preprocesados)

    # Configurar bloques e hilos dinámicamente
    bloques = (tamaño + 255) // 256
    hilos = 256

    # Procesar los datos en la GPU
    inicio_gpu = time.time()
    procesar_datos_gpu[bloques, hilos](entrada_gpu, salida_gpu)
    cuda.synchronize()
    fin_gpu = time.time()
    print(f"Tiempo de procesamiento en la GPU: {fin_gpu - inicio_gpu:.6f} segundos")

    # Copiar los resultados de la GPU a la CPU
    resultado = salida_gpu.copy_to_host()

    print(f"Resultado final: {resultado[:10]}...")

# 🚀 Ejecutar el pipeline híbrido
if __name__ == "__main__":
    tamaño_datos = 10_000_000  # Número de elementos a procesar
    pipeline_hibrido(tamaño_datos)
