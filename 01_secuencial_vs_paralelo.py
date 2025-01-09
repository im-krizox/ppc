import threading
import time
import sys

# Función que suma los números de una lista de manera secuencial
def suma_secuencial(lista):
    total = 0.0
    for numero in lista:
        total += numero
    return total

# Función que suma los números de una lista en un rango específico
def suma_paralela(lista, inicio, fin, resultado, indice):
    total = sum(lista[inicio:fin])
    resultado[indice] = total

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Uso: python 01_secuencial_vs_paralelo.py <tamaño_lista>")
        sys.exit(1)

    # Tamaño de la lista desde el argumento posicional
    tamaño_lista = int(sys.argv[1])
    lista = [float(i) for i in range(tamaño_lista)]

    # 🚀 Ejecución secuencial
    inicio_secuencial = time.time()
    resultado_secuencial = suma_secuencial(lista)
    fin_secuencial = time.time()
    print(f"Suma secuencial: {resultado_secuencial}")
    print(f"Tiempo secuencial: {fin_secuencial - inicio_secuencial:.5f} segundos\n")

    # 🚀 Ejecución paralela
    inicio_paralela = time.time()

    # Dividimos la lista en 4 partes y creamos 4 hilos
    num_hilos = 4
    longitud_parte = len(lista) // num_hilos
    resultado = [0.0] * num_hilos
    hilos = []

    for i in range(num_hilos):
        inicio = i * longitud_parte
        fin = (i + 1) * longitud_parte if i != num_hilos - 1 else len(lista)
        hilo = threading.Thread(target=suma_paralela, args=(lista, inicio, fin, resultado, i))
        hilos.append(hilo)
        hilo.start()

    # Esperamos que todos los hilos terminen
    for hilo in hilos:
        hilo.join()

    # Sumamos los resultados parciales
    resultado_paralelo = sum(resultado)
    fin_paralela = time.time()

    print(f"Suma paralela: {resultado_paralelo}")
    print(f"Tiempo paralelo: {fin_paralela - inicio_paralela:.5f} segundos\n")

    # 🔍 Comparación final
    print("Comparación de tiempos:")
    print(f"Tiempo secuencial: {fin_secuencial - inicio_secuencial:.5f} segundos")
    print(f"Tiempo paralelo: {fin_paralela - inicio_paralela:.5f} segundos")
