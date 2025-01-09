import threading
import argparse
import time

# Variable compartida
contador = 0

# Función que incrementa el contador
def incrementar_contador(num_incrementos):
    global contador
    for _ in range(num_incrementos):
        valor_actual = contador
        time.sleep(0.00001)  # Introducir un retraso
        contador = valor_actual + 1

# Configuración de argumentos posicionales
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulación de condición de carrera sin sincronización.")
    parser.add_argument("num_incrementos", type=int, help="Número de veces que cada hilo incrementará el contador.")

    # Obtener los argumentos
    args = parser.parse_args()
    num_incrementos = args.num_incrementos

    # Crear dos hilos que incrementan el contador simultáneamente
    hilo1 = threading.Thread(target=incrementar_contador, args=(num_incrementos,))
    hilo2 = threading.Thread(target=incrementar_contador, args=(num_incrementos,))

    # Iniciar los hilos
    hilo1.start()
    hilo2.start()

    # Esperar a que ambos hilos terminen
    hilo1.join()
    hilo2.join()

    # Mostrar el resultado final del contador
    print(f"Resultado esperado: {num_incrementos * 2}")
    print(f"Resultado real del contador: {contador}")
