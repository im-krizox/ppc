import threading
import argparse
import time

# Recursos compartidos
recurso_a = threading.Lock()
recurso_b = threading.Lock()

# Función para el primer hilo que adquiere los recursos en orden
def hilo_1(num_iteraciones):
    for _ in range(num_iteraciones):
        with recurso_a:
            print("Hilo 1: Recurso A adquirido")
            time.sleep(0.1)
            with recurso_b:
                print("Hilo 1: Recurso B adquirido")
        print("Hilo 1: Recursos liberados\n")

# Función para el segundo hilo que también adquiere los recursos en el mismo orden
def hilo_2(num_iteraciones):
    for _ in range(num_iteraciones):
        with recurso_a:
            print("Hilo 2: Recurso A adquirido")
            time.sleep(0.1)
            with recurso_b:
                print("Hilo 2: Recurso B adquirido")
        print("Hilo 2: Recursos liberados\n")

# Configuración de argumentos posicionales
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulación de adquisición ordenada de recursos para evitar interbloqueo.")
    parser.add_argument("num_iteraciones", type=int, help="Número de veces que los hilos intentarán adquirir los recursos.")

    # Obtener los argumentos
    args = parser.parse_args()
    num_iteraciones = args.num_iteraciones

    # Crear los hilos
    hilo1 = threading.Thread(target=hilo_1, args=(num_iteraciones,))
    hilo2 = threading.Thread(target=hilo_2, args=(num_iteraciones,))

    # Iniciar los hilos
    hilo1.start()
    hilo2.start()

    # Esperar a que ambos hilos terminen
    hilo1.join()
    hilo2.join()

    # Mensaje final
    print("Simulación finalizada.")
