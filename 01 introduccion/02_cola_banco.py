import threading
import time
import random
import argparse

# Función que simula la atención de un cliente por un cajero
def atender_cliente(cliente_id, semaforo):
    with semaforo:  # Adquirir un cajero disponible
        tiempo_atencion = random.randint(1, 5)  # Simulamos entre 1 y 5 segundos de atención
        print(f"Cajero atendiendo al Cliente {cliente_id}... Tiempo estimado: {tiempo_atencion} segundos")
        time.sleep(tiempo_atencion)
        print(f"Cliente {cliente_id} atendido en {tiempo_atencion} segundos\n")

# Simulación de una cola de banco con cajeros limitados
def cola_de_banco(num_clientes, num_cajeros):
    semaforo = threading.Semaphore(num_cajeros)  # Limitar el número de cajeros disponibles
    hilos = []

    # Crear un hilo para cada cliente
    for cliente_id in range(1, num_clientes + 1):
        hilo = threading.Thread(target=atender_cliente, args=(cliente_id, semaforo))
        hilos.append(hilo)
        hilo.start()  # Iniciar el hilo (atender cliente)

    # Esperar a que todos los hilos terminen
    for hilo in hilos:
        hilo.join()

# Configuración de argumentos posicionales
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulación de una cola de banco con múltiples clientes y cajeros.")
    parser.add_argument("num_clientes", type=int, help="Número de clientes a atender en la cola de banco.")
    parser.add_argument("num_cajeros", type=int, help="Número de cajeros disponibles para atender a los clientes.")

    # Obtener los argumentos
    args = parser.parse_args()
    num_clientes = args.num_clientes
    num_cajeros = args.num_cajeros

    # Medir el tiempo total de la simulación
    print("Iniciando la simulación de la cola de banco...\n")
    tiempo_inicio = time.time()
    cola_de_banco(num_clientes, num_cajeros)
    tiempo_total = time.time() - tiempo_inicio
    print(f"Todos los clientes han sido atendidos en {tiempo_total:.2f} segundos.")
