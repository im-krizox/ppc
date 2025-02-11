import threading
import time 

def tarea(nombre:str):
    print(f'Hola, soy {nombre}')
    time.sleep(2)
    print(f'{nombre} ha terminado su tarea')

hilos = [threading.Thread(target=tarea, args=(f'Hilo {i}',)) for i in range(10000)]

for hilo in hilos:
    hilo.start()

for hilo in hilos:
    hilo.join()

