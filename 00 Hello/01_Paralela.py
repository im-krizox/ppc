import multiprocessing
import time

def tarea(nombre:str):
    print(f'Hola, soy {nombre}')
    time.sleep(2)
    print(f'{nombre} ha terminado su tarea')

procesos = [multiprocessing.Process(target=tarea, args=(f'Proceso {i}',)) for i in range(10000)]


if __name__ == '__main__':
    for proceso in procesos:
        proceso.start()

    for proceso in procesos:
        proceso.join()
