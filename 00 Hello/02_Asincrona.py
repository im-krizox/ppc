import asyncio

async def tarea(nombre:str):
    print(f'Hola, soy {nombre}')
    await asyncio.sleep(2)
    print(f'{nombre} ha terminado su tarea')

async def main():
    tareas = [tarea(f'Hilo {i}') for i in range(1000000)]
    await asyncio.gather(*tareas)

if __name__ == '__main__':
    asyncio.run(main())