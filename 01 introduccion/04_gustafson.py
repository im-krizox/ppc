import argparse

# Función que calcula el Speedup usando la Ley de Gustafson
def calcular_speedup_gustafson(fraccion_paralelizable, num_procesadores):
    # Fórmula de la Ley de Gustafson
    speedup = num_procesadores - (1 - fraccion_paralelizable) * (num_procesadores - 1)
    return speedup

# Configuración de argumentos posicionales
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculadora del Speedup según la Ley de Gustafson.")
    parser.add_argument("fraccion_paralelizable", type=float, help="Fracción del programa que es paralelizable (entre 0 y 1).")
    parser.add_argument("num_procesadores", type=int, help="Número de procesadores disponibles.")

    # Obtener los argumentos
    args = parser.parse_args()
    fraccion_paralelizable = args.fraccion_paralelizable
    num_procesadores = args.num_procesadores

    # Validar los valores
    if not (0 <= fraccion_paralelizable <= 1):
        print("Error: La fracción paralelizable debe estar entre 0 y 1.")
        exit(1)

    if num_procesadores < 1:
        print("Error: El número de procesadores debe ser al menos 1.")
        exit(1)

    # Calcular y mostrar el Speedup
    speedup = calcular_speedup_gustafson(fraccion_paralelizable, num_procesadores)
    print(f"Fracción paralelizable: {fraccion_paralelizable * 100:.2f}%")
    print(f"Número de procesadores: {num_procesadores}")
    print(f"Speedup obtenido: {speedup:.2f}")
