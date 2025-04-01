>[!faq] Que el estudiante aplique y diferencie los paradigmas de programación paralela, concurrente y asíncrona mediante la simulación de un sistema realista que requiere procesamiento distribuido de tareas en distintos tiempos y recursos.

## Definición

Imagina un sistema hospitalario automatizado donde los pacientes llegan a urgencias y deben pasar por un proceso compuesto por distintas etapas:
- Registro
- Diagnóstico automatizado (uso de modelos IA preentrenados)
- Asignación de recursos (camas, doctores)
- Seguimiento y alta

Cada etapa puede tener diferentes tiempos de procesamiento, prioridades, y algunos procesos pueden ejecutarse en paralelo o de forma asíncrona. La simulación debe modelar este flujo con código, integrando múltiples técnicas de procesamiento.

### Requerimientos

El estudiante deberá implementar:

- Procesos concurrentes para simular múltiples pacientes interactuando con el sistema.
- Procesos paralelos para procesamiento intensivo (por ejemplo, análisis de datos del diagnóstico).
- Tareas asíncronas para interacciones con APIs simuladas o modelos IA mockeados (simulación de latencia).

El sistema debe tener:
- Control de concurrencia (mutexes, semáforos o similares).
- Procesamiento paralelo usando multiprocessing o librerías equivalentes.
- Uso de asyncio, aiohttp u otro marco asíncrono según el lenguaje elegido.

Python (preferido), Rust, Go o Java.

Un informe en PDF que incluya:
- Diagrama del sistema.
- Justificación del uso de cada paradigma en cada parte.
- Explicación detallada del diseño.
- Fragmentos clave de código con explicación.
- Resultados de pruebas y rendimiento.

Código fuente completo en repositorio con README.

Se permite usar IA como asistente puntual (por ejemplo, resolver errores, sugerencias de diseño), lo cual debe documentarse en el informe:

### Forma de evaluación

| Criterio | Ponderación |
| -------- | ----------- |
| Correcta implementación técnica | 40% |
| Creatividad y realismo del diseño | 20% |
| Claridad del informe PDF | 20% |
| Uso ético y documentado de IA | 10% |
| Buenas prácticas (estructura, pruebas) | 10% |