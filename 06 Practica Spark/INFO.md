>[!INFO] Aplicar los conceptos fundamentales de Spark (RDDs, DataFrames, transformaciones y acciones) para realizar un análisis distribuido de datos reales o semi-reales sobre movilidad urbana.

## Definición

El gobierno de la Ciudad de México publica datos abiertos sobre movilidad, tránsito y transporte público. Para esta tarea, el estudiante trabajará con un dataset que simule (o se base en) registros de viajes de transporte público, GPS de unidades o bicis compartidas.

El objetivo es realizar un análisis que responda a preguntas como:
- ¿Cuáles son las rutas con mayor congestión?
- ¿En qué horarios hay más viajes?
- ¿Qué zonas tienen más entrada o salida de vehículos?

### Requerimientos

El análisis debe incluir:
- Lectura de datos en formato CSV o JSON.
- Uso de y , con transformaciones y acciones.
- Al menos una operación de agregación distribuida (por ejemplo, reduceByKey, groupBy).
- Aplicación de filtros, mapas y joins (si aplica).
- Generación de un resumen en un archivo CSV o visualización simple (opcional con matplotlib o similar).

Se puede utilizar PySpark o Spark con Scala.

Pueden consultar IA para resolver dudas de sintaxis.

Deben documentar en el informe qué consultas hicieron a IA, en qué parte del código se usaron y cómo ayudaron.

### Entregable

- Descripción del dataset (fuente o cómo se generó si es sintético).
- Explicación de las operaciones realizadas y por qué se usaron.
- Análisis de los resultados.
- Fragmentos clave de código con comentarios.
- Reflexión sobre qué partes se pudieron distribuir eficientemente y cuáles no.
- Organizado y ejecutable en entorno local o en Databricks.

### Forma de evaluación

| Criterio | Ponderación |
| -------- | ----------- |
| Aplicación correcta de operaciones Spark | 40% |
| Calidad del análisis y justificación | 20% |
| Documentación y claridad del informe | 20% |
|Uso ético y documentado de IA como herramienta auxiliar | 10% |
| Organización del código y reproducibilidad | 10% |