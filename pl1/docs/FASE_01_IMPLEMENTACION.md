# Fase 01: Análisis de Retraso en Despegue (DEP_DELAY)

## Descripción General

Esta fase implementa un análisis CUDA de la columna `DEP_DELAY` del dataset de vuelos. El sistema identifica y muestra los vuelos que superan un umbral de retraso o adelanto especificado por el usuario.

## Funcionalidad Implementada

### 1. Kernel CUDA: `analyzeDepDelayKernel`

**Ubicación:** [main.cu](../main.cu)

**Parámetros:**
- `dep_delays`: Array de valores DEP_DELAY en memoria GPU
- `num_records`: Número total de registros
- `threshold`: Umbral en minutos
- `delay_type`: Tipo de análisis (true = retraso, false = adelanto)

**Comportamiento:**
- Cada hilo procesa un registro del dataset
- Calcula su ID global: `idx = blockIdx.x * blockDim.x + threadIdx.x`
- Verifica que el valor no sea NaN
- Compara el valor contra el umbral según el tipo:
  - **Retraso**: `delay >= threshold` (valores positivos)
  - **Adelanto**: `delay <= threshold` (valores negativos)
- Si cumple la condición, imprime: `#Hilo <idx>: Retraso de <delay> minutos.`

### 2. Detección y Dimensionamiento Automático de GPU

**Funciones Auxiliares:**

#### `getGPUConfiguration()`
Obtiene las características hardware de la GPU:
- Máximo de hilos por bloque
- Máximo de bloques en la grid
- Número de Streaming Multiprocessors (SMs)
- Nombre del dispositivo
- Memoria global total

#### `calculateOptimalDimensions()`
Calcula la configuración óptima de ejecución:
- **Hilos por bloque**: 256 (si la GPU soporta >= 512 threads/block) o 128
- **Número de bloques**: `ceil(num_records / threads_per_block)`
- Limitado al máximo permitido por la GPU
- Muestra información de configuración antes de la ejecución

### 3. Función Wrapper: `executeDepDelayAnalysis`

Orquesta la ejecución completa:
1. Calcula dimensiones óptimas según el hardware
2. Aloca memoria en la GPU
3. Copia datos del host al device
4. Lanza el kernel CUDA
5. Sincroniza y verifica errores
6. Libera memoria de la GPU

### 4. Interfaz de Usuario

**Ubicación:** [Menu.cpp](../src/Menu.cpp)

**Función:** `processDepartureDelay()`

**Flujo de interacción:**
1. Muestra información sobre el dataset y el análisis
2. Solicita el tipo de análisis:
   - Opción 1: Retrasos (vuelos que salen tarde)
   - Opción 2: Adelantos (vuelos que salen temprano)
3. Solicita el umbral en minutos:
   - Para retrasos: valores positivos (ej: 1440 para 24 horas)
   - Para adelantos: valores negativos (ej: -30 para 30 minutos adelantados)
4. Valida la entrada del usuario
5. Confirma parámetros antes de ejecutar
6. Ejecuta el análisis en GPU
7. Muestra resultados

## Ejemplo de Uso

### Caso 1: Detectar retrasos superiores a 24 horas

```
=== Análisis de Retraso en Salida (DEP_DELAY) ===

Registros en dataset: 500000

Tipo de análisis:
  1. Retrasos (vuelos que salen tarde)
  2. Adelantos (vuelos que salen temprano)
Opción: 1

Ingrese el umbral en minutos:
  (Ej: 1440 para detectar vuelos con retraso >= 24 horas)
Umbral: 1440

=== Parámetros de análisis ===
Tipo: retraso
Umbral: 1440 minutos

=== Configuración de ejecución CUDA ===
Dispositivo: NVIDIA GeForce RTX 3060
SMs disponibles: 28
Registros a procesar: 500000
Bloques: 1954
Hilos por bloque: 256
Total de hilos: 500224

Ejecutando análisis en GPU...

=== RESULTADOS ===
#Hilo 123454: Retraso de 1855 minutos.
#Hilo 234567: Retraso de 1620 minutos.
#Hilo 345678: Retraso de 2100 minutos.
...
=== FIN DE RESULTADOS ===
```

### Caso 2: Detectar adelantos superiores a 30 minutos

```
Tipo de análisis:
  1. Retrasos (vuelos que salen tarde)
  2. Adelantos (vuelos que salen temprano)
Opción: 2

Ingrese el umbral en minutos:
  (Ej: -30 para detectar vuelos adelantados >= 30 minutos)
  (Use valores negativos para adelantos)
Umbral: -30

=== RESULTADOS ===
#Hilo 1234: Retraso de -35 minutos.
#Hilo 5678: Retraso de -42 minutos.
...
```

## Características Técnicas

### Gestión de Valores Faltantes
- El kernel verifica valores NaN usando `isnan()`
- Los registros con valores faltantes se ignoran automáticamente

### Dimensionamiento Dinámico
- El sistema se adapta automáticamente a las características de la GPU
- Optimiza el número de bloques e hilos según el hardware disponible
- Compatible con diferentes arquitecturas CUDA

### Manejo de Errores
- Verificación de errores en todas las operaciones CUDA
- Sincronización explícita con `cudaDeviceSynchronize()`
- Mensajes de error descriptivos

## Compilación

El proyecto incluye un Makefile que maneja la compilación de archivos CUDA (.cu) y C++ (.cpp):

```bash
make
```

## Notas de Implementación

1. **Paralelización**: Cada hilo procesa exactamente un registro, maximizando el paralelismo
2. **Output**: Los resultados se imprimen directamente desde el kernel usando `printf()` de CUDA
3. **Validación**: El sistema valida que el umbral tenga el signo correcto según el tipo de análisis
4. **Extensibilidad**: La arquitectura permite añadir fácilmente nuevos análisis (DEP_DELAY, ARR_DELAY, etc.)

## Próximos Pasos

Las siguientes fases del proyecto pueden implementar:
- Fase 02: Análisis de retraso en llegada (ARR_DELAY)
- Fase 03: Reducción de retraso (diferencia entre DEP_DELAY y ARR_DELAY)
- Fase 04: Histogramas de aeropuertos más frecuentes
