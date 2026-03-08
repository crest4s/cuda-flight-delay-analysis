# Fase 04: Histograma de Aeropuertos

## Descripción General

Esta fase implementa un análisis CUDA para generar histogramas de frecuencia de aeropuertos utilizando los identificadores numéricos `ORIGIN_SEQ_ID` y `DEST_SEQ_ID` del dataset de vuelos. El sistema cuenta el número de vuelos asociados a cada aeropuerto y muestra los 10 aeropuertos más frecuentes.

## Funcionalidad Implementada

### 1. Kernel CUDA: `buildAirportHistogramKernel`

**Ubicación:** [main.cu](../main.cu)

**Parámetros:**
- `airport_ids`: Array de IDs numéricos de aeropuertos (ORIGIN_SEQ_ID o DEST_SEQ_ID) en memoria GPU
- `histogram`: Array para almacenar el conteo de vuelos por aeropuerto
- `num_records`: Número total de registros
- `max_airport_id`: ID máximo de aeropuerto para dimensionar el histograma

**Comportamiento:**
- Cada hilo procesa un registro del dataset
- Calcula su ID global: `idx = blockIdx.x * blockDim.x + threadIdx.x`
- Lee el ID del aeropuerto (origen o destino según configuración)
- Verifica que el ID sea válido (> 0 y <= max_airport_id)
- Usa operación atómica `atomicAdd(&histogram[airport_id], 1)` para incrementar el contador
- Evita condiciones de carrera mediante operaciones atómicas

**Ventajas de usar IDs numéricos:**
- ✅ Acceso directo al array de histograma: O(1)
- ✅ No requiere búsqueda o hash de strings
- ✅ Operaciones atómicas eficientes en enteros
- ✅ Menor uso de memoria (int vs string)
- ✅ Indexación directa sin colisiones

### 2. Función Wrapper: `executeAirportHistogram`

Orquesta la ejecución completa del análisis de histograma:

1. **Validación**: Verifica que el dataset no esté vacío
2. **Selección de datos**: Obtiene los IDs de aeropuertos (origen o destino)
3. **Cálculo del ID máximo**: Determina el tamaño necesario del histograma
4. **Configuración GPU**: Calcula dimensiones óptimas (bloques e hilos)
5. **Allocación de Memoria**: 
   - Reserva espacio para IDs de aeropuertos
   - Reserva espacio para histograma (max_airport_id + 1 posiciones)
6. **Inicialización**: Establece el histograma a cero con `cudaMemset`
7. **Transferencia Host→Device**: Copia los IDs de aeropuertos a la GPU
8. **Ejecución del Kernel**: Lanza `buildAirportHistogramKernel`
9. **Sincronización**: Espera a que termine la ejecución
10. **Transferencia Device→Host**: Copia el histograma de vuelta a CPU
11. **Post-procesamiento**: Encuentra y ordena los 10 aeropuertos más frecuentes
12. **Visualización**: Muestra los resultados al usuario
13. **Liberación**: Libera toda la memoria de la GPU

**Manejo de Errores:**
- Verifica cada operación de CUDA
- Imprime mensajes descriptivos en caso de fallo
- Realiza limpieza apropiada incluso si hay errores

### 3. Interfaz de Usuario

**Ubicación:** [Menu.cpp](../src/Menu.cpp)

**Función:** `processAirportHistogram()`

**Flujo de interacción:**

1. **Muestra información del análisis**:
   ```
   === Histograma de Aeropuertos ===
   Registros cargados: <número de registros>
   ```

2. **Solicita el tipo de aeropuerto**:
   - Opción 1: Aeropuertos de origen (ORIGIN_SEQ_ID)
   - Opción 2: Aeropuertos de destino (DEST_SEQ_ID)

3. **Ejecución**: Llama a `executeAirportHistogram` con el parámetro correspondiente

4. **Resultados**: 
   - Muestra configuración del histograma
   - Lista los 10 aeropuertos más frecuentes
   - Muestra el total de aeropuertos únicos encontrados

## Ejemplo de Uso

### Caso 1: Histograma de aeropuertos de origen

```
=== Histograma de Aeropuertos ===
Registros cargados: 500000

Tipo de aeropuerto:
  1. Aeropuertos de origen (ORIGIN_SEQ_ID)
  2. Aeropuertos de destino (DEST_SEQ_ID)
Opcion: 1

=== Configuracion del Histograma ===
Registros: 500000
ID maximo de aeropuerto: 16218
Tipo: Aeropuertos de origen

=== Configuracion de Ejecucion CUDA ===
GPU: NVIDIA GeForce RTX 3060
Compute Capability: 8.6
Max Threads por Bloque: 1024
Configuracion: 1954 bloques x 256 hilos
Total de hilos: 500224

=== Top 10 Aeropuertos mas Frecuentes ===
1. Aeropuerto ID 13930: 45230 vuelos
2. Aeropuerto ID 11298: 38421 vuelos
3. Aeropuerto ID 12892: 32145 vuelos
4. Aeropuerto ID 14747: 28934 vuelos
5. Aeropuerto ID 10397: 25678 vuelos
6. Aeropuerto ID 13487: 23456 vuelos
7. Aeropuerto ID 12478: 21890 vuelos
8. Aeropuerto ID 15016: 19234 vuelos
9. Aeropuerto ID 13232: 18567 vuelos
10. Aeropuerto ID 11066: 17234 vuelos

Total de aeropuertos unicos: 342
```

### Caso 2: Histograma de aeropuertos de destino

```
Tipo de aeropuerto:
  1. Aeropuertos de origen (ORIGIN_SEQ_ID)
  2. Aeropuertos de destino (DEST_SEQ_ID)
Opcion: 2

=== Configuracion del Histograma ===
Registros: 500000
ID maximo de aeropuerto: 16218
Tipo: Aeropuertos de destino

=== Top 10 Aeropuertos mas Frecuentes ===
1. Aeropuerto ID 13930: 44987 vuelos
2. Aeropuerto ID 11298: 38654 vuelos
...
```

## Características Técnicas

### Operaciones Atómicas

El kernel utiliza `atomicAdd()` para garantizar actualizaciones seguras del histograma:

```cuda
atomicAdd(&histogram[airport_id], 1);
```

**Ventajas:**
- Evita condiciones de carrera (race conditions)
- Permite que múltiples hilos actualicen el mismo bin simultáneamente
- Garantiza consistencia de datos sin necesidad de locks explícitos

**Consideraciones de rendimiento:**
- Las operaciones atómicas pueden causar serialización si muchos hilos intentan actualizar el mismo bin
- Para este caso de uso (histograma de aeropuertos disperso), el impacto es mínimo
- Los IDs de aeropuertos están bien distribuidos, minimizando colisiones

### Eficiencia de Memoria

**Comparación: Strings vs IDs numéricos**

| Aspecto | Strings (nombres) | IDs numéricos |
|---------|------------------|---------------|
| Memoria por registro | ~10-20 bytes | 4 bytes |
| Memoria total (500K registros) | ~5-10 MB | 2 MB |
| Acceso al histograma | O(n) o hash O(1) | O(1) directo |
| Operaciones atómicas | No disponibles | Eficientes |
| Transferencia GPU | Lenta | Rápida |

**Ahorro de memoria**: ~60-80% usando IDs numéricos

### Escalabilidad

El sistema maneja eficientemente:
- ✅ Miles de aeropuertos únicos
- ✅ Millones de registros de vuelos
- ✅ Histogramas con distribuciones disparejas
- ✅ IDs de aeropuertos no contiguos (usa ID máximo para dimensionar)

## Estructura de Datos

### En CPU (Host)

```cpp
std::vector<int> origin_seq_id_;  // IDs de aeropuertos origen
std::vector<int> dest_seq_id_;    // IDs de aeropuertos destino
std::vector<int> h_histogram;     // Histograma resultante
```

### En GPU (Device)

```cpp
int* d_airport_ids;     // Array de IDs de aeropuertos
int* d_histogram;       // Array de contadores (tamaño = max_id + 1)
```

## Mejoras Futuras Posibles

1. **Filtrado por rango de fechas**: Analizar histogramas por período temporal
2. **Visualización gráfica**: Generar gráficos de barras o mapas de calor
3. **Análisis combinado**: Pares origen-destino más frecuentes (matriz 2D)
4. **Estadísticas adicionales**: Media, mediana, desviación estándar de vuelos por aeropuerto
5. **Exportación de datos**: Guardar resultados en CSV o JSON
6. **GPU Streaming**: Procesar datasets que no caben en memoria GPU mediante chunks

## Validación

Para verificar que el kernel funciona correctamente:

1. **Suma total**: El total de vuelos en el histograma debe ser ≤ num_records
2. **IDs válidos**: Todos los bins con contador > 0 deben tener ID válido
3. **Consistencia**: Ejecutar dos veces debe dar el mismo resultado
4. **Comparación CPU**: Los resultados deben coincidir con un conteo secuencial en CPU

## Referencias

- Operaciones atómicas en CUDA: [CUDA C Programming Guide - Atomic Functions](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions)
- Histogramas en GPU: "GPU Gems 3" - Chapter 39: Parallel Histogram Computation
