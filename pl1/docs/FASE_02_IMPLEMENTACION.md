# Fase 02: Análisis de Retraso en Llegada (ARR_DELAY)

## Descripción General

Esta fase implementa un análisis CUDA de la columna `ARR_DELAY` del dataset de vuelos. El sistema identifica y muestra los vuelos que superan un umbral de retraso o adelanto en su llegada especificado por el usuario.

## Funcionalidad Implementada

### 1. Kernel CUDA: `analyzeArrDelayKernel`

**Ubicación:** [main.cu](../main.cu)

**Parámetros:**
- `arr_delays`: Array de valores ARR_DELAY en memoria GPU
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

### 2. Función Wrapper: `executeArrDelayAnalysis`

Orquesta la ejecución completa del análisis de ARR_DELAY:

1. **Validación**: Verifica que el dataset no esté vacío
2. **Configuración GPU**: Calcula dimensiones óptimas (bloques e hilos) según el hardware disponible
3. **Allocación de Memoria**: Reserva espacio en la memoria global de la GPU (`cudaMalloc`)
4. **Transferencia Host→Device**: Copia los datos de ARR_DELAY desde la RAM al VRAM
5. **Ejecución del Kernel**: Lanza `analyzeArrDelayKernel` con la configuración calculada
6. **Sincronización**: Espera a que termine la ejecución en GPU
7. **Liberación**: Libera la memoria de la GPU

**Manejo de Errores:**
- Verifica el resultado de cada operación CUDA
- Imprime mensajes descriptivos en caso de fallo
- Realiza limpieza apropiada (free) incluso si hay errores

### 3. Interfaz de Usuario

**Ubicación:** [Menu.cpp](../src/Menu.cpp)

**Función:** `processArrivalDelay()`

**Flujo de interacción:**

1. **Muestra información del análisis**:
   ```
   Retraso en Llegada (ARR_DELAY)
   Registros: <número de registros cargados>
   ```

2. **Solicita el tipo de análisis**:
   - Opción 1: Retrasos (vuelos que llegan tarde)
   - Opción 2: Adelantos (vuelos que llegan temprano)

3. **Solicita el umbral en minutos**:
   - Para retrasos: valores positivos (ej: 60 para detectar retrasos de 1 hora o más)
   - Para adelantos: valores negativos (ej: -15 para detectar adelantos de 15 minutos o más)

4. **Validación de entrada**: Verifica que el umbral ingresado sea un número válido

5. **Ejecución**: Llama a `executeArrDelayAnalysis` con los parámetros especificados

6. **Resultados**: Muestra en consola todos los vuelos que cumplen la condición

## Ejemplo de Uso

### Caso 1: Detectar retrasos en llegada superiores a 2 horas

```
=== Análisis de Retraso en Llegada (ARR_DELAY) ===

Registros en dataset: 500000

Tipo de análisis:
  1. Retrasos (vuelos que llegan tarde)
  2. Adelantos (vuelos que llegan temprano)
Opción: 1

Umbral (minutos): 120

=== Configuración de ejecución CUDA ===
Ejecutando en: NVIDIA GeForce RTX 3060
Configuración: 1954 bloques x 256 hilos

#Hilo 45: Retraso de 135 minutos.
#Hilo 128: Retraso de 240 minutos.
#Hilo 376: Retraso de 180 minutos.
...
```

### Caso 2: Detectar adelantos en llegada (llegadas tempranas)

```
=== Análisis de Retraso en Llegada (ARR_DELAY) ===

Registros en dataset: 500000

Tipo de análisis:
  1. Retrasos (vuelos que llegan tarde)
  2. Adelantos (vuelos que llegan temprano)
Opción: 2

Umbral (minutos): -20

=== Configuración de ejecución CUDA ===
Ejecutando en: NVIDIA GeForce RTX 3060
Configuración: 1954 bloques x 256 hilos

#Hilo 89: Retraso de -25 minutos.
#Hilo 234: Retraso de -30 minutos.
#Hilo 567: Retraso de -45 minutos.
...
```

## Estructura del Código

### Archivos Modificados

1. **main.cu**
   - Añadido kernel `analyzeArrDelayKernel`
   - Añadida función wrapper `executeArrDelayAnalysis`

2. **Menu.cpp**
   - Implementada función `processArrivalDelay()`
   - Añadida declaración externa de `executeArrDelayAnalysis`

### Flujo de Datos

```
Usuario (Menu.cpp)
    ↓ (threshold, delay_type)
executeArrDelayAnalysis (main.cu)
    ↓ (aloca memoria + copia datos)
GPU Memory
    ↓ (kernel execution)
analyzeArrDelayKernel (CUDA)
    ↓ (resultados por printf)
Consola
```

## Diferencias con DEP_DELAY (Fase 01)

Aunque la estructura es idéntica, la semántica es diferente:

| Aspecto | DEP_DELAY | ARR_DELAY |
|---------|-----------|-----------|
| **Evento** | Salida del vuelo | Llegada del vuelo |
| **Significado de retraso positivo** | Vuelo despegó tarde | Vuelo aterrizó tarde |
| **Significado de retraso negativo** | Vuelo despegó temprano | Vuelo aterrizó temprano |
| **Kernel CUDA** | `analyzeDepDelayKernel` | `analyzeArrDelayKernel` |
| **Wrapper** | `executeDepDelayAnalysis` | `executeArrDelayAnalysis` |
| **Menú** | Opción 1 | Opción 2 |

## Consideraciones de Implementación

### 1. Paralelismo
- Cada hilo CUDA procesa un vuelo de manera independiente
- No hay dependencias entre hilos
- Escalabilidad lineal con el número de SMs disponibles

### 2. Gestión de Memoria
- **Memoria Global**: Se utiliza para almacenar el array completo de ARR_DELAY
- **Acceso Linealizado**: Los datos se acceden en 1D como especifica el requisito
- **Copias Host↔Device**: Una sola copia de entrada (Host→Device), sin copia de salida (resultados vía printf)

### 3. Manejo de Valores Faltantes
- Los valores NaN (Not a Number) son ignorados automáticamente
- Se verifica con `isnan()` antes de realizar la comparación
- No se imprime nada para registros con datos faltantes

### 4. Configuración Dinámica
- Reutiliza `calculateOptimalDimensions()` de la Fase 01
- Se adapta al hardware disponible automáticamente
- Balance entre ocupación y eficiencia

## Compilación y Ejecución

El proyecto se compila con CUDA y se ejecuta desde el menú principal:

```bash
# Compilación (si usas Makefile)
make

# O compilación directa con nvcc
nvcc -o pl1 main.cu src/*.cpp -Iinclude

# Ejecución
./pl1
```

Luego desde el menú:
1. Cargar el dataset CSV
2. Seleccionar opción 2 (Retraso en llegada)
3. Elegir tipo de análisis
4. Ingresar umbral
5. Ver resultados en consola

## Próximos Pasos

- **Fase 03**: Análisis de reducción de retraso (diferencia entre DEP_DELAY y ARR_DELAY)
- **Fase 04**: Histograma de aeropuertos
- **Optimizaciones**: Uso de memoria compartida, reducción paralela, etc.

---

**Fecha de implementación**: 6 de marzo de 2026
