# 🚀 Sistema de Análisis de Vuelos - US Airline Dataset

## 📋 Descripción del Proyecto

Sistema profesional de análisis de grandes volúmenes de datos de vuelos utilizando **C++** para el procesamiento en CPU (Host) y **CUDA** para aceleración en GPU (Device). El proyecto procesa el dataset "US Airline Dataset" (~125 MB, 1.2M registros) de forma eficiente y escalable.

### 🎯 Características Principales

- ✅ **Arquitectura Modular**: Separación clara entre lógica CPU y GPU
- ✅ **Estructura Orientada a Objetos**: Diseño profesional y mantenible
- ✅ **Parser CSV Robusto**: Manejo automático de valores faltantes (NaN)
- ✅ **Almacenamiento Vectorizado**: Datos organizados por columnas para transferencia eficiente a GPU
- ✅ **Interfaz de Usuario Interactiva**: Menú intuitivo con múltiples opciones de análisis
- ✅ **Preparado para CUDA**: Base lista para integrar kernels de procesamiento paralelo

---

## 🏗️ Arquitectura del Proyecto

```
paradigmas-lab/
├── include/                    # Headers (.h)
│   ├── FlightDataset.h        # Estructura de datos del dataset
│   ├── CSVParser.h            # Parser CSV con limpieza de datos
│   └── Menu.h                 # Sistema de menú interactivo
├── src/                       # Implementaciones (.cpp)
│   ├── FlightDataset.cpp      
│   ├── CSVParser.cpp          
│   └── Menu.cpp               
├── main.cu                    # Punto de entrada (CUDA)
├── Makefile                   # Sistema de compilación
├── data/                      # Directorio para datasets (crear manualmente)
│   └── us_airline_dataset.csv # Dataset de vuelos
├── obj/                       # Archivos objeto (auto-generado)
├── bin/                       # Ejecutable (auto-generado)
└── README.md                  # Este archivo
```

### 📊 Componentes

#### 1. **FlightDataset** (`FlightDataset.h/cpp`)
Clase que encapsula el dataset completo usando vectores independientes por columna:
- `DEP_DELAY` (float) - Retraso en salida
- `ARR_DELAY` (float) - Retraso en llegada  
- `WEATHER_DELAY` (float) - Retraso por clima
- `TAIL_NUM` (string) - Número de cola del avión
- `ORIGIN_SEQ_ID` (int) - ID aeropuerto de origen
- `DEST_SEQ_ID` (int) - ID aeropuerto de destino

#### 2. **CSVParser** (`CSVParser.h/cpp`)
Parser robusto con características profesionales:
- Manejo automático de comillas y delimitadores
- Conversión de valores faltantes a `NaN` (en lugar de 0 o valores arbitrarios)
- Validación de formato de archivo
- Reporte de progreso en tiempo real
- Gestión de errores y registros inválidos

#### 3. **Menu** (`Menu.h/cpp`)
Sistema de interfaz de usuario con:
- Solicitud de ruta de archivo (con opción por defecto)
- Menú principal con 4 opciones de análisis + salida
- Preparado para integración con kernels CUDA

#### 4. **main.cu**
Orquestador principal que:
- Verifica disponibilidad de CUDA
- Muestra información del dispositivo GPU
- Coordina carga de datos y ejecución del menú
- Gestiona limpieza de recursos

---

## 🛠️ Requisitos del Sistema

### Software Necesario

1. **CUDA Toolkit** (versión 10.0 o superior)
   - Descarga: https://developer.nvidia.com/cuda-downloads
   
2. **Compilador C++** compatible con C++11:
   - Linux: `g++` (instalado con build-essential)
   - macOS: Xcode Command Line Tools
   - Windows: Visual Studio 2017 o superior

3. **GPU NVIDIA** con soporte CUDA (Compute Capability 5.0+)
   - Verificar compatibilidad: https://developer.nvidia.com/cuda-gpus

### Verificar Instalación

```bash
# Verificar CUDA
nvcc --version

# Verificar GPU
nvidia-smi

# Verificar compilador C++
g++ --version  # Linux/macOS
```

---

## 🚀 Compilación y Ejecución

### Opción 1: Usando Makefile (Recomendado)

```bash
# Compilar el proyecto
make

# Compilar y ejecutar
make run

# Limpiar archivos objeto
make clean

# Limpiar y recompilar
make rebuild

# Ver información del sistema CUDA
make info

# Ver ayuda
make help
```

### Opción 2: Compilación Manual

```bash
# Crear directorios necesarios
mkdir -p obj bin data

# Compilar archivos C++
g++ -std=c++11 -O2 -c src/FlightDataset.cpp -o obj/FlightDataset.o
g++ -std=c++11 -O2 -c src/CSVParser.cpp -o obj/CSVParser.o
g++ -std=c++11 -O2 -c src/Menu.cpp -o obj/Menu.o

# Compilar y enlazar con NVCC
nvcc -std=c++11 -O2 -arch=sm_50 \
     obj/FlightDataset.o obj/CSVParser.o obj/Menu.o main.cu \
     -o bin/flights_analyzer

# Ejecutar
./bin/flights_analyzer
```

---

## 📁 Preparación del Dataset

1. **Descargar el dataset** "US Airline Dataset" (o usar tu propio CSV)

2. **Verificar formato**: El CSV debe contener las siguientes columnas:
   ```
   DEP_DELAY,ARR_DELAY,WEATHER_DELAY,TAIL_NUM,
   ORIGIN_AIRPORT_SEQ_ID,DEST_AIRPORT_SEQ_ID,...
   ```

3. **Colocar el archivo**:
   ```bash
   # Opción 1: Usar ruta por defecto
   mkdir -p data
   cp tu_dataset.csv data/us_airline_dataset.csv
   
   # Opción 2: Especificar ruta al ejecutar el programa
   # El programa te pedirá la ruta interactivamente
   ```

---

## 💻 Uso del Programa

### 1. Iniciar el Programa

```bash
./bin/flights_analyzer
```

### 2. Cargar Dataset

El programa solicitará la ruta del archivo CSV:
- Presiona **Enter** para usar la ruta por defecto: `./data/us_airline_dataset.csv`
- O escribe una ruta personalizada

### 3. Menú Principal

```
╔═══════════════════════════════════════════════════════════╗
║     SISTEMA DE ANÁLISIS DE VUELOS - US AIRLINE DATASET   ║
║                   (CPU + GPU con CUDA)                    ║
╚═══════════════════════════════════════════════════════════╝

  Seleccione una opción:

    1. Retraso en salida (DEP_DELAY)
    2. Retraso en llegada (ARR_DELAY)
    3. Reducción de retraso
    4. Histograma de aeropuertos

    x. Salir
```

### 4. Opciones Disponibles

- **Opción 1**: Análisis de retrasos en salida
- **Opción 2**: Análisis de retrasos en llegada
- **Opción 3**: Cálculo de reducción de retraso (DEP - ARR)
- **Opción 4**: Generación de histograma de aeropuertos
- **Opción x**: Salir del programa

> **Nota**: Las opciones 1-4 están preparadas para integración con kernels CUDA en fases posteriores del proyecto.

---

## 🔧 Características Técnicas

### Limpieza de Datos

El parser implementa una limpieza automática según las mejores prácticas:

```cpp
// Valores faltantes o inválidos → NaN (no 0 o valores arbitrarios)
float value = parseFloat(csv_field);  // Retorna NaN si falta el valor
```

Esto evita sesgos en análisis estadísticos posteriores.

### Optimización de Memoria

- **Vectores pre-reservados**: Reduce realocaciones durante la carga
- **Almacenamiento SOA** (Structure of Arrays): Óptimo para transferencias GPU
- **Referencia por columna**: Acceso directo sin copias innecesarias

### Preparación para CUDA

Los vectores están diseñados para copiarse linealmente a memoria global GPU:

```cpp
// Ejemplo de transferencia futura a GPU:
float* d_dep_delay;
cudaMalloc(&d_dep_delay, dataset.size() * sizeof(float));
cudaMemcpy(d_dep_delay, dataset.getDepDelay().data(), 
           dataset.size() * sizeof(float), cudaMemcpyHostToDevice);
```

---

## 📊 Ejemplo de Ejecución

```
═════════════════════════════════════════════════════════════
           ANÁLISIS DE DATASET (C++ & CUDA)            
═════════════════════════════════════════════════════════════

╔═══════════════════════════════════════════════════════════╗
║                INFORMACIÓN DEL DISPOSITIVO GPU            ║
╚═══════════════════════════════════════════════════════════╝
  Dispositivo: NVIDIA GeForce RTX 3080
  Compute Capability: 8.6
  Memoria Global: 10240 MB
  Multiprocessors: 68
  CUDA Cores: ~8704 (aproximado)
═════════════════════════════════════════════════════════════

Cargando dataset desde: ./data/us_airline_dataset.csv
Por favor espera, esto puede tardar unos momentos...
  Procesados 100000 registros...
  Procesados 200000 registros...
  ...
  Procesados 1200000 registros...

✓ Carga completada exitosamente!
  Registros cargados: 1234567
  Registros omitidos: 25

╔════════════════════════════════════════════════════╗
║         ESTADÍSTICAS DEL DATASET CARGADO          ║
╚════════════════════════════════════════════════════╝
  Total de registros: 1234567

  Valores faltantes (NaN):
    - DEP_DELAY:        12345 (1.00%)
    - ARR_DELAY:        23456 (1.90%)
    - WEATHER_DELAY:   123456 (10.00%)

  Memoria aproximada: 47.52 MB
════════════════════════════════════════════════════
```

---

## 🗺️ Roadmap del Proyecto

### ✅ Fase 1: CSV Parser y Menú (COMPLETADA)
- [x] Estructura de datos orientada a objetos
- [x] Parser CSV robusto con limpieza de datos
- [x] Sistema de menú interactivo
- [x] Arquitectura modular CPU/GPU

### 🔄 Fase 2: Kernels CUDA (PENDIENTE)
- [ ] Kernel para estadísticas de DEP_DELAY
- [ ] Kernel para estadísticas de ARR_DELAY
- [ ] Kernel para reducción de retraso
- [ ] Kernel para histograma de aeropuertos

### 🔄 Fase 3: Optimización (PENDIENTE)
- [ ] Implementación de memoria compartida
- [ ] Optimización de patrones de acceso
- [ ] Análisis de performance (CPU vs GPU)
- [ ] Visualización de resultados

---

## 🐛 Solución de Problemas

### Error: "nvcc: command not found"
```bash
# Agregar CUDA al PATH (Linux/macOS)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Error: "undefined reference to cudaXXX"
Asegúrate de compilar con `nvcc` en lugar de `g++` para el enlace final.

### Error: "no CUDA-capable device is detected"
Verifica que tu sistema tenga una GPU NVIDIA con soporte CUDA:
```bash
nvidia-smi
```

### Dataset no carga correctamente
1. Verifica que el CSV tenga las columnas requeridas
2. Comprueba que no haya caracteres especiales en la ruta
3. Revisa que tengas permisos de lectura en el archivo

---

## 📚 Recursos Adicionales

- [Documentación CUDA](https://docs.nvidia.com/cuda/)
- [Guía de Programación CUDA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Ejemplos de CUDA](https://github.com/NVIDIA/cuda-samples)

---

## 👨‍💻 Autor

**Proyecto Universitario - Paradigmas de Programación**  
Universidad: [Tu Universidad]  
Curso: Paradigmas de Programación  
Año: 2026

---

## 📄 Licencia

Este proyecto es material académico desarrollado con fines educativos.

---

## 🤝 Reglas de Desarrollo

> **IMPORTANTE**: A partir de este punto, para todas las mejoras, adaptaciones e implementaciones de nuevas fases:
> 
> - ✅ **Especificar cambios sobre código existente** (no crear archivos nuevos innecesariamente)
> - ✅ **Asegurar compatibilidad** con el resto del proyecto
> - ✅ **Integrar en todos los lugares necesarios** (actualizar menú, llamadas, etc.)
> - ✅ **Mantener la arquitectura modular** (separación CPU/GPU)

---

**¿Preguntas o problemas?** Revisa la sección de [Solución de Problemas](#-solución-de-problemas) o consulta la documentación oficial de CUDA.