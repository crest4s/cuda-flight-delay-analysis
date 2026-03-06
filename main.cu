/**
 * @file main.cu
 * @brief Punto de entrada principal del sistema de análisis de vuelos
 * 
 * Este archivo integra la lógica CPU (Host) con CUDA (Device) para el
 * procesamiento eficiente del US Airline Dataset.
 * 
 * Arquitectura:
 *   - Host (CPU): Carga de datos, parsing CSV, interfaz de usuario
 *   - Device (GPU): Procesamiento paralelo mediante kernels CUDA
 * 
 * @author Proyecto Paradigmas de Programación
 * @date 2026
 */

#include <iostream>
#include <climits>
#include <cuda_runtime.h>
#include "include/FlightDataset.h"
#include "include/CSVParser.h"
#include "include/Menu.h"

const std::string DEFAULT_CSV_PATH = "./data/Airline_dataset.csv";

<<<<<<< HEAD
__global__ void reduceSimple(const int* data, int size, int* result, bool find_max) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        int value = data[idx];
        if (find_max) {
            atomicMax(result, value);
        } else {
            atomicMin(result, value);
        }
    }
}

__global__ void reduceBasic(const int* data, int size, int* result, bool find_max) {
    extern __shared__ int shared_data[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Cargar elemento propio en memoria compartida
    if (idx < size) {
        shared_data[tid] = data[idx];
    } else {
        shared_data[tid] = find_max ? INT_MIN : INT_MAX;
    }
    __syncthreads();

    if (idx < size) {
        int local_extreme = shared_data[tid]; // posicion actual

        // Posicion anterior
        if (tid > 0) {
            // el hilo anterior esta en el mismo bloque
            int prev = shared_data[tid - 1];
            local_extreme = find_max ? max(local_extreme, prev) : min(local_extreme, prev);
        } else if (idx > 0) {
            // primer hilo del bloque: leer de memoria global
            int prev = data[idx - 1];
            local_extreme = find_max ? max(local_extreme, prev) : min(local_extreme, prev);
        }

        // Posicion siguiente
        if (tid < blockDim.x - 1) {
            if (idx + 1 < size) {
                int next = shared_data[tid + 1];
                local_extreme = find_max ? max(local_extreme, next) : min(local_extreme, next);
            }
        } else if (idx + 1 < size) {
            // ultimo hilo del bloque: leer de memoria global
            int next = data[idx + 1];
            local_extreme = find_max ? max(local_extreme, next) : min(local_extreme, next);
        }

        if (find_max) {
            atomicMax(result, local_extreme);
        } else {
            atomicMin(result, local_extreme);
        }
    }
}

__global__ void reduceIntermediate(const int* data, int size, int* result, bool find_max) {
    extern __shared__ int shared_data[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Cargar elemento propio en memoria compartida
    if (idx < size) {
        shared_data[tid] = data[idx];
    } else {
        shared_data[tid] = find_max ? INT_MIN : INT_MAX;
    }
    __syncthreads();

    // Cada hilo calcula el extremo de sus 3 posiciones (anterior, actual, siguiente)
    int local_extreme = shared_data[tid];

    if (idx < size) {
        // Posicion anterior
        if (tid > 0) {
            int prev = shared_data[tid - 1];
            local_extreme = find_max ? max(local_extreme, prev) : min(local_extreme, prev);
        } else if (idx > 0) {
            int prev = data[idx - 1];
            local_extreme = find_max ? max(local_extreme, prev) : min(local_extreme, prev);
        }

        // Posicion siguiente
        if (tid < blockDim.x - 1) {
            if (idx + 1 < size) {
                int next = shared_data[tid + 1];
                local_extreme = find_max ? max(local_extreme, next) : min(local_extreme, next);
            }
        } else if (idx + 1 < size) {
            int next = data[idx + 1];
            local_extreme = find_max ? max(local_extreme, next) : min(local_extreme, next);
        }
    }

    // Guardar en memoria compartida (en la posicion del hilo)
    shared_data[tid] = local_extreme;
    __syncthreads();

    // Solo hilos pares: miran su valor y el siguiente, operacion atomica global
    if (tid % 2 == 0) {
        int extreme = shared_data[tid];
        if (tid + 1 < blockDim.x) {
            extreme = find_max ? max(extreme, shared_data[tid + 1])
                               : min(extreme, shared_data[tid + 1]);
        }
        if (find_max) {
            atomicMax(result, extreme);
        } else {
            atomicMin(result, extreme);
        }
    }
}

__global__ void reduceTreePattern(const int* data, int size, int* partial_results, bool find_max) {
    extern __shared__ int shared_data[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    if (idx < size) {
        shared_data[tid] = data[idx];
    } else {
        if (find_max) {
            shared_data[tid] = INT_MIN;
        } else {
            shared_data[tid] = INT_MAX;
        }
    }
    __syncthreads();
    
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (find_max) {
                shared_data[tid] = max(shared_data[tid], shared_data[tid + stride]);
            } else {
                shared_data[tid] = min(shared_data[tid], shared_data[tid + stride]);
            }
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        partial_results[blockIdx.x] = shared_data[0];
    }
}

=======
/**
 * @brief Verifica si CUDA está disponible en el sistema
 * 
 * @return true si hay dispositivos CUDA disponibles, false en caso contrario
 */
>>>>>>> feature/fase1-despegues
bool checkCudaAvailability() {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    
    if (error != cudaSuccess || device_count == 0) {
        std::cerr << "\n╔═══════════════════════════════════════════════════════════╗\n";
        std::cerr << "║                  ADVERTENCIA: CUDA                        ║\n";
        std::cerr << "╚═══════════════════════════════════════════════════════════╝\n";
        std::cerr << "  No se detectaron dispositivos CUDA en el sistema.\n";
        std::cerr << "  El programa continuará pero las funcionalidades GPU\n";
        std::cerr << "  no estarán disponibles.\n\n";
        return false;
    }
    
    // Mostrar información del dispositivo
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
<<<<<<< HEAD
    std::cout << "\n=== Dispositivo GPU ===\n";
    std::cout << "Nombre: " << prop.name << "\n";
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Memoria Global: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB\n";
    std::cout << "Max hilos por bloque: " << prop.maxThreadsPerBlock << "\n";
    std::cout << "Max bloques (dim X): " << prop.maxGridSize[0] << "\n";
    std::cout << "Multiprocesadores (SM): " << prop.multiProcessorCount << "\n\n";
=======
    std::cout << "\n╔═══════════════════════════════════════════════════════════╗\n";
    std::cout << "║                INFORMACIÓN DEL DISPOSITIVO GPU            ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════╝\n";
    std::cout << "  Dispositivo: " << prop.name << "\n";
    std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "  Memoria Global: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB\n";
    std::cout << "  Multiprocessors: " << prop.multiProcessorCount << "\n";
    std::cout << "  CUDA Cores: ~" << (prop.multiProcessorCount * 128) << " (aproximado)\n";
    std::cout << "═════════════════════════════════════════════════════════════\n\n";
>>>>>>> feature/fase1-despegues
    
    return true;
}

// ============================================================================
// FASE 01: Análisis de Retrasos en Despegues (DEP_DELAY)
// ============================================================================

/**
 * @brief Kernel CUDA para analizar retrasos en despegues
 * 
 * Cada hilo procesa un vuelo. Si el retraso supera el umbral (positivo o negativo),
 * imprime el identificador global del hilo y el valor del retraso.
 * 
 * @param dep_delay Array con los retrasos en despegue (en minutos)
 * @param n Número total de vuelos
 * @param threshold Umbral de retraso (en minutos)
 */
__global__ void analyzeDepDelayKernel(const float* dep_delay, int n, float threshold) {
    // Calcular el identificador global del hilo (acceso linealizado 1D)
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Verificar que el hilo esté dentro del rango válido
    if (global_id < n) {
        float delay = dep_delay[global_id];
        
        // Verificar si el retraso supera el umbral (positivo) o es inferior (negativo/adelanto)
        // Si threshold > 0: detecta retrasos mayores a threshold
        // Si threshold < 0: detecta adelantos mayores al valor absoluto
        if ((threshold >= 0 && delay > threshold) || (threshold < 0 && delay < threshold)) {
            // Imprimir desde la GPU el ID global y el valor del retraso
            printf("Hilo %d: Retraso = %.2f minutos\n", global_id, delay);
        }
    }
}

/**
 * @brief Función host que gestiona la transferencia de datos y ejecución del kernel
 * para el análisis de retrasos en despegues.
 * 
 * @param dep_delay Vector con los retrasos en despegue (host)
 * @param threshold Umbral de retraso para filtrar (en minutos)
 */
void analyzeDepDelayGPU(const std::vector<float>& dep_delay, float threshold) {
    int n = dep_delay.size();
    
    if (n == 0) {
        std::cout << "No hay datos para procesar.\n";
        return;
    }
    
    std::cout << "\n=== Iniciando análisis en GPU ===\n";
    std::cout << "Total de vuelos: " << n << "\n";
    std::cout << "Umbral de retraso: " << threshold << " minutos\n\n";
    
    // Obtener propiedades del dispositivo para dimensionamiento dinámico
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    // Dimensionamiento dinámico basado en las características de la GPU
    // Usamos 256 hilos por bloque (múltiplo de 32/warp) para eficiencia
    int threads_per_block = 256;
    
    // Calcular número de bloques necesarios
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    
    std::cout << "Configuración de la GPU:\n";
    std::cout << "  - Dispositivo: " << prop.name << "\n";
    std::cout << "  - Bloques (N): " << num_blocks << "\n";
    std::cout << "  - Hilos por bloque (M): " << threads_per_block << "\n";
    std::cout << "  - Total de hilos lanzados: " << (num_blocks * threads_per_block) << "\n\n";
    
    // Reservar memoria en el dispositivo (GPU)
    float* d_dep_delay = nullptr;
    size_t size = n * sizeof(float);
    
    cudaError_t err = cudaMalloc((void**)&d_dep_delay, size);
    if (err != cudaSuccess) {
        std::cerr << "ERROR: cudaMalloc falló - " << cudaGetErrorString(err) << "\n";
        return;
    }
    
    // Copiar datos del host (CPU) al device (GPU)
    err = cudaMemcpy(d_dep_delay, dep_delay.data(), size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "ERROR: cudaMemcpy (Host->Device) falló - " << cudaGetErrorString(err) << "\n";
        cudaFree(d_dep_delay);
        return;
    }
    
    std::cout << "--- Resultados del kernel (salida desde GPU) ---\n\n";
    
    // Lanzar el kernel con configuración dinámica N bloques x M hilos
    analyzeDepDelayKernel<<<num_blocks, threads_per_block>>>(d_dep_delay, n, threshold);
    
    // Sincronizar para esperar que todos los hilos terminen
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "\nERROR: Kernel execution failed - " << cudaGetErrorString(err) << "\n";
        cudaFree(d_dep_delay);
        return;
    }
    
    std::cout << "\n--- Fin de la salida del kernel ---\n\n";
    std::cout << "Análisis completado exitosamente.\n";
    
    // Liberar memoria del dispositivo
    cudaFree(d_dep_delay);
}

/**
 * @brief Función principal
 * 
 * Flujo del programa:
 *   1. Verificar disponibilidad de CUDA
 *   2. Solicitar ruta del archivo CSV
 *   3. Cargar y parsear el dataset
 *   4. Mostrar estadísticas
 *   5. Ejecutar menú interactivo
 * 
 * @return Código de salida (0 = éxito, 1 = error)
 */
int main() {
    std::cout << "\n";
    std::cout << "███████╗██╗     ██╗ ██████╗ ██╗  ██╗████████╗███████╗\n";
    std::cout << "██╔════╝██║     ██║██╔════╝ ██║  ██║╚══██╔══╝██╔════╝\n";
    std::cout << "█████╗  ██║     ██║██║  ███╗███████║   ██║   ███████╗\n";
    std::cout << "██╔══╝  ██║     ██║██║   ██║██╔══██║   ██║   ╚════██║\n";
    std::cout << "██║     ███████╗██║╚██████╔╝██║  ██║   ██║   ███████║\n";
    std::cout << "╚═╝     ╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚══════╝\n";
    std::cout << "           ANÁLISIS DE DATASET (C++ & CUDA)            \n";
    std::cout << "═════════════════════════════════════════════════════════════\n";
    
    // Paso 1: Verificar disponibilidad de CUDA
    bool cuda_available = checkCudaAvailability();
    if (!cuda_available) {
        std::cout << "Presione Enter para continuar...";
        std::cin.get();
    }
    
    // Paso 2: Crear instancia del menú y solicitar ruta del archivo
    Menu menu;
    std::string csv_path = menu.promptForCSVPath(DEFAULT_CSV_PATH);
    
    // Paso 3: Crear parser y dataset
    CSVParser parser(csv_path);
    FlightDataset dataset;
    
    // Verificar que el archivo existe antes de intentar parsearlo
    if (!parser.fileExists()) {
        std::cerr << "\nERROR: El archivo especificado no existe.\n";
        std::cerr << "Ruta: " << csv_path << "\n\n";
        std::cerr << "Por favor, verifica la ruta e intenta nuevamente.\n";
        return 1;
    }
    
    // Parsear el archivo CSV
    std::cout << "\n";
    if (!parser.parse(dataset)) {
        std::cerr << "\nERROR: Fallo al cargar el dataset.\n";
        std::cerr << "Verifica que el archivo tenga el formato correcto.\n";
        return 1;
    }
    
    dataset.printStats();
    
    menu.setDataset(&dataset);
    
    std::cout << "Iniciando menú interactivo...\n";
    std::cout << "Presione Enter para continuar...";
    std::cin.get();
    
    int exit_code = menu.run();
    
    if (cuda_available) {
        cudaDeviceReset();
    }
    
    std::cout << "\n¡Programa finalizado correctamente!\n\n";
    
    return exit_code;
}
