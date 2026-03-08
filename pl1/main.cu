#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include "include/FlightDataset.h"
#include "include/CSVParser.h"
#include "include/Menu.h"

const std::string DEFAULT_CSV_PATH = "./data/Airline_dataset.csv";
const int MAX_TAIL_NUM_LENGTH = 20;

// Memoria constante para el umbral (threshold)
__constant__ float d_threshold;

// Kernel para analizar retrasos en salida (DEP_DELAY)
__global__ void analyzeDepDelayKernel(const float* dep_delays, const char* tail_nums, 
                                       int num_records, bool delay_type, int* counter,
                                       char* output_tail_nums, float* output_delays) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_records) {
        float delay = dep_delays[idx];
        
        // Verificar que no sea NaN
        if (!isnan(delay)) {
            bool condition_met = false;
            
            if (delay_type) {
                // Retraso positivo (vuelo sale tarde) - usando memoria constante
                condition_met = (delay >= d_threshold);
            } else {
                // Adelanto negativo (vuelo sale temprano) - usando memoria constante
                condition_met = (delay <= d_threshold);
            }
            
            if (condition_met) {
                // Operación atómica para obtener índice único
                int pos = atomicAdd(counter, 1);
                
                // Guardar matrícula en el array de salida
                for (int i = 0; i < MAX_TAIL_NUM_LENGTH; i++) {
                    output_tail_nums[pos * MAX_TAIL_NUM_LENGTH + i] = tail_nums[idx * MAX_TAIL_NUM_LENGTH + i];
                }
                
                // Guardar delay en el array de salida
                output_delays[pos] = delay;
            }
        }
    }
}

// Kernel para analizar retrasos en llegada (ARR_DELAY)
__global__ void analyzeArrDelayKernel(const float* arr_delays, const char* tail_nums,
                                       int num_records, bool delay_type, int* counter,
                                       char* output_tail_nums, float* output_delays) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_records) {
        float delay = arr_delays[idx];
        
        // Verificar que no sea NaN
        if (!isnan(delay)) {
            bool condition_met = false;
            
            if (delay_type) {
                // Retraso positivo (vuelo llega tarde) - usando memoria constante
                condition_met = (delay >= d_threshold);
            } else {
                // Adelanto negativo (vuelo llega temprano) - usando memoria constante
                condition_met = (delay <= d_threshold);
            }
            
            if (condition_met) {
                // Operación atómica para obtener índice único
                int pos = atomicAdd(counter, 1);
                
                // Guardar matrícula en el array de salida
                for (int i = 0; i < MAX_TAIL_NUM_LENGTH; i++) {
                    output_tail_nums[pos * MAX_TAIL_NUM_LENGTH + i] = tail_nums[idx * MAX_TAIL_NUM_LENGTH + i];
                }
                
                // Guardar delay en el array de salida
                output_delays[pos] = delay;
                
                // Imprimir desde GPU (requisito de la Fase 02)
                // Calcular el puntero al inicio de la matrícula para este registro
                const char* tail_num_ptr = &tail_nums[idx * MAX_TAIL_NUM_LENGTH];
                
                if (delay_type) {
                    printf("Hilo #%d | Matricula: %.10s | Retraso (llegada): %.0f min\n", 
                           idx, tail_num_ptr, delay);
                } else {
                    printf("Hilo #%d | Matricula: %.10s | Adelanto (llegada): %.0f min\n", 
                           idx, tail_num_ptr, -delay);
                }
            }
        }
    }
}

// ============================================================================
// FASE 04: KERNEL DE HISTOGRAMA DE AEROPUERTOS
// ============================================================================

// [4.1] Kernel básico: Solo memoria global con operaciones atómicas
__global__ void buildAirportHistogramKernel(const int* airport_ids, int* histogram, 
                                             int num_records, int max_airport_id) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_records) {
        int airport_id = airport_ids[idx];
        
        // Verificar que el ID sea válido (no cero, ya que 0 indica dato faltante)
        if (airport_id > 0 && airport_id <= max_airport_id) {
            // Incrementar el contador para este aeropuerto usando operación atómica
            atomicAdd(&histogram[airport_id], 1);
        }
    }
}

// [4.2] Kernel optimizado: Memoria compartida + memoria global
// Cada bloque construye su histograma local en shared memory (más rápido)
// y luego lo combina con el histograma global
__global__ void buildAirportHistogramSharedKernel(const int* airport_ids, int* histogram, 
                                                   int num_records, int max_airport_id) {
    // Memoria compartida para histograma local del bloque
    extern __shared__ int shared_histogram[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Fase 1: Inicializar histograma compartido a cero
    // Cada hilo inicializa múltiples bins si es necesario
    for (int i = tid; i <= max_airport_id; i += blockDim.x) {
        shared_histogram[i] = 0;
    }
    
    __syncthreads();
    
    // Fase 2: Construir histograma local en memoria compartida
    if (idx < num_records) {
        int airport_id = airport_ids[idx];
        
        if (airport_id > 0 && airport_id <= max_airport_id) {
            // Operación atómica en memoria compartida (mucho más rápida)
            atomicAdd(&shared_histogram[airport_id], 1);
        }
    }
    
    __syncthreads();
    
    // Fase 3: Combinar histograma local con el global
    // Cada hilo copia múltiples bins si es necesario
    for (int i = tid; i <= max_airport_id; i += blockDim.x) {
        if (shared_histogram[i] > 0) {
            // Solo una operación atómica por bin por bloque en memoria global
            atomicAdd(&histogram[i], shared_histogram[i]);
        }
    }
}

// [4.3] Kernel híbrido: Para histogramas muy grandes que no caben en shared memory
// Usa estrategia de privatización parcial
__global__ void buildAirportHistogramHybridKernel(const int* airport_ids, int* histogram, 
                                                   int num_records, int max_airport_id,
                                                   int bins_per_block) {
    // Memoria compartida limitada para los bins más comunes
    extern __shared__ int shared_histogram[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Inicializar shared memory
    for (int i = tid; i < bins_per_block; i += blockDim.x) {
        shared_histogram[i] = 0;
    }
    
    __syncthreads();
    
    // Procesar registro
    if (idx < num_records) {
        int airport_id = airport_ids[idx];
        
        if (airport_id > 0 && airport_id <= max_airport_id) {
            // Si el ID cabe en el rango de shared memory, usarla
            if (airport_id < bins_per_block) {
                atomicAdd(&shared_histogram[airport_id], 1);
            } else {
                // Si no cabe, ir directo a memoria global
                atomicAdd(&histogram[airport_id], 1);
            }
        }
    }
    
    __syncthreads();
    
    // Copiar shared memory a global memory
    for (int i = tid; i < bins_per_block; i += blockDim.x) {
        if (shared_histogram[i] > 0) {
            atomicAdd(&histogram[i], shared_histogram[i]);
        }
    }
}

// ============================================================================
// FASE 03: KERNELS DE REDUCCION MAX/MIN
// ============================================================================

// [3.1. Simple] Kernel simple: cada hilo revisa su posición y aplica operación atómica
__global__ void reduceSimpleKernel(const float* data, int* result, int n, bool find_max) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float value = data[idx];
        
        // Verificar que no sea NaN
        if (!isnan(value)) {
            int int_value = static_cast<int>(value); // Truncar a entero
            
            if (find_max) {
                atomicMax(result, int_value);
            } else {
                atomicMin(result, int_value);
            }
        }
    }
}

// [3.2. Básica] Kernel básico: cada hilo mira 3 posiciones (anterior, actual, posterior)
__global__ void reduceBasicKernel(const float* data, int* result, int n, bool find_max) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Memoria compartida para el bloque (tamaño dinámico)
    extern __shared__ float shared_data[];
    
    // Cargar datos a memoria compartida
    if (idx < n) {
        shared_data[threadIdx.x] = data[idx];
    } else {
        // Valores fuera de rango se establecen como NaN
        shared_data[threadIdx.x] = NAN;
    }
    
    __syncthreads();
    
    if (idx < n) {
        float current = shared_data[threadIdx.x];
        
        // Solo procesar si el valor actual es válido
        if (!isnan(current)) {
            float local_value = current;
            
            // Mirar valor anterior (si existe y está en el mismo bloque)
            if (threadIdx.x > 0) {
                float prev = shared_data[threadIdx.x - 1];
                if (!isnan(prev)) {
                    if (find_max) {
                        local_value = (prev > local_value) ? prev : local_value;
                    } else {
                        local_value = (prev < local_value) ? prev : local_value;
                    }
                }
            }
            
            // Mirar valor posterior (si existe y está en el mismo bloque)
            if (threadIdx.x < blockDim.x - 1 && idx + 1 < n) {
                float next = shared_data[threadIdx.x + 1];
                if (!isnan(next)) {
                    if (find_max) {
                        local_value = (next > local_value) ? next : local_value;
                    } else {
                        local_value = (next < local_value) ? next : local_value;
                    }
                }
            }
            
            // Guardar resultado de forma atómica en memoria global
            int int_value = static_cast<int>(local_value);
            if (find_max) {
                atomicMax(result, int_value);
            } else {
                atomicMin(result, int_value);
            }
        }
    }
}

// [3.3. Intermedia] Kernel intermedio: cada hilo mira 3 posiciones y luego hilos pares comparan
__global__ void reduceIntermediateKernel(const float* data, int* result, int n, bool find_max) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Memoria compartida para el bloque
    extern __shared__ float shared_data[];
    
    // Cargar datos a memoria compartida
    if (idx < n) {
        shared_data[threadIdx.x] = data[idx];
    } else {
        shared_data[threadIdx.x] = NAN;
    }
    
    __syncthreads();
    
    // Fase 1: Cada hilo mira 3 posiciones y guarda el resultado en memoria compartida
    if (idx < n) {
        float current = shared_data[threadIdx.x];
        
        if (!isnan(current)) {
            float local_value = current;
            
            // Mirar valor anterior
            if (threadIdx.x > 0) {
                float prev = shared_data[threadIdx.x - 1];
                if (!isnan(prev)) {
                    if (find_max) {
                        local_value = (prev > local_value) ? prev : local_value;
                    } else {
                        local_value = (prev < local_value) ? prev : local_value;
                    }
                }
            }
            
            // Mirar valor posterior
            if (threadIdx.x < blockDim.x - 1 && idx + 1 < n) {
                float next = shared_data[threadIdx.x + 1];
                if (!isnan(next)) {
                    if (find_max) {
                        local_value = (next > local_value) ? next : local_value;
                    } else {
                        local_value = (next < local_value) ? next : local_value;
                    }
                }
            }
            
            // Guardar en memoria compartida
            shared_data[threadIdx.x] = local_value;
        }
    }
    
    __syncthreads();
    
    // Fase 2: Los hilos con ID par comparan su valor con el siguiente
    if (idx < n && threadIdx.x % 2 == 0) {
        float value = shared_data[threadIdx.x];
        
        if (!isnan(value)) {
            float compare_value = value;
            
            // Comparar con el siguiente si existe
            if (threadIdx.x + 1 < blockDim.x && idx + 1 < n) {
                float next = shared_data[threadIdx.x + 1];
                if (!isnan(next)) {
                    if (find_max) {
                        compare_value = (next > compare_value) ? next : compare_value;
                    } else {
                        compare_value = (next < compare_value) ? next : compare_value;
                    }
                }
            }
            
            // Guardar resultado de forma atómica en memoria global
            int int_value = static_cast<int>(compare_value);
            if (find_max) {
                atomicMax(result, int_value);
            } else {
                atomicMin(result, int_value);
            }
        }
    }
}

// [3.4. Patrón de Reducción] Kernel usando patrón de reducción con árbol
__global__ void reduceTreeKernel(const float* data, int* partial_results, int n, bool find_max) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Memoria compartida para el bloque
    extern __shared__ float shared_data[];
    
    // Cargar datos a memoria compartida
    if (idx < n) {
        float value = data[idx];
        shared_data[threadIdx.x] = isnan(value) ? (find_max ? -2147483648.0f : 2147483647.0f) : value;
    } else {
        // Valores fuera de rango: usar valores extremos
        shared_data[threadIdx.x] = find_max ? -2147483648.0f : 2147483647.0f;
    }
    
    __syncthreads();
    
    // Patrón de reducción con árbol (tree reduction)
    // Cada iteración reduce a la mitad el número de hilos activos
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            float current = shared_data[threadIdx.x];
            float other = shared_data[threadIdx.x + stride];
            
            if (find_max) {
                shared_data[threadIdx.x] = (other > current) ? other : current;
            } else {
                shared_data[threadIdx.x] = (other < current) ? other : current;
            }
        }
        __syncthreads();
    }
    
    // El hilo 0 escribe el resultado parcial de este bloque
    if (threadIdx.x == 0) {
        partial_results[blockIdx.x] = static_cast<int>(shared_data[0]);
    }
}

void calculateOptimalDimensions(int num_records, int& blocks, int& threads_per_block) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    // Seleccionar número de hilos por bloque basado en las capacidades del hardware
    threads_per_block = (prop.maxThreadsPerBlock >= 512) ? 256 : 128;
    
    // Calcular número de bloques necesarios
    blocks = (num_records + threads_per_block - 1) / threads_per_block;
    
    // Verificar que no se exceda el límite máximo de bloques
    if (blocks > prop.maxGridSize[0]) {
        blocks = prop.maxGridSize[0];
    }
    
    // Mostrar información de la ejecución
    std::cout << "\n=== Configuracion de Ejecucion CUDA ===\n";
    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Max Threads por Bloque: " << prop.maxThreadsPerBlock << "\n";
    std::cout << "Configuracion: " << blocks << " bloques x " << threads_per_block << " hilos\n";
    std::cout << "Total de hilos: " << (blocks * threads_per_block) << "\n\n";
}
// Función wrapper para análisis de DEP_DELAY
void executeDepDelayAnalysis(const FlightDataset& dataset, float threshold, bool delay_type) {
    size_t num_records = dataset.size();
    if (num_records == 0) return;
    
    int blocks, threads_per_block;
    calculateOptimalDimensions(num_records, blocks, threads_per_block);
    
    const std::vector<float>& dep_delays = dataset.getDepDelay();
    const std::vector<std::string>& tail_nums = dataset.getTailNum();
    
    // Preparar array de caracteres para TAIL_NUM
    char* h_tail_nums = new char[num_records * MAX_TAIL_NUM_LENGTH];
    memset(h_tail_nums, 0, num_records * MAX_TAIL_NUM_LENGTH);
    
    for (size_t i = 0; i < num_records; i++) {
        strncpy(&h_tail_nums[i * MAX_TAIL_NUM_LENGTH], tail_nums[i].c_str(), MAX_TAIL_NUM_LENGTH - 1);
    }
    
    // Copiar threshold a memoria constante
    cudaError_t err = cudaMemcpyToSymbol(d_threshold, &threshold, sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "ERROR: cudaMemcpyToSymbol failed\n";
        delete[] h_tail_nums;
        return;
    }
    
    // Alocar memoria en GPU
    float* d_dep_delays = nullptr;
    char* d_tail_nums = nullptr;
    int* d_counter = nullptr;
    char* d_output_tail_nums = nullptr;
    float* d_output_delays = nullptr;
    
    int h_counter = 0;
    
    err = cudaMalloc(&d_dep_delays, num_records * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "ERROR: cudaMalloc failed\n";
        delete[] h_tail_nums;
        return;
    }
    
    err = cudaMalloc(&d_tail_nums, num_records * MAX_TAIL_NUM_LENGTH);
    if (err != cudaSuccess) {
        std::cerr << "ERROR: cudaMalloc failed\n";
        cudaFree(d_dep_delays);
        delete[] h_tail_nums;
        return;
    }
    
    err = cudaMalloc(&d_counter, sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "ERROR: cudaMalloc failed\n";
        cudaFree(d_dep_delays);
        cudaFree(d_tail_nums);
        delete[] h_tail_nums;
        return;
    }
    
    err = cudaMalloc(&d_output_tail_nums, num_records * MAX_TAIL_NUM_LENGTH);
    if (err != cudaSuccess) {
        std::cerr << "ERROR: cudaMalloc failed\n";
        cudaFree(d_dep_delays);
        cudaFree(d_tail_nums);
        cudaFree(d_counter);
        delete[] h_tail_nums;
        return;
    }
    
    err = cudaMalloc(&d_output_delays, num_records * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "ERROR: cudaMalloc failed\n";
        cudaFree(d_dep_delays);
        cudaFree(d_tail_nums);
        cudaFree(d_counter);
        cudaFree(d_output_tail_nums);
        delete[] h_tail_nums;
        return;
    }
    
    // Inicializar contador a 0
    err = cudaMemcpy(d_counter, &h_counter, sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "ERROR: cudaMemcpy failed\n";
        cudaFree(d_dep_delays);
        cudaFree(d_tail_nums);
        cudaFree(d_counter);
        cudaFree(d_output_tail_nums);
        cudaFree(d_output_delays);
        delete[] h_tail_nums;
        return;
    }
    
    // Copiar datos de entrada a GPU
    err = cudaMemcpy(d_dep_delays, dep_delays.data(), num_records * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "ERROR: cudaMemcpy failed\n";
        cudaFree(d_dep_delays);
        cudaFree(d_tail_nums);
        cudaFree(d_counter);
        cudaFree(d_output_tail_nums);
        cudaFree(d_output_delays);
        delete[] h_tail_nums;
        return;
    }
    
    err = cudaMemcpy(d_tail_nums, h_tail_nums, num_records * MAX_TAIL_NUM_LENGTH, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "ERROR: cudaMemcpy failed\n";
        cudaFree(d_dep_delays);
        cudaFree(d_tail_nums);
        cudaFree(d_counter);
        cudaFree(d_output_tail_nums);
        cudaFree(d_output_delays);
        delete[] h_tail_nums;
        return;
    }
    
    // Lanzar kernel
    analyzeDepDelayKernel<<<blocks, threads_per_block>>>(d_dep_delays, d_tail_nums, num_records, 
                                                          delay_type, d_counter, 
                                                          d_output_tail_nums, d_output_delays);
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "ERROR: kernel execution failed\n";
        cudaFree(d_dep_delays);
        cudaFree(d_tail_nums);
        cudaFree(d_counter);
        cudaFree(d_output_tail_nums);
        cudaFree(d_output_delays);
        delete[] h_tail_nums;
        return;
    }
    
    // Copiar resultado de vuelta al host
    err = cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "ERROR: cudaMemcpy failed\n";
        cudaFree(d_dep_delays);
        cudaFree(d_tail_nums);
        cudaFree(d_counter);
        cudaFree(d_output_tail_nums);
        cudaFree(d_output_delays);
        delete[] h_tail_nums;
        return;
    }
    
    std::cout << "\n=== RESULTADOS DEP_DELAY ===\n";
    std::cout << "Total de vuelos que cumplen la condicion: " << h_counter << "\n\n";
    
    if (h_counter > 0) {
        // Alocar memoria en host para resultados
        char* h_output_tail_nums = new char[h_counter * MAX_TAIL_NUM_LENGTH];
        float* h_output_delays = new float[h_counter];
        
        // Copiar arrays de salida
        err = cudaMemcpy(h_output_tail_nums, d_output_tail_nums, 
                        h_counter * MAX_TAIL_NUM_LENGTH, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "ERROR: cudaMemcpy failed\n";
        }
        
        err = cudaMemcpy(h_output_delays, d_output_delays, 
                        h_counter * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "ERROR: cudaMemcpy failed\n";
        }
        
        // Imprimir resultados desde CPU
        std::cout << "MATRICULA (TAIL_NUM)\tDEP_DELAY (min)\n";
        std::cout << "----------------------------------------\n";
        
        int max_print = (h_counter > 20) ? 20 : h_counter;
        for (int i = 0; i < max_print; i++) {
            char tail_num[MAX_TAIL_NUM_LENGTH];
            strncpy(tail_num, &h_output_tail_nums[i * MAX_TAIL_NUM_LENGTH], MAX_TAIL_NUM_LENGTH - 1);
            tail_num[MAX_TAIL_NUM_LENGTH - 1] = '\0';
            std::cout << tail_num << "\t\t" << h_output_delays[i] << "\n";
        }
        
        if (h_counter > 20) {
            std::cout << "... (" << (h_counter - 20) << " resultados mas)\n";
        }
        
        delete[] h_output_tail_nums;
        delete[] h_output_delays;
    }
    
    // Liberar memoria
    cudaFree(d_dep_delays);
    cudaFree(d_tail_nums);
    cudaFree(d_counter);
    cudaFree(d_output_tail_nums);
    cudaFree(d_output_delays);
    delete[] h_tail_nums;
}

// Función wrapper para análisis de ARR_DELAY
void executeArrDelayAnalysis(const FlightDataset& dataset, float threshold, bool delay_type) {
    size_t num_records = dataset.size();
    if (num_records == 0) return;
    
    int blocks, threads_per_block;
    calculateOptimalDimensions(num_records, blocks, threads_per_block);
    
    const std::vector<float>& arr_delays = dataset.getArrDelay();
    const std::vector<std::string>& tail_nums = dataset.getTailNum();
    
    // Preparar array de caracteres para TAIL_NUM
    char* h_tail_nums = new char[num_records * MAX_TAIL_NUM_LENGTH];
    memset(h_tail_nums, 0, num_records * MAX_TAIL_NUM_LENGTH);
    
    for (size_t i = 0; i < num_records; i++) {
        strncpy(&h_tail_nums[i * MAX_TAIL_NUM_LENGTH], tail_nums[i].c_str(), MAX_TAIL_NUM_LENGTH - 1);
    }
    
    // Copiar threshold a memoria constante
    cudaError_t err = cudaMemcpyToSymbol(d_threshold, &threshold, sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "ERROR: cudaMemcpyToSymbol failed\n";
        delete[] h_tail_nums;
        return;
    }
    
    // Alocar memoria en GPU
    float* d_arr_delays = nullptr;
    char* d_tail_nums = nullptr;
    int* d_counter = nullptr;
    char* d_output_tail_nums = nullptr;
    float* d_output_delays = nullptr;
    
    int h_counter = 0;
    
    err = cudaMalloc(&d_arr_delays, num_records * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "ERROR: cudaMalloc failed\n";
        delete[] h_tail_nums;
        return;
    }
    
    err = cudaMalloc(&d_tail_nums, num_records * MAX_TAIL_NUM_LENGTH);
    if (err != cudaSuccess) {
        std::cerr << "ERROR: cudaMalloc failed\n";
        cudaFree(d_arr_delays);
        delete[] h_tail_nums;
        return;
    }
    
    err = cudaMalloc(&d_counter, sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "ERROR: cudaMalloc failed\n";
        cudaFree(d_arr_delays);
        cudaFree(d_tail_nums);
        delete[] h_tail_nums;
        return;
    }
    
    err = cudaMalloc(&d_output_tail_nums, num_records * MAX_TAIL_NUM_LENGTH);
    if (err != cudaSuccess) {
        std::cerr << "ERROR: cudaMalloc failed\n";
        cudaFree(d_arr_delays);
        cudaFree(d_tail_nums);
        cudaFree(d_counter);
        delete[] h_tail_nums;
        return;
    }
    
    err = cudaMalloc(&d_output_delays, num_records * sizeof(float));
    if (err != cudaSuccess) {
        std::cerr << "ERROR: cudaMalloc failed\n";
        cudaFree(d_arr_delays);
        cudaFree(d_tail_nums);
        cudaFree(d_counter);
        cudaFree(d_output_tail_nums);
        delete[] h_tail_nums;
        return;
    }
    
    // Inicializar contador a 0
    err = cudaMemcpy(d_counter, &h_counter, sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "ERROR: cudaMemcpy failed\n";
        cudaFree(d_arr_delays);
        cudaFree(d_tail_nums);
        cudaFree(d_counter);
        cudaFree(d_output_tail_nums);
        cudaFree(d_output_delays);
        delete[] h_tail_nums;
        return;
    }
    
    // Copiar datos de entrada a GPU
    err = cudaMemcpy(d_arr_delays, arr_delays.data(), num_records * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "ERROR: cudaMemcpy failed\n";
        cudaFree(d_arr_delays);
        cudaFree(d_tail_nums);
        cudaFree(d_counter);
        cudaFree(d_output_tail_nums);
        cudaFree(d_output_delays);
        delete[] h_tail_nums;
        return;
    }
    
    err = cudaMemcpy(d_tail_nums, h_tail_nums, num_records * MAX_TAIL_NUM_LENGTH, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "ERROR: cudaMemcpy failed\n";
        cudaFree(d_arr_delays);
        cudaFree(d_tail_nums);
        cudaFree(d_counter);
        cudaFree(d_output_tail_nums);
        cudaFree(d_output_delays);
        delete[] h_tail_nums;
        return;
    }
    
    // Lanzar kernel
    analyzeArrDelayKernel<<<blocks, threads_per_block>>>(d_arr_delays, d_tail_nums, num_records, 
                                                          delay_type, d_counter, 
                                                          d_output_tail_nums, d_output_delays);
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "ERROR: kernel execution failed\n";
        cudaFree(d_arr_delays);
        cudaFree(d_tail_nums);
        cudaFree(d_counter);
        cudaFree(d_output_tail_nums);
        cudaFree(d_output_delays);
        delete[] h_tail_nums;
        return;
    }
    
    // Copiar resultado de vuelta al host
    err = cudaMemcpy(&h_counter, d_counter, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        std::cerr << "ERROR: cudaMemcpy failed\n";
        cudaFree(d_arr_delays);
        cudaFree(d_tail_nums);
        cudaFree(d_counter);
        cudaFree(d_output_tail_nums);
        cudaFree(d_output_delays);
        delete[] h_tail_nums;
        return;
    }
    
    if (h_counter > 0) {
        // Alocar memoria en host para resultados
        char* h_output_tail_nums = new char[h_counter * MAX_TAIL_NUM_LENGTH];
        float* h_output_delays = new float[h_counter];
        
        // Copiar arrays de salida
        err = cudaMemcpy(h_output_tail_nums, d_output_tail_nums, 
                        h_counter * MAX_TAIL_NUM_LENGTH, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "ERROR: cudaMemcpy failed\n";
        }
        
        err = cudaMemcpy(h_output_delays, d_output_delays, 
                        h_counter * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "ERROR: cudaMemcpy failed\n";
        }
        
        // Imprimir resultados desde CPU (según formato de la Fase 02)
        std::cout << "\nResultados completados de calcular en la CPU:\n";
        std::cout << "Se han encontrado " << h_counter << " aviones\n\n";
        
        for (int i = 0; i < h_counter; i++) {
            char tail_num[MAX_TAIL_NUM_LENGTH];
            strncpy(tail_num, &h_output_tail_nums[i * MAX_TAIL_NUM_LENGTH], MAX_TAIL_NUM_LENGTH - 1);
            tail_num[MAX_TAIL_NUM_LENGTH - 1] = '\0';
            
            if (delay_type) {
                std::cout << "Matricula " << tail_num << " Retraso:" << static_cast<int>(h_output_delays[i]) << " minutos\n";
            } else {
                std::cout << "Matricula " << tail_num << " Adelanto:" << static_cast<int>(-h_output_delays[i]) << " minutos\n";
            }
        }
        
        delete[] h_output_tail_nums;
        delete[] h_output_delays;
    } else {
        std::cout << "\nNo se han encontrado aviones que cumplan con el criterio especificado.\n";
    }
    
    // Liberar memoria
    cudaFree(d_arr_delays);
    cudaFree(d_tail_nums);
    cudaFree(d_counter);
    cudaFree(d_output_tail_nums);
    cudaFree(d_output_delays);
    delete[] h_tail_nums;
}

// ============================================================================
// FUNCIONES WRAPPER PARA EJECUCION DE KERNELS DE REDUCCION
// ============================================================================

// Función wrapper para kernel simple
int executeReduceSimple(const std::vector<float>& data, bool find_max) {
    size_t n = data.size();
    if (n == 0) return 0;
    
    // Configurar dimensiones
    int blocks, threads_per_block;
    calculateOptimalDimensions(n, blocks, threads_per_block);
    
    // Alocar memoria en GPU
    float* d_data = nullptr;
    int* d_result = nullptr;
    
    cudaError_t err = cudaMalloc(&d_data, n * sizeof(float));
    if (err != cudaSuccess) return 0;
    
    err = cudaMalloc(&d_result, sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(d_data);
        return 0;
    }
    
    // Inicializar resultado en GPU con valor extremo
    int init_value = find_max ? -2147483648 : 2147483647;
    cudaMemcpy(d_result, &init_value, sizeof(int), cudaMemcpyHostToDevice);
    
    // Copiar datos a GPU
    cudaMemcpy(d_data, data.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    
    // Lanzar kernel
    reduceSimpleKernel<<<blocks, threads_per_block>>>(d_data, d_result, n, find_max);
    cudaDeviceSynchronize();
    
    // Copiar resultado de vuelta
    int result;
    cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    
    // Liberar memoria
    cudaFree(d_data);
    cudaFree(d_result);
    
    return result;
}

// Función wrapper para kernel básico
int executeReduceBasic(const std::vector<float>& data, bool find_max) {
    size_t n = data.size();
    if (n == 0) return 0;
    
    // Configurar dimensiones
    int blocks, threads_per_block;
    calculateOptimalDimensions(n, blocks, threads_per_block);
    
    // Alocar memoria en GPU
    float* d_data = nullptr;
    int* d_result = nullptr;
    
    cudaError_t err = cudaMalloc(&d_data, n * sizeof(float));
    if (err != cudaSuccess) return 0;
    
    err = cudaMalloc(&d_result, sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(d_data);
        return 0;
    }
    
    // Inicializar resultado en GPU con valor extremo
    int init_value = find_max ? -2147483648 : 2147483647;
    cudaMemcpy(d_result, &init_value, sizeof(int), cudaMemcpyHostToDevice);
    
    // Copiar datos a GPU
    cudaMemcpy(d_data, data.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    
    // Lanzar kernel con memoria compartida
    size_t shared_mem_size = threads_per_block * sizeof(float);
    reduceBasicKernel<<<blocks, threads_per_block, shared_mem_size>>>(d_data, d_result, n, find_max);
    cudaDeviceSynchronize();
    
    // Copiar resultado de vuelta
    int result;
    cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    
    // Liberar memoria
    cudaFree(d_data);
    cudaFree(d_result);
    
    return result;
}

// Función wrapper para kernel intermedio
int executeReduceIntermediate(const std::vector<float>& data, bool find_max) {
    size_t n = data.size();
    if (n == 0) return 0;
    
    // Configurar dimensiones
    int blocks, threads_per_block;
    calculateOptimalDimensions(n, blocks, threads_per_block);
    
    // Alocar memoria en GPU
    float* d_data = nullptr;
    int* d_result = nullptr;
    
    cudaError_t err = cudaMalloc(&d_data, n * sizeof(float));
    if (err != cudaSuccess) return 0;
    
    err = cudaMalloc(&d_result, sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(d_data);
        return 0;
    }
    
    // Inicializar resultado en GPU con valor extremo
    int init_value = find_max ? -2147483648 : 2147483647;
    cudaMemcpy(d_result, &init_value, sizeof(int), cudaMemcpyHostToDevice);
    
    // Copiar datos a GPU
    cudaMemcpy(d_data, data.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    
    // Lanzar kernel con memoria compartida
    size_t shared_mem_size = threads_per_block * sizeof(float);
    reduceIntermediateKernel<<<blocks, threads_per_block, shared_mem_size>>>(d_data, d_result, n, find_max);
    cudaDeviceSynchronize();
    
    // Copiar resultado de vuelta
    int result;
    cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    
    // Liberar memoria
    cudaFree(d_data);
    cudaFree(d_result);
    
    return result;
}

// Función wrapper para kernel de patrón de reducción
int executeReduceTree(const std::vector<float>& data, bool find_max) {
    size_t n = data.size();
    if (n == 0) return 0;
    
    const int MAX_PARTIAL_RESULTS = 10;
    
    // Configurar dimensiones iniciales
    int blocks, threads_per_block;
    calculateOptimalDimensions(n, blocks, threads_per_block);
    
    // Alocar memoria en GPU para datos originales
    float* d_data = nullptr;
    int* d_partial = nullptr;
    
    cudaError_t err = cudaMalloc(&d_data, n * sizeof(float));
    if (err != cudaSuccess) return 0;
    
    err = cudaMalloc(&d_partial, blocks * sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(d_data);
        return 0;
    }
    
    // Copiar datos a GPU
    cudaMemcpy(d_data, data.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    
    // Lanzar kernel con memoria compartida
    size_t shared_mem_size = threads_per_block * sizeof(float);
    reduceTreeKernel<<<blocks, threads_per_block, shared_mem_size>>>(d_data, d_partial, n, find_max);
    cudaDeviceSynchronize();
    
    // Copiar resultados parciales de vuelta
    std::vector<int> h_partial(blocks);
    cudaMemcpy(h_partial.data(), d_partial, blocks * sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(d_data);
    cudaFree(d_partial);
    
    // Si hay más de MAX_PARTIAL_RESULTS, hacer reducciones sucesivas
    while (h_partial.size() > MAX_PARTIAL_RESULTS) {
        blocks = (h_partial.size() + threads_per_block - 1) / threads_per_block;
        
        // Copiar parciales como float para el kernel
        std::vector<float> temp_data(h_partial.size());
        for (size_t i = 0; i < h_partial.size(); i++) {
            temp_data[i] = static_cast<float>(h_partial[i]);
        }
        
        // Alocar memoria para nueva reducción
        float* d_temp = nullptr;
        int* d_new_partial = nullptr;
        
        cudaMalloc(&d_temp, temp_data.size() * sizeof(float));
        cudaMalloc(&d_new_partial, blocks * sizeof(int));
        
        cudaMemcpy(d_temp, temp_data.data(), temp_data.size() * sizeof(float), cudaMemcpyHostToDevice);
        
        // Lanzar kernel de reducción
        reduceTreeKernel<<<blocks, threads_per_block, shared_mem_size>>>(
            d_temp, d_new_partial, temp_data.size(), find_max);
        cudaDeviceSynchronize();
        
        // Copiar nuevos resultados parciales
        h_partial.resize(blocks);
        cudaMemcpy(h_partial.data(), d_new_partial, blocks * sizeof(int), cudaMemcpyDeviceToHost);
        
        cudaFree(d_temp);
        cudaFree(d_new_partial);
    }
    
    // Reducción final en CPU
    int result = h_partial[0];
    for (size_t i = 1; i < h_partial.size(); i++) {
        if (find_max) {
            result = (h_partial[i] > result) ? h_partial[i] : result;
        } else {
            result = (h_partial[i] < result) ? h_partial[i] : result;
        }
    }
    
    return result;
}

// ============================================================================
// FUNCIONES WRAPPER PARA FASE 04: HISTOGRAMA DE AEROPUERTOS
// ============================================================================

void executeAirportHistogram(const FlightDataset& dataset, bool use_origin) {
    size_t num_records = dataset.size();
    if (num_records == 0) {
        std::cout << "No hay registros en el dataset.\n";
        return;
    }
    
    // Obtener los IDs de aeropuertos (origen o destino)
    const std::vector<int>& airport_ids = use_origin ? 
        dataset.getOriginSeqId() : dataset.getDestSeqId();
    
    // Encontrar el ID máximo para dimensionar el histograma
    int max_airport_id = 0;
    for (size_t i = 0; i < airport_ids.size(); i++) {
        if (airport_ids[i] > max_airport_id) {
            max_airport_id = airport_ids[i];
        }
    }
    
    if (max_airport_id == 0) {
        std::cout << "No hay datos validos de aeropuertos.\n";
        return;
    }
    
    std::cout << "\n=== Configuracion del Histograma ===\n";
    std::cout << "Registros: " << num_records << "\n";
    std::cout << "ID maximo de aeropuerto: " << max_airport_id << "\n";
    std::cout << "Tipo: " << (use_origin ? "Aeropuertos de origen" : "Aeropuertos de destino") << "\n";
    
    // Obtener propiedades de la GPU
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    // Configurar dimensiones de ejecución CUDA
    int blocks, threads_per_block;
    calculateOptimalDimensions(num_records, blocks, threads_per_block);
    
    // Calcular memoria compartida disponible y requerida
    size_t shared_mem_available = prop.sharedMemPerBlock;
    size_t shared_mem_required = (max_airport_id + 1) * sizeof(int);
    
    // Decidir estrategia de memoria basada en tamaño del histograma
    enum HistogramStrategy { GLOBAL_ONLY, SHARED_FULL, HYBRID };
    HistogramStrategy strategy;
    size_t shared_mem_size = 0;
    int bins_per_block = 0;
    
    if (shared_mem_required <= shared_mem_available * 0.8) {
        // Histograma cabe completamente en shared memory
        strategy = SHARED_FULL;
        shared_mem_size = shared_mem_required;
        std::cout << "Estrategia: Memoria compartida completa\n";
        std::cout << "Memoria compartida usada: " << (shared_mem_size / 1024) << " KB\n";
    } else if (max_airport_id > 10000) {
        // Histograma muy grande: estrategia híbrida
        strategy = HYBRID;
        // Usar el 80% de la memoria compartida disponible
        bins_per_block = static_cast<int>((shared_mem_available * 0.8) / sizeof(int));
        shared_mem_size = bins_per_block * sizeof(int);
        std::cout << "Estrategia: Hibrida (shared + global)\n";
        std::cout << "Bins en shared memory: " << bins_per_block << "\n";
        std::cout << "Bins en global memory: " << (max_airport_id - bins_per_block + 1) << "\n";
    } else {
        // Histograma mediano: usar solo memoria global
        strategy = GLOBAL_ONLY;
        std::cout << "Estrategia: Solo memoria global\n";
    }
    
    // Alocar memoria en GPU
    int* d_airport_ids = nullptr;
    int* d_histogram = nullptr;
    
    // Tamaño del histograma (max_airport_id + 1 para incluir índice 0)
    size_t histogram_size = (max_airport_id + 1) * sizeof(int);
    
    cudaError_t err = cudaMalloc(&d_airport_ids, num_records * sizeof(int));
    if (err != cudaSuccess) {
        std::cerr << "ERROR: cudaMalloc failed para airport_ids\n";
        return;
    }
    
    err = cudaMalloc(&d_histogram, histogram_size);
    if (err != cudaSuccess) {
        std::cerr << "ERROR: cudaMalloc failed para histogram\n";
        cudaFree(d_airport_ids);
        return;
    }
    
    // Inicializar histograma a cero
    cudaMemset(d_histogram, 0, histogram_size);
    
    // Copiar IDs de aeropuertos a GPU
    cudaMemcpy(d_airport_ids, airport_ids.data(), num_records * sizeof(int), 
               cudaMemcpyHostToDevice);
    
    // Lanzar kernel según estrategia seleccionada
    std::cout << "\nEjecutando kernel...\n";
    
    switch (strategy) {
        case SHARED_FULL:
            buildAirportHistogramSharedKernel<<<blocks, threads_per_block, shared_mem_size>>>(
                d_airport_ids, d_histogram, num_records, max_airport_id);
            break;
            
        case HYBRID:
            buildAirportHistogramHybridKernel<<<blocks, threads_per_block, shared_mem_size>>>(
                d_airport_ids, d_histogram, num_records, max_airport_id, bins_per_block);
            break;
            
        case GLOBAL_ONLY:
        default:
            buildAirportHistogramKernel<<<blocks, threads_per_block>>>(
                d_airport_ids, d_histogram, num_records, max_airport_id);
            break;
    }
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "ERROR: Kernel execution failed\n";
        cudaFree(d_airport_ids);
        cudaFree(d_histogram);
        return;
    }
    
    // Copiar histograma de vuelta a CPU
    std::vector<int> h_histogram(max_airport_id + 1);
    cudaMemcpy(h_histogram.data(), d_histogram, histogram_size, cudaMemcpyDeviceToHost);
    
    // Liberar memoria GPU
    cudaFree(d_airport_ids);
    cudaFree(d_histogram);
    
    // Mostrar resultados: encontrar los 10 aeropuertos más frecuentes
    std::cout << "\n=== Top 10 Aeropuertos mas Frecuentes ===\n";
    
    // Crear lista de pares (ID, frecuencia) para aeropuertos con al menos 1 vuelo
    std::vector<std::pair<int, int>> airport_freqs;
    for (int i = 1; i <= max_airport_id; i++) {
        if (h_histogram[i] > 0) {
            airport_freqs.push_back(std::make_pair(i, h_histogram[i]));
        }
    }
    
    // Ordenar por frecuencia descendente (burbuja simple para top 10)
    for (size_t i = 0; i < airport_freqs.size() && i < 10; i++) {
        for (size_t j = i + 1; j < airport_freqs.size(); j++) {
            if (airport_freqs[j].second > airport_freqs[i].second) {
                std::swap(airport_freqs[i], airport_freqs[j]);
            }
        }
    }
    
    // Mostrar top 10
    int count = 0;
    for (size_t i = 0; i < airport_freqs.size() && count < 10; i++) {
        std::cout << (count + 1) << ". Aeropuerto ID " << airport_freqs[i].first 
                  << ": " << airport_freqs[i].second << " vuelos\n";
        count++;
    }
    
    std::cout << "\nTotal de aeropuertos unicos: " << airport_freqs.size() << "\n";
}

bool checkCudaAvailability() {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    
    if (error != cudaSuccess || device_count == 0) {
        std::cerr << "\nNo se detectaron dispositivos CUDA\n";
        return false;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << " (" << (prop.totalGlobalMem / (1024 * 1024)) << " MB)\n\n";
    
    return true;
}

int main() {
    std::cout << "\nAnalisis de Vuelos (CUDA)\n";
    
    bool cuda_available = checkCudaAvailability();
    
    Menu menu;
    std::string csv_path = menu.promptForCSVPath(DEFAULT_CSV_PATH);
    
    CSVParser parser(csv_path);
    FlightDataset dataset;
    
    if (!parser.fileExists()) {
        std::cerr << "\nERROR: Archivo no existe: " << csv_path << "\n";
        return 1;
    }
    
    if (!parser.parse(dataset)) {
        std::cerr << "ERROR: Fallo al cargar dataset\n";
        return 1;
    }
    
    dataset.printStats();
    menu.setDataset(&dataset);
    
    std::cout << "\nEnter para continuar...";
    std::cin.get();
    
    int exit_code = menu.run();
    
    if (cuda_available) {
        cudaDeviceReset();
    }
    
    return exit_code;
}
