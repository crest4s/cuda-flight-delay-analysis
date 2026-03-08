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
