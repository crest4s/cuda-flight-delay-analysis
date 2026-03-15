#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <climits>
#include <cstring>
#include <algorithm>
#include <unordered_map>
#include <iomanip>
#include "include/FlightDataset.h"
#include "include/CSVParser.h"
#include "include/Menu.h"

const std::string DEFAULT_CSV_PATH = "./data/Airline_dataset.csv";
const int MAX_TAIL_NUM_LENGTH = 20;
const int PRINT_MAX_RESULTS = 20;
const int MAX_BAR_WIDTH = 50;

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
                int pos = atomicAdd(counter, 1);

                // Guardar matrícula y delay solo si pos está dentro del rango asignado
                if (pos < num_records) {
                    for (int i = 0; i < MAX_TAIL_NUM_LENGTH; i++) {
                        output_tail_nums[pos * MAX_TAIL_NUM_LENGTH + i] = tail_nums[idx * MAX_TAIL_NUM_LENGTH + i];
                    }
                    output_delays[pos] = delay;
                }
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
                int pos = atomicAdd(counter, 1);

                if (pos < num_records) {
                    for (int i = 0; i < MAX_TAIL_NUM_LENGTH; i++) {
                        output_tail_nums[pos * MAX_TAIL_NUM_LENGTH + i] = tail_nums[idx * MAX_TAIL_NUM_LENGTH + i];
                    }
                    output_delays[pos] = delay;
                }

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
        
        if (!isnan(value)) {
            int int_value = static_cast<int>(roundf(value));

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
            
            int int_value = static_cast<int>(roundf(local_value));
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
            
            int int_value = static_cast<int>(roundf(compare_value));
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
    
    if (idx < n) {
        float value = data[idx];
        shared_data[threadIdx.x] = isnan(value) ? (find_max ? (float)INT_MIN : (float)INT_MAX) : value;
    } else {
        shared_data[threadIdx.x] = find_max ? (float)INT_MIN : (float)INT_MAX;
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
        partial_results[blockIdx.x] = static_cast<int>(roundf(shared_data[0]));
    }
}

void calculateOptimalDimensions(int num_records, int& blocks, int& threads_per_block) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    // Seleccionar número de hilos por bloque basado en las capacidades del hardware
    threads_per_block = (prop.maxThreadsPerBlock >= 512) ? 256 : 128;
    
    // Calcular número de bloques necesarios
    blocks = (num_records + threads_per_block - 1) / threads_per_block;
    
    if (blocks > prop.maxGridSize[0]) {
        std::cerr << "ADVERTENCIA: Se reducen los bloques de " << blocks
                  << " a " << prop.maxGridSize[0] << " (limite del hardware). "
                  << "Algunos registros no seran procesados.\n";
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
        char* h_output_tail_nums = new char[h_counter * MAX_TAIL_NUM_LENGTH];
        float* h_output_delays = new float[h_counter];
        bool copy_ok = true;

        err = cudaMemcpy(h_output_tail_nums, d_output_tail_nums,
                        h_counter * MAX_TAIL_NUM_LENGTH, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "ERROR: cudaMemcpy failed (tail_nums)\n";
            copy_ok = false;
        }

        err = cudaMemcpy(h_output_delays, d_output_delays,
                        h_counter * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            std::cerr << "ERROR: cudaMemcpy failed (delays)\n";
            copy_ok = false;
        }

        if (copy_ok) {
            std::cout << "MATRICULA (TAIL_NUM)\tDEP_DELAY (min)\n";
            std::cout << "----------------------------------------\n";

            int max_print = (h_counter > PRINT_MAX_RESULTS) ? PRINT_MAX_RESULTS : h_counter;
            for (int i = 0; i < max_print; i++) {
                char tail_num[MAX_TAIL_NUM_LENGTH];
                strncpy(tail_num, &h_output_tail_nums[i * MAX_TAIL_NUM_LENGTH], MAX_TAIL_NUM_LENGTH - 1);
                tail_num[MAX_TAIL_NUM_LENGTH - 1] = '\0';
                std::cout << tail_num << "\t\t" << h_output_delays[i] << "\n";
            }

            if (h_counter > PRINT_MAX_RESULTS) {
                std::cout << "... (" << (h_counter - PRINT_MAX_RESULTS) << " resultados mas)\n";
            }
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
    
    int init_value = find_max ? INT_MIN : INT_MAX;
    err = cudaMemcpy(d_result, &init_value, sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { cudaFree(d_data); cudaFree(d_result); return 0; }

    err = cudaMemcpy(d_data, data.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { cudaFree(d_data); cudaFree(d_result); return 0; }

    reduceSimpleKernel<<<blocks, threads_per_block>>>(d_data, d_result, n, find_max);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { cudaFree(d_data); cudaFree(d_result); return 0; }

    int result;
    err = cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { cudaFree(d_data); cudaFree(d_result); return 0; }

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
    
    int init_value = find_max ? INT_MIN : INT_MAX;
    err = cudaMemcpy(d_result, &init_value, sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { cudaFree(d_data); cudaFree(d_result); return 0; }

    err = cudaMemcpy(d_data, data.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { cudaFree(d_data); cudaFree(d_result); return 0; }

    size_t shared_mem_size = threads_per_block * sizeof(float);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (shared_mem_size > prop.sharedMemPerBlock) {
        std::cerr << "ERROR: shared memory insuficiente para reduceBasicKernel\n";
        cudaFree(d_data); cudaFree(d_result); return 0;
    }

    reduceBasicKernel<<<blocks, threads_per_block, shared_mem_size>>>(d_data, d_result, n, find_max);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { cudaFree(d_data); cudaFree(d_result); return 0; }

    int result;
    err = cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { cudaFree(d_data); cudaFree(d_result); return 0; }

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
    
    int init_value = find_max ? INT_MIN : INT_MAX;
    err = cudaMemcpy(d_result, &init_value, sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { cudaFree(d_data); cudaFree(d_result); return 0; }

    err = cudaMemcpy(d_data, data.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { cudaFree(d_data); cudaFree(d_result); return 0; }

    size_t shared_mem_size = threads_per_block * sizeof(float);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (shared_mem_size > prop.sharedMemPerBlock) {
        std::cerr << "ERROR: shared memory insuficiente para reduceIntermediateKernel\n";
        cudaFree(d_data); cudaFree(d_result); return 0;
    }

    reduceIntermediateKernel<<<blocks, threads_per_block, shared_mem_size>>>(d_data, d_result, n, find_max);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { cudaFree(d_data); cudaFree(d_result); return 0; }

    int result;
    err = cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { cudaFree(d_data); cudaFree(d_result); return 0; }

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
    
    cudaError_t err = cudaMemcpy(d_data, data.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { cudaFree(d_data); cudaFree(d_partial); return 0; }

    size_t shared_mem_size = threads_per_block * sizeof(float);
    reduceTreeKernel<<<blocks, threads_per_block, shared_mem_size>>>(d_data, d_partial, n, find_max);
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { cudaFree(d_data); cudaFree(d_partial); return 0; }

    std::vector<int> h_partial(blocks);
    err = cudaMemcpy(h_partial.data(), d_partial, blocks * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { cudaFree(d_data); cudaFree(d_partial); return 0; }

    cudaFree(d_data);
    cudaFree(d_partial);

    while (h_partial.size() > (size_t)MAX_PARTIAL_RESULTS) {
        blocks = (h_partial.size() + threads_per_block - 1) / threads_per_block;

        std::vector<float> temp_data(h_partial.size());
        for (size_t i = 0; i < h_partial.size(); i++) {
            temp_data[i] = static_cast<float>(h_partial[i]);
        }

        float* d_temp = nullptr;
        int* d_new_partial = nullptr;

        err = cudaMalloc(&d_temp, temp_data.size() * sizeof(float));
        if (err != cudaSuccess) return 0;

        err = cudaMalloc(&d_new_partial, blocks * sizeof(int));
        if (err != cudaSuccess) { cudaFree(d_temp); return 0; }

        err = cudaMemcpy(d_temp, temp_data.data(), temp_data.size() * sizeof(float), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) { cudaFree(d_temp); cudaFree(d_new_partial); return 0; }

        reduceTreeKernel<<<blocks, threads_per_block, shared_mem_size>>>(
            d_temp, d_new_partial, temp_data.size(), find_max);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) { cudaFree(d_temp); cudaFree(d_new_partial); return 0; }

        h_partial.resize(blocks);
        err = cudaMemcpy(h_partial.data(), d_new_partial, blocks * sizeof(int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) { cudaFree(d_temp); cudaFree(d_new_partial); return 0; }

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
// FASE 04: KERNEL DE HISTOGRAMA DE AEROPUERTOS
// ============================================================================

// [4.1. Básico] Kernel simple usando solo memoria global
__global__ void airportHistogramKernel(const int* airport_ids, int num_records, 
                                        int* histogram, int max_id) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_records) {
        int airport_id = airport_ids[idx];
        
        // Verificar que el ID es válido (mayor que 0 y dentro del rango)
        if (airport_id > 0 && airport_id <= max_id) {
            // Usar operación atómica para incrementar el contador de este aeropuerto
            // en memoria global (puede tener alta contención)
            atomicAdd(&histogram[airport_id], 1);
        }
    }
}

// [4.2. Optimizado] Kernel con memoria compartida para reducir contención
// Estrategia: Cada bloque mantiene un histograma local en shared memory
__global__ void airportHistogramSharedKernel(const int* airport_ids, int num_records,
                                              int* global_histogram, int max_id) {
    // Memoria compartida dinámica para el histograma local del bloque
    extern __shared__ int shared_histogram[];
    
    // Fase 1: Inicializar histograma compartido a cero
    // Cada hilo inicializa múltiples posiciones si es necesario
    for (int i = threadIdx.x; i <= max_id; i += blockDim.x) {
        shared_histogram[i] = 0;
    }
    
    __syncthreads();
    
    // Fase 2: Cada hilo procesa sus datos y actualiza el histograma local
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_records) {
        int airport_id = airport_ids[idx];
        
        // Verificar que el ID es válido
        if (airport_id > 0 && airport_id <= max_id) {
            // atomicAdd en memoria compartida es MUCHO más rápido que en global
            atomicAdd(&shared_histogram[airport_id], 1);
        }
    }
    
    __syncthreads();
    
    // Fase 3: Combinar histograma local con el global
    // Cada hilo copia múltiples categorías al histograma global
    for (int i = threadIdx.x; i <= max_id; i += blockDim.x) {
        if (shared_histogram[i] > 0) {
            atomicAdd(&global_histogram[i], shared_histogram[i]);
        }
    }
}

// [4.3. Privatización] Kernel con histogramas privados por bloque en memoria global
// Reduce contención entre bloques
__global__ void airportHistogramPrivateKernel(const int* airport_ids, int num_records,
                                               int* block_histograms, int max_id, int num_blocks) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calcular offset del histograma privado de este bloque
    int histogram_offset = blockIdx.x * (max_id + 1);
    
    if (idx < num_records) {
        int airport_id = airport_ids[idx];
        
        // Verificar que el ID es válido
        if (airport_id > 0 && airport_id <= max_id) {
            // Cada bloque tiene su propio histograma, sin contención entre bloques
            atomicAdd(&block_histograms[histogram_offset + airport_id], 1);
        }
    }
}

// Kernel auxiliar para reducir histogramas privados en uno final
__global__ void reduceHistogramsKernel(const int* block_histograms, int* final_histogram,
                                        int max_id, int num_blocks) {
    int airport_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (airport_id <= max_id) {
        int sum = 0;
        
        // Sumar todos los valores de este airport_id de todos los bloques
        for (int block = 0; block < num_blocks; block++) {
            sum += block_histograms[block * (max_id + 1) + airport_id];
        }
        
        final_histogram[airport_id] = sum;
    }
}

// ============================================================================
// FUNCIONES WRAPPER PARA HISTOGRAMA DE AEROPUERTOS
// ============================================================================

// Función helper para mostrar resultados del histograma
// Recibe el mapa de ID -> código de aeropuerto y el umbral mínimo
void displayHistogramResults(const std::vector<int>& h_histogram, int max_id,
                             const std::unordered_map<int, std::string>& id_to_airport,
                             int threshold) {
    std::cout << "\n=== RESULTADOS DEL HISTOGRAMA ===\n\n";
    
    int total_airports_with_traffic = 0;
    int total_flights = 0;
    
    for (int id = 1; id <= max_id; id++) {
        if (h_histogram[id] > 0) {
            total_airports_with_traffic++;
            total_flights += h_histogram[id];
        }
    }
    
    std::cout << "Aeropuertos unicos encontrados: " << total_airports_with_traffic << "\n";
    std::cout << "Total de vuelos contados: " << total_flights << "\n";
    std::cout << "Umbral minimo de ocurrencias: " << threshold << "\n\n";
    
    // Crear lista de pares (id, count) para ordenar
    std::vector<std::pair<int, int>> airport_counts;
    for (int id = 1; id <= max_id; id++) {
        if (h_histogram[id] >= threshold) {  // Filtrar por umbral
            airport_counts.push_back({id, h_histogram[id]});
        }
    }
    
    if (airport_counts.empty()) {
        std::cout << "No hay aeropuertos con al menos " << threshold << " ocurrencias.\n";
        return;
    }
    
    // Ordenar por cantidad de vuelos (descendente)
    std::sort(airport_counts.begin(), airport_counts.end(),
              [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                  return a.second > b.second;
              });
    
    std::cout << "Aeropuertos con al menos " << threshold << " ocurrencias: " 
              << airport_counts.size() << "\n\n";
    
    int max_count = airport_counts[0].second;

    std::cout << "Histograma de Aeropuertos:\n";
    std::cout << std::string(70, '=') << "\n\n";
    
    for (const auto& pair : airport_counts) {
        int airport_id = pair.first;
        int count = pair.second;
        
        // Buscar código de aeropuerto
        std::string airport_code = "N/A";
        auto it = id_to_airport.find(airport_id);
        if (it != id_to_airport.end()) {
            airport_code = it->second;
        }
        
        // Calcular ancho de barra proporcional
        int bar_width = (count * MAX_BAR_WIDTH) / max_count;
        if (bar_width == 0 && count > 0) {
            bar_width = 1;  // Al menos 1 carácter si hay ocurrencias
        }
        
        // Mostrar código de aeropuerto (padding a 4 caracteres)
        std::cout << std::left << std::setw(4) << airport_code;
        
        // Mostrar ID entre paréntesis (padding a 8 caracteres)
        std::cout << " (" << std::right << std::setw(5) << airport_id << ")";
        
        // Separador
        std::cout << " | ";
        
        // Mostrar conteo (padding a 8 caracteres, alineado a la derecha)
        std::cout << std::right << std::setw(8) << count << " ";
        
        // Mostrar barra visual
        std::cout << std::string(bar_width, '#');
        
        std::cout << "\n";
    }
    
    std::cout << "\n" << std::string(70, '=') << "\n";
}

// [4.1] Versión básica: Solo memoria global
void executeAirportHistogramBasic(const std::vector<int>& airport_ids,
                                   int num_records, int max_id,
                                   const std::unordered_map<int, std::string>& id_to_airport,
                                   int threshold) {
    std::cout << "\n=== Ejecutando Kernel BASICO (Memoria Global) ===\n";
    
    int blocks, threads_per_block;
    calculateOptimalDimensions(num_records, blocks, threads_per_block);
    
    int* d_airport_ids = nullptr;
    int* d_histogram = nullptr;
    size_t histogram_size = (max_id + 1) * sizeof(int);

    cudaError_t err = cudaMalloc(&d_airport_ids, num_records * sizeof(int));
    if (err != cudaSuccess) { std::cerr << "ERROR: cudaMalloc failed\n"; return; }

    err = cudaMalloc(&d_histogram, histogram_size);
    if (err != cudaSuccess) { std::cerr << "ERROR: cudaMalloc failed\n"; cudaFree(d_airport_ids); return; }

    cudaMemset(d_histogram, 0, histogram_size);

    err = cudaMemcpy(d_airport_ids, airport_ids.data(), num_records * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { std::cerr << "ERROR: cudaMemcpy failed\n"; cudaFree(d_airport_ids); cudaFree(d_histogram); return; }

    airportHistogramKernel<<<blocks, threads_per_block>>>(
        d_airport_ids, num_records, d_histogram, max_id);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { std::cerr << "ERROR: kernel failed\n"; cudaFree(d_airport_ids); cudaFree(d_histogram); return; }

    std::vector<int> h_histogram(max_id + 1);
    err = cudaMemcpy(h_histogram.data(), d_histogram, histogram_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { std::cerr << "ERROR: cudaMemcpy failed\n"; cudaFree(d_airport_ids); cudaFree(d_histogram); return; }

    cudaFree(d_airport_ids);
    cudaFree(d_histogram);

    displayHistogramResults(h_histogram, max_id, id_to_airport, threshold);
}

// [4.2] Versión optimizada: Memoria compartida por bloque
void executeAirportHistogramShared(const std::vector<int>& airport_ids,
                                    int num_records, int max_id,
                                    const std::unordered_map<int, std::string>& id_to_airport,
                                    int threshold) {
    std::cout << "\n=== Ejecutando Kernel COMPARTIDO (Shared Memory) ===\n";
    
    int blocks, threads_per_block;
    calculateOptimalDimensions(num_records, blocks, threads_per_block);
    
    // Verificar que el histograma cabe en memoria compartida
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    size_t histogram_size = (max_id + 1) * sizeof(int);
    size_t shared_mem_size = histogram_size;
    
    if (shared_mem_size > prop.sharedMemPerBlock) {
        std::cout << "ADVERTENCIA: Histograma muy grande para shared memory\n";
        std::cout << "Necesario: " << (shared_mem_size / 1024) << " KB, ";
        std::cout << "Disponible: " << (prop.sharedMemPerBlock / 1024) << " KB\n";
        std::cout << "Usando version basica en su lugar...\n";
        executeAirportHistogramBasic(airport_ids, num_records, max_id,
                                    id_to_airport, threshold);
        return;
    }
    
    std::cout << "Memoria compartida usada: " << (shared_mem_size / 1024) << " KB por bloque\n";
    
    int* d_airport_ids = nullptr;
    int* d_histogram = nullptr;

    cudaError_t err = cudaMalloc(&d_airport_ids, num_records * sizeof(int));
    if (err != cudaSuccess) { std::cerr << "ERROR: cudaMalloc failed\n"; return; }

    err = cudaMalloc(&d_histogram, histogram_size);
    if (err != cudaSuccess) { std::cerr << "ERROR: cudaMalloc failed\n"; cudaFree(d_airport_ids); return; }

    cudaMemset(d_histogram, 0, histogram_size);

    err = cudaMemcpy(d_airport_ids, airport_ids.data(), num_records * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { std::cerr << "ERROR: cudaMemcpy failed\n"; cudaFree(d_airport_ids); cudaFree(d_histogram); return; }

    airportHistogramSharedKernel<<<blocks, threads_per_block, shared_mem_size>>>(
        d_airport_ids, num_records, d_histogram, max_id);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { std::cerr << "ERROR: kernel failed\n"; cudaFree(d_airport_ids); cudaFree(d_histogram); return; }

    std::vector<int> h_histogram(max_id + 1);
    err = cudaMemcpy(h_histogram.data(), d_histogram, histogram_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { std::cerr << "ERROR: cudaMemcpy failed\n"; cudaFree(d_airport_ids); cudaFree(d_histogram); return; }

    cudaFree(d_airport_ids);
    cudaFree(d_histogram);

    displayHistogramResults(h_histogram, max_id, id_to_airport, threshold);
}

// [4.3] Versión con privatización: Histograma privado por bloque
void executeAirportHistogramPrivate(const std::vector<int>& airport_ids,
                                     int num_records, int max_id,
                                     const std::unordered_map<int, std::string>& id_to_airport,
                                     int threshold) {
    std::cout << "\n=== Ejecutando Kernel PRIVADO (Histogramas por Bloque) ===\n";
    
    int blocks, threads_per_block;
    calculateOptimalDimensions(num_records, blocks, threads_per_block);
    
    std::cout << "Creando " << blocks << " histogramas privados...\n";
    
    int* d_airport_ids = nullptr;
    int* d_block_histograms = nullptr;
    int* d_final_histogram = nullptr;
    
    size_t histogram_size = (max_id + 1) * sizeof(int);
    size_t total_private_size = blocks * histogram_size;
    
    std::cout << "Memoria para histogramas privados: " 
              << (total_private_size / (1024 * 1024)) << " MB\n";
    
    cudaError_t err = cudaMalloc(&d_airport_ids, num_records * sizeof(int));
    if (err != cudaSuccess) { std::cerr << "ERROR: cudaMalloc failed\n"; return; }

    err = cudaMalloc(&d_block_histograms, total_private_size);
    if (err != cudaSuccess) { std::cerr << "ERROR: cudaMalloc failed\n"; cudaFree(d_airport_ids); return; }

    err = cudaMalloc(&d_final_histogram, histogram_size);
    if (err != cudaSuccess) { std::cerr << "ERROR: cudaMalloc failed\n"; cudaFree(d_airport_ids); cudaFree(d_block_histograms); return; }

    cudaMemset(d_block_histograms, 0, total_private_size);
    cudaMemset(d_final_histogram, 0, histogram_size);

    err = cudaMemcpy(d_airport_ids, airport_ids.data(), num_records * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { std::cerr << "ERROR: cudaMemcpy failed\n"; cudaFree(d_airport_ids); cudaFree(d_block_histograms); cudaFree(d_final_histogram); return; }

    airportHistogramPrivateKernel<<<blocks, threads_per_block>>>(
        d_airport_ids, num_records, d_block_histograms, max_id, blocks);

    int reduce_blocks = ((max_id + 1) + threads_per_block - 1) / threads_per_block;
    reduceHistogramsKernel<<<reduce_blocks, threads_per_block>>>(
        d_block_histograms, d_final_histogram, max_id, blocks);

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { std::cerr << "ERROR: kernel failed\n"; cudaFree(d_airport_ids); cudaFree(d_block_histograms); cudaFree(d_final_histogram); return; }

    std::vector<int> h_histogram(max_id + 1);
    err = cudaMemcpy(h_histogram.data(), d_final_histogram, histogram_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { std::cerr << "ERROR: cudaMemcpy failed\n"; cudaFree(d_airport_ids); cudaFree(d_block_histograms); cudaFree(d_final_histogram); return; }

    cudaFree(d_airport_ids);
    cudaFree(d_block_histograms);
    cudaFree(d_final_histogram);

    displayHistogramResults(h_histogram, max_id, id_to_airport, threshold);
}

// Función principal que elige automáticamente la mejor estrategia
void executeAirportHistogram(const FlightDataset& dataset, bool use_origin,
                             int strategy, int threshold) {
    size_t num_records = dataset.size();
    if (num_records == 0) {
        std::cout << "\nNo hay registros en el dataset.\n";
        return;
    }

    const std::vector<int>& airport_ids = use_origin ?
        dataset.getOriginSeqId() : dataset.getDestSeqId();

    const std::vector<std::string>& airport_codes = use_origin ?
        dataset.getOriginAirport() : dataset.getDestAirport();

    // Construir mapa raw_id -> código de aeropuerto
    std::unordered_map<int, std::string> raw_id_to_code;
    std::cout << "\nCreando mapa de IDs a codigos de aeropuerto...\n";
    for (size_t i = 0; i < num_records; i++) {
        int id = airport_ids[i];
        const std::string& code = airport_codes[i];
        if (id > 0 && !code.empty() && raw_id_to_code.find(id) == raw_id_to_code.end()) {
            raw_id_to_code[id] = code;
        }
    }
    std::cout << "Mapeo creado: " << raw_id_to_code.size() << " aeropuertos unicos mapeados\n";

    if (raw_id_to_code.empty()) {
        std::cout << "\nNo hay IDs válidos en el dataset.\n";
        return;
    }

    // Los IDs crudos (ej: ORIGIN_SEQ_ID) son identificadores federales dispersos
    // (max puede ser ~1.7M con solo ~409 aeropuertos únicos). Usarlos directamente
    // como índices de histograma haría el array gigante (hasta 30 GB en modo privado).
    // Solución: reasignar compact IDs 1..N antes de enviar a la GPU.
    int compact_max_id = static_cast<int>(raw_id_to_code.size());
    std::unordered_map<int, int> raw_to_compact;
    std::unordered_map<int, std::string> compact_to_airport;
    int next_compact = 1;
    for (const auto& kv : raw_id_to_code) {
        raw_to_compact[kv.first] = next_compact;
        compact_to_airport[next_compact] = kv.second;
        next_compact++;
    }

    // Reasignar el vector de IDs a IDs compactos
    std::vector<int> compact_ids(num_records);
    for (size_t i = 0; i < num_records; i++) {
        auto it = raw_to_compact.find(airport_ids[i]);
        compact_ids[i] = (it != raw_to_compact.end()) ? it->second : 0;
    }

    std::cout << "\n=== Generando Histograma de Aeropuertos ===\n";
    std::cout << "Registros a procesar: " << num_records << "\n";
    std::cout << "Aeropuertos unicos: " << compact_max_id << "\n";
    std::cout << "Tamano del histograma (compacto): "
              << ((compact_max_id + 1) * sizeof(int) / 1024 + 1) << " KB\n";
    std::cout << "Tipo: " << (use_origin ? "ORIGIN (Salidas)" : "DEST (Llegadas)") << "\n";
    std::cout << "Umbral minimo: " << threshold << " ocurrencias\n";

    // Ejecutar según estrategia seleccionada
    if (strategy == 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        size_t histogram_size = (compact_max_id + 1) * sizeof(int);

        if (histogram_size <= prop.sharedMemPerBlock / 2) {
            std::cout << "\nEstrategia AUTO: Usando memoria compartida\n";
            executeAirportHistogramShared(compact_ids, num_records, compact_max_id,
                                         compact_to_airport, threshold);
        } else {
            std::cout << "\nEstrategia AUTO: Usando privatizacion\n";
            executeAirportHistogramPrivate(compact_ids, num_records, compact_max_id,
                                          compact_to_airport, threshold);
        }
    } else if (strategy == 1) {
        executeAirportHistogramBasic(compact_ids, num_records, compact_max_id,
                                    compact_to_airport, threshold);
    } else if (strategy == 2) {
        executeAirportHistogramShared(compact_ids, num_records, compact_max_id,
                                     compact_to_airport, threshold);
    } else if (strategy == 3) {
        executeAirportHistogramPrivate(compact_ids, num_records, compact_max_id,
                                       compact_to_airport, threshold);
    }
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
