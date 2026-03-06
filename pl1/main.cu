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
            }
        }
    }
}

void calculateOptimalDimensions(int num_records, int& blocks, int& threads_per_block) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    threads_per_block = (prop.maxThreadsPerBlock >= 512) ? 256 : 128;
    blocks = (num_records + threads_per_block - 1) / threads_per_block;
    
    if (blocks > prop.maxGridSize[0]) {
        blocks = prop.maxGridSize[0];
    }
    
    std::cout << "\nEjecutando en: " << prop.name << "\n";
    std::cout << "Configuración: " << blocks << " bloques x " << threads_per_block << " hilos\n\n";
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
    
    std::cout << "\n=== RESULTADOS ARR_DELAY ===\n";
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
        std::cout << "MATRICULA (TAIL_NUM)\tARR_DELAY (min)\n";
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
    cudaFree(d_arr_delays);
    cudaFree(d_tail_nums);
    cudaFree(d_counter);
    cudaFree(d_output_tail_nums);
    cudaFree(d_output_delays);
    delete[] h_tail_nums;
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
