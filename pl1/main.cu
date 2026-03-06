#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include "include/FlightDataset.h"
#include "include/CSVParser.h"
#include "include/Menu.h"

const std::string DEFAULT_CSV_PATH = "./data/Airline_dataset.csv";

__global__ void analyzeDepDelayKernel(const float* dep_delays, int num_records, 
                                       float threshold, bool delay_type) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_records) {
        float delay = dep_delays[idx];
        
        // Verificar que no sea NaN
        if (!isnan(delay)) {
            bool condition_met = false;
            
            if (delay_type) {
                // Retraso positivo (vuelo sale tarde)
                condition_met = (delay >= threshold);
            } else {
                // Adelanto negativo (vuelo sale temprano)
                condition_met = (delay <= threshold);
            }
            
            if (condition_met) {
                printf("#Hilo %d: Retraso de %.0f minutos.\n", idx, delay);
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
void executeDepDelayAnalysis(const FlightDataset& dataset, float threshold, bool delay_type) {
    size_t num_records = dataset.size();
    if (num_records == 0) return;
    
    int blocks, threads_per_block;
    calculateOptimalDimensions(num_records, blocks, threads_per_block);
    
    const std::vector<float>& dep_delays = dataset.getDepDelay();
    size_t data_size = num_records * sizeof(float);
    
    float* d_dep_delays = nullptr;
    cudaError_t err = cudaMalloc(&d_dep_delays, data_size);
    if (err != cudaSuccess) {
        std::cerr << "ERROR: cudaMalloc failed\n";
        return;
    }
    
    err = cudaMemcpy(d_dep_delays, dep_delays.data(), data_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "ERROR: cudaMemcpy failed\n";
        cudaFree(d_dep_delays);
        return;
    }
    
    analyzeDepDelayKernel<<<blocks, threads_per_block>>>(d_dep_delays, num_records, threshold, delay_type);
    
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "ERROR: kernel execution failed\n";
    }
    
    cudaFree(d_dep_delays);
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
