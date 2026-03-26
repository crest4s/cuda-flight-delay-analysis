#include <iostream>
#include <cuda_runtime.h>

// Muestra información de la GPU una única vez por fase
void printGPUInfo() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "\n=== Configuracion de Ejecucion CUDA ===\n";
    std::cout << "GPU: " << prop.name << "\n";
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Max Threads por Bloque: " << prop.maxThreadsPerBlock << "\n\n";
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

    std::cout << "Configuracion: " << blocks << " bloques x " << threads_per_block << " hilos\n";
    std::cout << "Total de hilos: " << (blocks * threads_per_block) << "\n\n";
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
