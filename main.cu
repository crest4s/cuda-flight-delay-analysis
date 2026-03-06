#include <iostream>
#include <climits>
#include <cuda_runtime.h>
#include "include/FlightDataset.h"
#include "include/CSVParser.h"
#include "include/Menu.h"

const std::string DEFAULT_CSV_PATH = "./data/Airline_dataset.csv";

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

bool checkCudaAvailability() {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    
    if (error != cudaSuccess || device_count == 0) {
        std::cerr << "\nADVERTENCIA: No se detectaron dispositivos CUDA\n";
        std::cerr << "El programa continuara sin funcionalidades GPU\n\n";
        return false;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    std::cout << "\n=== Dispositivo GPU ===\n";
    std::cout << "Nombre: " << prop.name << "\n";
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Memoria Global: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB\n";
    std::cout << "Max hilos por bloque: " << prop.maxThreadsPerBlock << "\n";
    std::cout << "Max bloques (dim X): " << prop.maxGridSize[0] << "\n";
    std::cout << "Multiprocesadores (SM): " << prop.multiProcessorCount << "\n\n";
    
    return true;
}

int main() {
    std::cout << "\n=== Analisis de Vuelos (CUDA) ===\n";
    
    bool cuda_available = checkCudaAvailability();
    if (!cuda_available) {
        std::cout << "Presione Enter para continuar...";
        std::cin.get();
    }
    
    Menu menu;
    std::string csv_path = menu.promptForCSVPath(DEFAULT_CSV_PATH);
    
    CSVParser parser(csv_path);
    FlightDataset dataset;
    
    if (!parser.fileExists()) {
        std::cerr << "\nERROR: Archivo no existe: " << csv_path << "\n";
        return 1;
    }
    
    std::cout << "\n";
    if (!parser.parse(dataset)) {
        std::cerr << "\nERROR: Fallo al cargar dataset\n";
        return 1;
    }
    
    dataset.printStats();
    
    menu.setDataset(&dataset);
    
    std::cout << "Enter para iniciar menu...";
    std::cin.get();
    
    int exit_code = menu.run();
    
    if (cuda_available) {
        cudaDeviceReset();
    }
    
    std::cout << "\nPrograma finalizado\n\n";
    
    return exit_code;
}
