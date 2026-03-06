#include "../include/FlightDataset.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <climits>

// Kernels definidos en main.cu
extern __global__ void reduceSimple(const int* data, int size, int* result, bool find_max);
extern __global__ void reduceBasic(const int* data, int size, int* result, bool find_max);
extern __global__ void reduceIntermediate(const int* data, int size, int* result, bool find_max);
extern __global__ void reduceTreePattern(const int* data, int size, int* partial_results, bool find_max);

// Consulta las propiedades del dispositivo y devuelve el numero optimo de hilos por bloque.
// Siempre devuelve una potencia de 2 para compatibilidad con el patron de reduccion en arbol.
static int getOptimalThreadsPerBlock() {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        return 256; // valor por defecto si falla la consulta
    }
    // Tomamos la menor entre maxThreadsPerBlock y 1024,
    // y ajustamos a la potencia de 2 inferior.
    int max_threads = prop.maxThreadsPerBlock;
    if (max_threads > 1024) max_threads = 1024;
    int threads = 1;
    while (threads * 2 <= max_threads) {
        threads *= 2;
    }
    return threads;
}

// Calcula el numero de bloques asegurandose de no superar maxGridSize[0].
static int safeBlocks(int needed_blocks) {
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        return needed_blocks;
    }
    int max_blocks = prop.maxGridSize[0];
    return (needed_blocks < max_blocks) ? needed_blocks : max_blocks;
}

// Convierte float a int y filtra NaN
static std::vector<int> convertToIntAndSkipNaN(const std::vector<float>& data) {
    std::vector<int> result;
    result.reserve(data.size());
    
    for (float value : data) {
        if (!std::isnan(value)) {
            result.push_back(static_cast<int>(std::trunc(value)));
        }
    }
    
    return result;
}

static const std::vector<float>& getColumnData(const FlightDataset* dataset, int column_index) {
    switch (column_index) {
        case 0: return dataset->getDepDelay();
        case 1: return dataset->getArrDelay();
        case 2: return dataset->getWeatherDelay();
        default: return dataset->getDepDelay();
    }
}

int FlightDataset::reduceDelaySimple(int column_index, bool find_max) {
    std::vector<int> host_data = convertToIntAndSkipNaN(getColumnData(this, column_index));
    int size = host_data.size();

    if (size == 0) {
        std::cerr << "ERROR: No hay datos validos para procesar\n";
        return 0;
    }

    int threads_per_block = getOptimalThreadsPerBlock();
    int blocks = safeBlocks((size + threads_per_block - 1) / threads_per_block);

    int* d_data = nullptr;
    int* d_result = nullptr;

    cudaMalloc(&d_data, size * sizeof(int));
    cudaMalloc(&d_result, sizeof(int));
    cudaMemcpy(d_data, host_data.data(), size * sizeof(int), cudaMemcpyHostToDevice);

    int init_value = find_max ? INT_MIN : INT_MAX;
    cudaMemcpy(d_result, &init_value, sizeof(int), cudaMemcpyHostToDevice);

    reduceSimple<<<blocks, threads_per_block>>>(d_data, size, d_result, find_max);
    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "ERROR en kernel reduceSimple: " << cudaGetErrorString(error) << "\n";
        cudaFree(d_data);
        cudaFree(d_result);
        return 0;
    }

    int result = 0;
    cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    cudaFree(d_result);
    return result;
}

int FlightDataset::reduceDelayBasic(int column_index, bool find_max) {
    std::vector<int> host_data = convertToIntAndSkipNaN(getColumnData(this, column_index));
    int size = host_data.size();

    if (size == 0) {
        std::cerr << "ERROR: No hay datos validos para procesar\n";
        return 0;
    }

    // Un hilo por elemento; cada hilo lee anterior, actual y siguiente.
    int threads_per_block = getOptimalThreadsPerBlock();
    int blocks = safeBlocks((size + threads_per_block - 1) / threads_per_block);
    size_t shared_mem_bytes = threads_per_block * sizeof(int);

    int* d_data = nullptr;
    int* d_result = nullptr;
    size_t data_bytes = size * sizeof(int);
    cudaMalloc(&d_data, data_bytes);
    cudaMalloc(&d_result, sizeof(int));
    cudaMemcpy(d_data, host_data.data(), data_bytes, cudaMemcpyHostToDevice);

    int init_value = find_max ? INT_MIN : INT_MAX;
    cudaMemcpy(d_result, &init_value, sizeof(int), cudaMemcpyHostToDevice);

    reduceBasic<<<blocks, threads_per_block, shared_mem_bytes>>>(d_data, size, d_result, find_max);
    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "ERROR en kernel reduceBasic: " << cudaGetErrorString(error) << "\n";
        cudaFree(d_data);
        cudaFree(d_result);
        return 0;
    }

    int result = 0;
    cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    cudaFree(d_result);
    return result;
}

int FlightDataset::reduceDelayIntermediate(int column_index, bool find_max) {
    std::vector<int> host_data = convertToIntAndSkipNaN(getColumnData(this, column_index));
    int size = host_data.size();

    if (size == 0) {
        std::cerr << "ERROR: No hay datos validos para procesar\n";
        return 0;
    }

    // Un hilo por elemento.
    int threads_per_block = getOptimalThreadsPerBlock();
    int blocks = safeBlocks((size + threads_per_block - 1) / threads_per_block);
    size_t shared_mem_bytes = threads_per_block * sizeof(int);

    int* d_data = nullptr;
    int* d_result = nullptr;
    size_t data_bytes = size * sizeof(int);
    cudaMalloc(&d_data, data_bytes);
    cudaMalloc(&d_result, sizeof(int));
    cudaMemcpy(d_data, host_data.data(), data_bytes, cudaMemcpyHostToDevice);

    int init_value = find_max ? INT_MIN : INT_MAX;
    cudaMemcpy(d_result, &init_value, sizeof(int), cudaMemcpyHostToDevice);

    reduceIntermediate<<<blocks, threads_per_block, shared_mem_bytes>>>(d_data, size, d_result, find_max);
    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "ERROR en kernel reduceIntermediate: " << cudaGetErrorString(error) << "\n";
        cudaFree(d_data);
        cudaFree(d_result);
        return 0;
    }

    int result = 0;
    cudaMemcpy(&result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    cudaFree(d_result);
    return result;
}

int FlightDataset::reduceDelayTreePattern(int column_index, bool find_max) {
    std::vector<int> host_data = convertToIntAndSkipNaN(getColumnData(this, column_index));
    int size = host_data.size();

    if (size == 0) {
        std::cerr << "ERROR: No hay datos validos para procesar\n";
        return 0;
    }

    // El patron en arbol requiere threads_per_block potencia de 2.
    int threads_per_block = getOptimalThreadsPerBlock();
    int blocks = safeBlocks((size + threads_per_block - 1) / threads_per_block);
    size_t shared_mem_bytes = threads_per_block * sizeof(int);

    int* d_data = nullptr;
    int* d_partial = nullptr;
    cudaMalloc(&d_data, size * sizeof(int));
    cudaMalloc(&d_partial, blocks * sizeof(int));
    cudaMemcpy(d_data, host_data.data(), size * sizeof(int), cudaMemcpyHostToDevice);

    reduceTreePattern<<<blocks, threads_per_block, shared_mem_bytes>>>(d_data, size, d_partial, find_max);
    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "ERROR en kernel: " << cudaGetErrorString(error) << "\n";
        cudaFree(d_data);
        cudaFree(d_partial);
        return 0;
    }

    int current_size = blocks;
    while (current_size > 10) {
        int next_blocks = (current_size + threads_per_block - 1) / threads_per_block;
        int* d_next_partial = nullptr;
        cudaMalloc(&d_next_partial, next_blocks * sizeof(int));

        reduceTreePattern<<<next_blocks, threads_per_block, shared_mem_bytes>>>(
            d_partial, current_size, d_next_partial, find_max);
        cudaDeviceSynchronize();

        error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "ERROR en kernel: " << cudaGetErrorString(error) << "\n";
            cudaFree(d_data);
            cudaFree(d_partial);
            cudaFree(d_next_partial);
            return 0;
        }

        cudaFree(d_partial);
        d_partial = d_next_partial;
        current_size = next_blocks;
    }

    std::vector<int> final_results(current_size);
    cudaMemcpy(final_results.data(), d_partial, current_size * sizeof(int), cudaMemcpyDeviceToHost);

    int result = final_results[0];
    for (int i = 1; i < current_size; i++) {
        result = find_max ? std::max(result, final_results[i]) : std::min(result, final_results[i]);
    }

    cudaFree(d_data);
    cudaFree(d_partial);
    return result;
}
