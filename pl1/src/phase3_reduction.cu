#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <climits>
#include <vector>

// Forward declarations of helpers defined in gpu_utils.cu
void calculateOptimalDimensions(int num_records, int& blocks, int& threads_per_block);

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

    err = cudaMemcpy(d_data, data.data(), n * sizeof(float), cudaMemcpyHostToDevice);
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
