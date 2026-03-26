#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include "../include/FlightDataset.h"
#include "../include/cuda_constants.h"

// Memoria constante para el umbral (threshold) - local a esta unidad de traducción
__constant__ float d_threshold;

// Forward declarations of helpers defined in gpu_utils.cu
void printGPUInfo();
void calculateOptimalDimensions(int num_records, int& blocks, int& threads_per_block);

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

// Función wrapper para análisis de ARR_DELAY
void executeArrDelayAnalysis(const FlightDataset& dataset, float threshold, bool delay_type) {
    size_t num_records = dataset.size();
    if (num_records == 0) return;

    printGPUInfo();

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
