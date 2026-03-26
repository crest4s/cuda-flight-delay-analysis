#include <iostream>
#include <cuda_runtime.h>
#include <algorithm>
#include <unordered_map>
#include <iomanip>
#include <vector>
#include <string>
#include "../include/FlightDataset.h"
#include "../include/cuda_constants.h"

// Forward declarations of helpers defined in gpu_utils.cu
void printGPUInfo();
void calculateOptimalDimensions(int num_records, int& blocks, int& threads_per_block);

// ============================================================================
// FASE 04: KERNELS DE HISTOGRAMA DE AEROPUERTOS
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
static void displayHistogramResults(const std::vector<int>& h_histogram, int max_id,
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

static void executeAirportHistogramBasic(const std::vector<int>& airport_ids,
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

static void executeAirportHistogramShared(const std::vector<int>& airport_ids,
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

static void executeAirportHistogramPrivate(const std::vector<int>& airport_ids,
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

    printGPUInfo();

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
