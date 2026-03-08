#include "../include/FlightDataset.h"
#include <iostream>
#include <cmath>
#include <iomanip>

// Declaraciones externas de funciones implementadas en main.cu
extern int executeReduceSimple(const std::vector<float>& data, bool find_max);
extern int executeReduceBasic(const std::vector<float>& data, bool find_max);
extern int executeReduceIntermediate(const std::vector<float>& data, bool find_max);
extern int executeReduceTree(const std::vector<float>& data, bool find_max);

FlightDataset::FlightDataset() : num_records_(0) {}

FlightDataset::~FlightDataset() {}

void FlightDataset::addRecord(float dep_delay, float arr_delay, float weather_delay,
                              const std::string& tail_num, int origin_seq_id, int dest_seq_id) {
    dep_delay_.push_back(dep_delay);
    arr_delay_.push_back(arr_delay);
    weather_delay_.push_back(weather_delay);
    tail_num_.push_back(tail_num);
    origin_seq_id_.push_back(origin_seq_id);
    dest_seq_id_.push_back(dest_seq_id);
    num_records_++;
}

void FlightDataset::reserve(size_t capacity) {
    dep_delay_.reserve(capacity);
    arr_delay_.reserve(capacity);
    weather_delay_.reserve(capacity);
    tail_num_.reserve(capacity);
    origin_seq_id_.reserve(capacity);
    dest_seq_id_.reserve(capacity);
}

void FlightDataset::clear() {
    dep_delay_.clear();
    arr_delay_.clear();
    weather_delay_.clear();
    tail_num_.clear();
    origin_seq_id_.clear();
    dest_seq_id_.clear();
    num_records_ = 0;
}

void FlightDataset::printStats() const {
    std::cout << "\nEstadisticas del Dataset\n";
    std::cout << "Registros: " << num_records_ << "\n";
    
    size_t nan_dep = 0, nan_arr = 0, nan_weather = 0;
    for (size_t i = 0; i < num_records_; i++) {
        if (std::isnan(dep_delay_[i])) nan_dep++;
        if (std::isnan(arr_delay_[i])) nan_arr++;
        if (std::isnan(weather_delay_[i])) nan_weather++;
    }
    
    std::cout << "NaN: DEP_DELAY=" << nan_dep 
              << " ARR_DELAY=" << nan_arr
              << " WEATHER_DELAY=" << nan_weather << "\n";
}

// ============================================================================
// IMPLEMENTACIONES DE FASE 03: REDUCCION DE RETRASO
// ============================================================================

// [3.1. Simple] Reducción simple: cada hilo mira su posición
int FlightDataset::reduceDelaySimple(int column_index, bool find_max) {
    const std::vector<float>* data_ptr = nullptr;
    
    // Seleccionar la columna según el índice
    switch (column_index) {
        case 0: data_ptr = &dep_delay_; break;
        case 1: data_ptr = &arr_delay_; break;
        case 2: data_ptr = &weather_delay_; break;
        default:
            std::cerr << "ERROR: Indice de columna invalido\n";
            return 0;
    }
    
    return executeReduceSimple(*data_ptr, find_max);
}

// [3.2. Básica] Reducción básica: cada hilo mira 3 posiciones
int FlightDataset::reduceDelayBasic(int column_index, bool find_max) {
    const std::vector<float>* data_ptr = nullptr;
    
    // Seleccionar la columna según el índice
    switch (column_index) {
        case 0: data_ptr = &dep_delay_; break;
        case 1: data_ptr = &arr_delay_; break;
        case 2: data_ptr = &weather_delay_; break;
        default:
            std::cerr << "ERROR: Indice de columna invalido\n";
            return 0;
    }
    
    return executeReduceBasic(*data_ptr, find_max);
}

// [3.3. Intermedia] Reducción intermedia: hilos pares hacen comparación adicional
int FlightDataset::reduceDelayIntermediate(int column_index, bool find_max) {
    const std::vector<float>* data_ptr = nullptr;
    
    // Seleccionar la columna según el índice
    switch (column_index) {
        case 0: data_ptr = &dep_delay_; break;
        case 1: data_ptr = &arr_delay_; break;
        case 2: data_ptr = &weather_delay_; break;
        default:
            std::cerr << "ERROR: Indice de columna invalido\n";
            return 0;
    }
    
    return executeReduceIntermediate(*data_ptr, find_max);
}

// [3.4. Patrón de Reducción] Reducción con patrón de árbol
int FlightDataset::reduceDelayTreePattern(int column_index, bool find_max) {
    const std::vector<float>* data_ptr = nullptr;
    
    // Seleccionar la columna según el índice
    switch (column_index) {
        case 0: data_ptr = &dep_delay_; break;
        case 1: data_ptr = &arr_delay_; break;
        case 2: data_ptr = &weather_delay_; break;
        default:
            std::cerr << "ERROR: Indice de columna invalido\n";
            return 0;
    }
    
    return executeReduceTree(*data_ptr, find_max);
}

