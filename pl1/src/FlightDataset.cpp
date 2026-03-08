#include "../include/FlightDataset.h"
#include <iostream>
#include <cmath>
#include <iomanip>
#include <climits>

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

// Declaraciones de funciones externas implementadas en main.cu
extern int executeReduceSimple(const std::vector<float>& data, bool find_max);
extern int executeReduceBasic(const std::vector<float>& data, bool find_max);
extern int executeReduceIntermediate(const std::vector<float>& data, bool find_max);
extern int executeReduceTreePattern(const std::vector<float>& data, bool find_max);

// Método auxiliar para obtener el vector de datos según columna
const std::vector<float>& FlightDataset::getColumnData(int column_index) const {
    switch (column_index) {
        case 0: return dep_delay_;
        case 1: return arr_delay_;
        case 2: return weather_delay_;
        default: return dep_delay_;  // Por defecto
    }
}

// [3.1. Simple] Reducción simple
int FlightDataset::reduceDelaySimple(int column_index, bool find_max) {
    const std::vector<float>& data = getColumnData(column_index);
    return executeReduceSimple(data, find_max);
}

// [3.2. Básica] Reducción básica con 3 posiciones
int FlightDataset::reduceDelayBasic(int column_index, bool find_max) {
    const std::vector<float>& data = getColumnData(column_index);
    return executeReduceBasic(data, find_max);
}

// [3.3. Intermedia] Reducción intermedia con hilos pares
int FlightDataset::reduceDelayIntermediate(int column_index, bool find_max) {
    const std::vector<float>& data = getColumnData(column_index);
    return executeReduceIntermediate(data, find_max);
}

// [3.4. Patrón de Reducción] Reducción con patrón de árbol
int FlightDataset::reduceDelayTreePattern(int column_index, bool find_max) {
    const std::vector<float>& data = getColumnData(column_index);
    return executeReduceTreePattern(data, find_max);
}

