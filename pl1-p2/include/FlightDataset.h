#ifndef FLIGHT_DATASET_H
#define FLIGHT_DATASET_H

#include <vector>
#include <string>
#include <cstddef>

// Clase para almacenar el dataset
class FlightDataset {
private:
    std::vector<float> dep_delay_;
    std::vector<float> arr_delay_;
    std::vector<float> weather_delay_;
    std::vector<std::string> tail_num_;
    std::vector<int> origin_seq_id_;
    std::vector<int> dest_seq_id_;
    size_t num_records_;

public:
    FlightDataset();
    ~FlightDataset();

    void addRecord(float dep_delay, float arr_delay, float weather_delay,
                   const std::string& tail_num, int origin_seq_id, int dest_seq_id);
    void reserve(size_t capacity);
    void clear();

    // Getters para acceso a los datos (const)
    const std::vector<float>& getDepDelay() const { return dep_delay_; }
    const std::vector<float>& getArrDelay() const { return arr_delay_; }
    const std::vector<float>& getWeatherDelay() const { return weather_delay_; }
    const std::vector<std::string>& getTailNum() const { return tail_num_; }
    const std::vector<int>& getOriginSeqId() const { return origin_seq_id_; }
    const std::vector<int>& getDestSeqId() const { return dest_seq_id_; }

    // Getters para acceso no-const (útil para transferencias a GPU)
    std::vector<float>& getDepDelay() { return dep_delay_; }
    std::vector<float>& getArrDelay() { return arr_delay_; }
    std::vector<float>& getWeatherDelay() { return weather_delay_; }
    std::vector<std::string>& getTailNum() { return tail_num_; }
    std::vector<int>& getOriginSeqId() { return origin_seq_id_; }
    std::vector<int>& getDestSeqId() { return dest_seq_id_; }

    size_t size() const { return num_records_; }
    bool empty() const { return num_records_ == 0; }
    void printStats() const;
    
    // Fase 03: Reducción de retraso
    int reduceDelaySimple(int column_index, bool find_max);
    int reduceDelayBasic(int column_index, bool find_max);
    int reduceDelayIntermediate(int column_index, bool find_max);
    int reduceDelayTreePattern(int column_index, bool find_max);
};

#endif // FLIGHT_DATASET_H
