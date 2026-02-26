#ifndef FLIGHT_DATASET_H
#define FLIGHT_DATASET_H

#include <vector>
#include <string>
#include <cstddef>

/**
 * @brief Clase que encapsula el dataset de vuelos
 * 
 * Almacena cada columna del CSV como un vector independiente,
 * facilitando la transferencia linealizada a memoria GPU.
 * Diseñada siguiendo principios de orientación a objetos.
 */
class FlightDataset {
private:
    // Vectores para almacenar las columnas del dataset
    std::vector<float> dep_delay_;      // Retraso en salida
    std::vector<float> arr_delay_;      // Retraso en llegada
    std::vector<float> weather_delay_;  // Retraso por clima
    std::vector<std::string> tail_num_; // Número de cola del avión
    std::vector<int> origin_seq_id_;    // ID secuencial del aeropuerto de origen
    std::vector<int> dest_seq_id_;      // ID secuencial del aeropuerto de destino
    
    size_t num_records_;                // Número total de registros cargados

public:
    /**
     * @brief Constructor por defecto
     */
    FlightDataset();

    /**
     * @brief Destructor
     */
    ~FlightDataset();

    /**
     * @brief Agrega un registro al dataset
     * 
     * @param dep_delay Retraso en salida
     * @param arr_delay Retraso en llegada
     * @param weather_delay Retraso por clima
     * @param tail_num Número de cola del avión
     * @param origin_seq_id ID del aeropuerto de origen
     * @param dest_seq_id ID del aeropuerto de destino
     */
    void addRecord(float dep_delay, float arr_delay, float weather_delay,
                   const std::string& tail_num, int origin_seq_id, int dest_seq_id);

    /**
     * @brief Reserva memoria para un número estimado de registros
     * 
     * @param capacity Número de registros a reservar
     */
    void reserve(size_t capacity);

    /**
     * @brief Limpia todos los datos del dataset
     */
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

    /**
     * @brief Obtiene el número de registros cargados
     * 
     * @return Número de registros
     */
    size_t size() const { return num_records_; }

    /**
     * @brief Verifica si el dataset está vacío
     * 
     * @return true si está vacío, false en caso contrario
     */
    bool empty() const { return num_records_ == 0; }

    /**
     * @brief Imprime estadísticas básicas del dataset
     */
    void printStats() const;
};

#endif // FLIGHT_DATASET_H
