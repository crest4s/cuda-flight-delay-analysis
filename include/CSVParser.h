#ifndef CSV_PARSER_H
#define CSV_PARSER_H

#include "FlightDataset.h"
#include <string>
#include <vector>
#include <map>

/**
 * @brief Parser robusto para archivos CSV del US Airline Dataset
 * 
 * Implementa limpieza de datos automática, convirtiendo valores faltantes
 * en NaN para columnas numéricas. Diseñado para eficiencia y escalabilidad.
 */
class CSVParser {
private:
    std::string file_path_;
    std::map<std::string, int> column_indices_; // Mapeo nombre_columna -> índice
    
    /**
     * @brief Divide una línea CSV en tokens respetando comillas
     * 
     * @param line Línea a dividir
     * @return Vector de tokens (columnas)
     */
    std::vector<std::string> splitCSVLine(const std::string& line) const;
    
    /**
     * @brief Elimina espacios en blanco al inicio y final de un string
     * 
     * @param str String a limpiar
     * @return String limpio
     */
    std::string trim(const std::string& str) const;
    
    /**
     * @brief Convierte un string a float, retorna NaN si falla
     * 
     * @param str String a convertir
     * @return Valor float o NaN si no es válido
     */
    float parseFloat(const std::string& str) const;
    
    /**
     * @brief Convierte un string a int, retorna valor por defecto si falla
     * 
     * @param str String a convertir
     * @param default_value Valor por defecto si falla
     * @return Valor int o default_value
     */
    int parseInt(const std::string& str, int default_value = 0) const;
    
    /**
     * @brief Procesa la línea de cabecera del CSV
     * 
     * @param header_line Línea de cabecera
     * @return true si se procesó correctamente, false en caso contrario
     */
    bool processHeader(const std::string& header_line);

public:
    /**
     * @brief Constructor
     * 
     * @param file_path Ruta del archivo CSV a parsear
     */
    explicit CSVParser(const std::string& file_path);
    
    /**
     * @brief Destructor
     */
    ~CSVParser();
    
    /**
     * @brief Parsea el archivo CSV y carga los datos en el dataset
     * 
     * @param dataset Referencia al objeto FlightDataset donde cargar los datos
     * @return true si se cargó correctamente, false en caso contrario
     */
    bool parse(FlightDataset& dataset);
    
    /**
     * @brief Verifica si el archivo existe y es accesible
     * 
     * @return true si el archivo existe, false en caso contrario
     */
    bool fileExists() const;
};

#endif // CSV_PARSER_H
