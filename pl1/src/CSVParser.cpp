#include "../include/CSVParser.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <sys/stat.h>

CSVParser::CSVParser(const std::string& file_path) : file_path_(file_path) {}

CSVParser::~CSVParser() {}

bool CSVParser::fileExists() const {
    struct stat buffer;
    return (stat(file_path_.c_str(), &buffer) == 0);
}

std::string CSVParser::trim(const std::string& str) const {
    const std::string whitespace = " \t\n\r\f\v";
    size_t start = str.find_first_not_of(whitespace);
    if (start == std::string::npos) return "";
    size_t end = str.find_last_not_of(whitespace);
    return str.substr(start, end - start + 1);
}

std::vector<std::string> CSVParser::splitCSVLine(const std::string& line) const {
    std::vector<std::string> tokens;
    std::string current_token;
    bool in_quotes = false;
    
    for (size_t i = 0; i < line.length(); i++) {
        char c = line[i];
        
        if (c == '"') {
            in_quotes = !in_quotes;
        } else if (c == ',' && !in_quotes) {
            tokens.push_back(trim(current_token));
            current_token.clear();
        } else {
            current_token += c;
        }
    }
    
    tokens.push_back(trim(current_token));
    
    return tokens;
}

float CSVParser::parseFloat(const std::string& str) const {
    std::string cleaned = trim(str);
    
    if (cleaned.empty() || cleaned == "NA" || cleaned == "N/A" || 
        cleaned == "NULL" || cleaned == "null" || cleaned == "NaN") {
        return std::nanf("");
    }
    
    try {
        size_t pos;
        float value = std::stof(cleaned, &pos);
        
        if (pos != cleaned.length()) {
            return std::nanf("");
        }
        
        return value;
    } catch (...) {
        return std::nanf("");
    }
}

int CSVParser::parseInt(const std::string& str, int default_value) const {
    std::string cleaned = trim(str);
    
    if (cleaned.empty() || cleaned == "NA" || cleaned == "N/A" || 
        cleaned == "NULL" || cleaned == "null") {
        return default_value;
    }
    
    try {
        // Primero intentar parsear como float (para manejar "1247805.0")
        // y luego convertir a int
        float float_value = std::stof(cleaned);
        return static_cast<int>(float_value);
    } catch (...) {
        return default_value;
    }
}

bool CSVParser::processHeader(const std::string& header_line) {
    std::vector<std::string> headers = splitCSVLine(header_line);
    
    column_indices_.clear();
    for (size_t i = 0; i < headers.size(); i++) {
        std::string header = trim(headers[i]);
        column_indices_[header] = static_cast<int>(i);
    }
    
    std::vector<std::string> required_columns = {
        "DEP_DELAY", "ARR_DELAY", "WEATHER_DELAY",
        "TAIL_NUM", "ORIGIN_SEQ_ID", "DEST_SEQ_ID",
        "ORIGIN_AIRPORT", "DEST_AIRPORT"
    };
    
    bool all_found = true;
    for (const auto& col : required_columns) {
        if (column_indices_.find(col) == column_indices_.end()) {
            std::cerr << "ERROR: Columna requerida no encontrada: " << col << "\n";
            all_found = false;
        }
    }
    
    return all_found;
}

bool CSVParser::parse(FlightDataset& dataset) {
    if (!fileExists()) {
        std::cerr << "ERROR: El archivo no existe: " << file_path_ << "\n";
        return false;
    }
    
    std::ifstream file(file_path_);
    if (!file.is_open()) {
        std::cerr << "ERROR: No se pudo abrir el archivo: " << file_path_ << "\n";
        return false;
    }
    
    std::cout << "Cargando dataset: " << file_path_ << "\n";
    std::cout << "Espera un momento...\n";
    
    std::string line;
    bool is_first_line = true;
    size_t line_number = 0;
    size_t records_loaded = 0;
    size_t records_skipped = 0;
    
    dataset.clear();
    dataset.reserve(1300000);
    
    while (std::getline(file, line)) {
        line_number++;
        
        if (is_first_line) {
            is_first_line = false;
            if (!processHeader(line)) {
                std::cerr << "ERROR: Formato de archivo incorrecto\n";
                file.close();
                return false;
            }
            continue;
        }
        
        if (trim(line).empty()) {
            continue;
        }
        
        std::vector<std::string> tokens = splitCSVLine(line);
        
        if (tokens.size() < column_indices_.size()) {
            records_skipped++;
            continue;
        }
        
        try {
            float dep_delay = parseFloat(tokens[column_indices_["DEP_DELAY"]]);
            float arr_delay = parseFloat(tokens[column_indices_["ARR_DELAY"]]);
            float weather_delay = parseFloat(tokens[column_indices_["WEATHER_DELAY"]]);
            std::string tail_num = trim(tokens[column_indices_["TAIL_NUM"]]);
            int origin_seq_id = parseInt(tokens[column_indices_["ORIGIN_SEQ_ID"]], 0);
            int dest_seq_id = parseInt(tokens[column_indices_["DEST_SEQ_ID"]], 0);
            std::string origin_airport = trim(tokens[column_indices_["ORIGIN_AIRPORT"]]);
            std::string dest_airport = trim(tokens[column_indices_["DEST_AIRPORT"]]);
            
            dataset.addRecord(dep_delay, arr_delay, weather_delay,
                            tail_num, origin_seq_id, dest_seq_id,
                            origin_airport, dest_airport);
            records_loaded++;
            
            if (records_loaded % 100000 == 0) {
                std::cout << "  Procesados " << records_loaded << " registros...\n";
            }
            
        } catch (const std::exception& e) {
            records_skipped++;
            if (records_skipped < 10) {
                std::cerr << "WARNING: Error en linea " << line_number << "\n";
            }
        }
    }
    
    file.close();
    
    std::cout << "\nCarga completada!\n";
    std::cout << "Registros cargados: " << records_loaded << "\n";
    if (records_skipped > 0) {
        std::cout << "Registros omitidos: " << records_skipped << "\n";
    }
    
    return records_loaded > 0;
}
