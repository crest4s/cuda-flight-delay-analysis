#ifndef CSV_PARSER_H
#define CSV_PARSER_H

#include "FlightDataset.h"
#include <string>
#include <vector>
#include <map>

class CSVParser {
private:
    std::string file_path_;
    std::map<std::string, int> column_indices_; // Mapeo nombre_columna -> índice
    
    std::vector<std::string> splitCSVLine(const std::string& line) const;
    std::string trim(const std::string& str) const;
    float parseFloat(const std::string& str) const;
    int parseInt(const std::string& str, int default_value = 0) const;
    bool processHeader(const std::string& header_line);

public:
    explicit CSVParser(const std::string& file_path);
    ~CSVParser();
    
    bool parse(FlightDataset& dataset);
    bool fileExists() const;
};

#endif // CSV_PARSER_H
