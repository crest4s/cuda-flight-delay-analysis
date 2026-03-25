#ifndef CSV_PARSER_H
#define CSV_PARSER_H

#include "FlightDataset.h"
#include <string>
#include <cstdio>

class CSVParser {
private:
    std::string file_path_;

    // Índices de las 8 columnas requeridas, indexados por slot fijo
    // SLOT: 0=DEP_DELAY, 1=ARR_DELAY, 2=WEATHER_DELAY, 3=TAIL_NUM,
    //       4=ORIGIN_SEQ_ID, 5=DEST_SEQ_ID, 6=ORIGIN_AIRPORT, 7=DEST_AIRPORT
    static const int NUM_COLS = 8;
    int col_idx_[NUM_COLS];
    int total_cols_;

    bool processHeader(const char* line, int len);

    static bool nextField(const char*& p, const char* end,
                          const char*& fs, int& fl);
    static void trimPtr(const char*& s, int& len);
    static float parseFloatFast(const char* s, int len);
    static int   parseIntFast(const char* s, int len, int def);

    void processLine(const char* line, int len, int max_col,
                     FlightDataset& ds, size_t& loaded, size_t& skipped);

public:
    explicit CSVParser(const std::string& file_path);
    ~CSVParser();

    bool parse(FlightDataset& dataset);
    bool fileExists() const;
};

#endif // CSV_PARSER_H
