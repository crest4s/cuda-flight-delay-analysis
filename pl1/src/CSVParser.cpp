#include "../include/CSVParser.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <cerrno>
#include <sys/stat.h>
#include <iostream>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static const int READ_BUF_SIZE = 8 * 1024 * 1024; // 8 MB read chunks

// Slot indices (must match col_idx_ documentation in header)
enum ColSlot {
    SLOT_DEP_DELAY     = 0,
    SLOT_ARR_DELAY     = 1,
    SLOT_WEATHER_DELAY = 2,
    SLOT_TAIL_NUM      = 3,
    SLOT_ORIGIN_SEQ_ID = 4,
    SLOT_DEST_SEQ_ID   = 5,
    SLOT_ORIGIN_AIRPORT= 6,
    SLOT_DEST_AIRPORT  = 7,
};

// Column names for each slot (same order as enum above)
static const char* COL_NAMES[8] = {
    "DEP_DELAY", "ARR_DELAY", "WEATHER_DELAY", "TAIL_NUM",
    "ORIGIN_SEQ_ID", "DEST_SEQ_ID", "ORIGIN_AIRPORT", "DEST_AIRPORT"
};

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------
CSVParser::CSVParser(const std::string& file_path)
    : file_path_(file_path), total_cols_(0)
{
    for (int i = 0; i < NUM_COLS; i++) col_idx_[i] = -1;
}

CSVParser::~CSVParser() {}

// ---------------------------------------------------------------------------
// fileExists
// ---------------------------------------------------------------------------
bool CSVParser::fileExists() const {
    struct stat buffer;
    return (stat(file_path_.c_str(), &buffer) == 0);
}

// ---------------------------------------------------------------------------
// nextField — advance p past one CSV field; set fs/fl to start/length
// Returns false only when p was already at end before the call.
// ---------------------------------------------------------------------------
bool CSVParser::nextField(const char*& p, const char* end,
                          const char*& fs, int& fl)
{
    if (p >= end) return false;
    bool in_q = false;
    fs = p;
    while (p < end) {
        char c = *p;
        if (c == '"')                   { in_q = !in_q; p++; continue; }
        if (!in_q && c == ',')          { fl = (int)(p - fs); p++; return true; }
        if (!in_q && (c == '\n' || c == '\r')) { fl = (int)(p - fs); return true; }
        p++;
    }
    fl = (int)(p - fs);
    return true;
}

// ---------------------------------------------------------------------------
// trimPtr — adjust pointer+length to strip surrounding whitespace
// ---------------------------------------------------------------------------
void CSVParser::trimPtr(const char*& s, int& len) {
    while (len > 0 && (*s == ' ' || *s == '\t' || *s == '\r')) { s++; len--; }
    while (len > 0 && (s[len-1] == ' ' || s[len-1] == '\t' || s[len-1] == '\r')) len--;
}

// ---------------------------------------------------------------------------
// parseFloatFast — NA/NaN/NULL detection + strtof (no exceptions)
// ---------------------------------------------------------------------------
float CSVParser::parseFloatFast(const char* s, int len) {
    if (len <= 0) return std::nanf("");
    // Quick NA/NaN/NULL detection via length + first chars
    if (len == 2 && s[0] == 'N' && s[1] == 'A') return std::nanf("");
    if (len == 3 && s[0] == 'N' && s[1] == '/' && s[2] == 'A') return std::nanf("");
    if (len == 3 && s[0] == 'N' && s[1] == 'a' && s[2] == 'N') return std::nanf("");
    if (len == 4 && memcmp(s, "NULL", 4) == 0) return std::nanf("");
    if (len == 4 && memcmp(s, "null", 4) == 0) return std::nanf("");

    char* endptr;
    float v = strtof(s, &endptr);
    if (endptr == s) return std::nanf(""); // nothing parsed
    return v;
}

// ---------------------------------------------------------------------------
// parseIntFast — parse integer (may arrive as float string like "1247805.0")
// ---------------------------------------------------------------------------
int CSVParser::parseIntFast(const char* s, int len, int def) {
    if (len <= 0) return def;
    if (len == 2 && s[0] == 'N' && s[1] == 'A') return def;
    if (len == 3 && s[0] == 'N' && s[1] == '/' && s[2] == 'A') return def;
    if (len == 4 && memcmp(s, "NULL", 4) == 0) return def;
    if (len == 4 && memcmp(s, "null", 4) == 0) return def;

    char* endptr;
    float v = strtof(s, &endptr);
    if (endptr == s) return def;
    return static_cast<int>(v);
}

// ---------------------------------------------------------------------------
// processHeader — fill col_idx_ array; return true if all 8 slots found
// ---------------------------------------------------------------------------
bool CSVParser::processHeader(const char* line, int len) {
    for (int i = 0; i < NUM_COLS; i++) col_idx_[i] = -1;

    const char* p   = line;
    const char* end = line + len;
    int col_num = 0;
    int max_col = 0;

    const char* fs; int fl;
    while (nextField(p, end, fs, fl)) {
        trimPtr(fs, fl);

        // Compare against each required column name
        for (int s = 0; s < NUM_COLS; s++) {
            const char* cname = COL_NAMES[s];
            int clen = (int)strlen(cname);
            if (fl == clen && memcmp(fs, cname, clen) == 0) {
                col_idx_[s] = col_num;
                if (col_num > max_col) max_col = col_num;
                break;
            }
        }
        col_num++;
    }
    total_cols_ = col_num;

    bool all_found = true;
    for (int s = 0; s < NUM_COLS; s++) {
        if (col_idx_[s] < 0) {
            std::cerr << "ERROR: Columna requerida no encontrada: " << COL_NAMES[s] << "\n";
            all_found = false;
        }
    }
    return all_found;
}

// ---------------------------------------------------------------------------
// processLine — parse one data line and add record to dataset
// ---------------------------------------------------------------------------
void CSVParser::processLine(const char* line, int len, int max_col,
                            FlightDataset& ds, size_t& loaded, size_t& skipped)
{
    // Walk to max_col+1 fields, saving pointers for the 8 slots
    const char* field_ptrs[NUM_COLS] = {};
    int         field_lens[NUM_COLS] = {};

    const char* p   = line;
    const char* end = line + len;
    int col_num = 0;

    const char* fs; int fl;
    while (col_num <= max_col && nextField(p, end, fs, fl)) {
        // Check if this column number maps to any required slot
        for (int s = 0; s < NUM_COLS; s++) {
            if (col_idx_[s] == col_num) {
                field_ptrs[s] = fs;
                field_lens[s] = fl;
                break;
            }
        }
        col_num++;
    }

    if (col_num < max_col) {
        skipped++;
        return;
    }

    // Trim all 8 fields
    for (int s = 0; s < NUM_COLS; s++) {
        if (field_ptrs[s]) trimPtr(field_ptrs[s], field_lens[s]);
    }

    float dep_delay     = parseFloatFast(field_ptrs[SLOT_DEP_DELAY],     field_lens[SLOT_DEP_DELAY]);
    float arr_delay     = parseFloatFast(field_ptrs[SLOT_ARR_DELAY],     field_lens[SLOT_ARR_DELAY]);
    float weather_delay = parseFloatFast(field_ptrs[SLOT_WEATHER_DELAY], field_lens[SLOT_WEATHER_DELAY]);
    int   origin_seq_id = parseIntFast  (field_ptrs[SLOT_ORIGIN_SEQ_ID], field_lens[SLOT_ORIGIN_SEQ_ID], 0);
    int   dest_seq_id   = parseIntFast  (field_ptrs[SLOT_DEST_SEQ_ID],   field_lens[SLOT_DEST_SEQ_ID],   0);

    // Build strings only for the 3 text fields
    std::string tail_num(field_ptrs[SLOT_TAIL_NUM],       field_lens[SLOT_TAIL_NUM]);
    std::string origin  (field_ptrs[SLOT_ORIGIN_AIRPORT], field_lens[SLOT_ORIGIN_AIRPORT]);
    std::string dest    (field_ptrs[SLOT_DEST_AIRPORT],   field_lens[SLOT_DEST_AIRPORT]);

    ds.addRecord(dep_delay, arr_delay, weather_delay,
                 tail_num, origin_seq_id, dest_seq_id,
                 origin, dest);
    loaded++;
}

// ---------------------------------------------------------------------------
// parse — main entry point: block I/O with fread
// ---------------------------------------------------------------------------
bool CSVParser::parse(FlightDataset& dataset) {
    if (!fileExists()) {
        std::cerr << "ERROR: El archivo no existe: " << file_path_ << "\n";
        return false;
    }

    FILE* f = fopen(file_path_.c_str(), "rb");
    if (!f) {
        std::cerr << "ERROR: No se pudo abrir el archivo: " << file_path_ << "\n";
        return false;
    }

    std::cout << "Cargando dataset: " << file_path_ << "\n";
    std::cout << "Espera un momento...\n";

    dataset.clear();
    dataset.reserve(1300000);

    static char buf[READ_BUF_SIZE];
    char carry[4096];
    int  carry_len = 0;

    bool header_done  = false;
    bool header_error = false;
    int  max_col      = 0;
    size_t loaded     = 0;
    size_t skipped    = 0;

    auto handle_line = [&](const char* line, int len) {
        // Strip trailing \r/\n
        while (len > 0 && (line[len-1] == '\r' || line[len-1] == '\n')) len--;
        if (len == 0) return;

        if (!header_done) {
            if (processHeader(line, len)) {
                for (int s = 0; s < NUM_COLS; s++) {
                    if (col_idx_[s] > max_col) max_col = col_idx_[s];
                }
                header_done = true;
            } else {
                // processHeader already printed the error
                header_done  = true;
                header_error = true;
            }
            return;
        }

        if (header_error) return;
        processLine(line, len, max_col, dataset, loaded, skipped);
    };

    size_t bytes_read;
    while ((bytes_read = fread(buf, 1, READ_BUF_SIZE, f)) > 0) {
        const char* chunk     = buf;
        const char* chunk_end = buf + bytes_read;

        const char* p = chunk;
        while (p < chunk_end) {
            const char* nl = (const char*)memchr(p, '\n', chunk_end - p);
            if (!nl) {
                // No newline in remaining chunk — stash in carry
                int remaining = (int)(chunk_end - p);
                if (carry_len + remaining < (int)sizeof(carry)) {
                    memcpy(carry + carry_len, p, remaining);
                    carry_len += remaining;
                }
                // else: line too long, skip
                break;
            }

            if (carry_len > 0) {
                // Combine carry + up-to-nl
                int seg = (int)(nl - p + 1);
                if (carry_len + seg < (int)sizeof(carry)) {
                    memcpy(carry + carry_len, p, seg);
                    carry_len += seg;
                    handle_line(carry, carry_len);
                }
                carry_len = 0;
            } else {
                handle_line(p, (int)(nl - p + 1));
            }
            p = nl + 1;
        }
    }

    // Handle last line without trailing newline
    if (carry_len > 0) {
        handle_line(carry, carry_len);
    }

    fclose(f);

    if (header_error) {
        std::cerr << "ERROR: Formato de archivo incorrecto\n";
        return false;
    }

    std::cout << "\nCarga completada!\n";
    std::cout << "Registros cargados: " << loaded << "\n";
    if (skipped > 0) {
        std::cout << "Registros omitidos: " << skipped << "\n";
    }

    return loaded > 0;
}
