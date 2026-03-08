#include "../include/Menu.h"
#include <iostream>
#include <limits>
#include <cstdlib>
#include <sstream>

// Declaración de funciones externas implementadas en main.cu
extern void executeDepDelayAnalysis(const FlightDataset& dataset, float threshold, bool delay_type);
extern void executeArrDelayAnalysis(const FlightDataset& dataset, float threshold, bool delay_type);

Menu::Menu() : dataset_(nullptr), dataset_loaded_(false) {
    // Constructor
}

Menu::~Menu() {
    // Destructor - el dataset se gestiona externamente
}

void Menu::clearScreen() const {
#ifdef _WIN32
    system("cls");
#else
    system("clear");
#endif
}

void Menu::displayMainMenu() const {
    std::cout << "\nAnalisis de Vuelos\n\n";
    std::cout << "1. Retraso en salida\n";
    std::cout << "2. Retraso en llegada\n";
    std::cout << "3. Reduccion de retraso\n";
    std::cout << "4. Histograma de aeropuertos\n";
    std::cout << "x. Salir\n";
    std::cout << "\nOpcion: ";
}

std::string Menu::getInput() const {
    std::string input;
    std::getline(std::cin, input);
    return input;
}

void Menu::waitForEnter() const {
    std::cout << "\nPresiona Enter para continuar...";
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
}

std::string Menu::promptForCSVPath(const std::string& default_path) const {
    clearScreen();
    std::cout << "\nCarga de Dataset\n\n";
    std::cout << "Ruta del CSV (Enter para default): " << default_path << "\n";
    std::cout << "Ruta: ";
    
    std::string input;
    std::getline(std::cin, input);
    
    // Si el usuario no ingresa nada, usar la ruta por defecto
    if (input.empty() || input == "\n") {
        return default_path;
    }
    
    return input;
}

void Menu::setDataset(FlightDataset* dataset) {
    dataset_ = dataset;
    dataset_loaded_ = (dataset_ != nullptr && !dataset_->empty());
}

int Menu::run() {
    if (!dataset_loaded_) {
        std::cerr << "ERROR: No hay dataset cargado. No se puede ejecutar el menú.\n";
        return 1;
    }
    
    bool running = true;
    
    while (running) {
        clearScreen();
        displayMainMenu();
        
        std::string option = getInput();
        
        if (option == "1") {
            processDepartureDelay();
        } else if (option == "2") {
            processArrivalDelay();
        } else if (option == "3") {
            processDelayReduction();
        } else if (option == "4") {
            processAirportHistogram();
        } else if (option == "x" || option == "X") {
            std::cout << "\nSaliendo...\n";
            running = false;
        } else {
            std::cout << "\nOpcion invalida\n";
            waitForEnter();
        }
    }
    
    return 0;
}

void Menu::processDepartureDelay() {
    clearScreen();
    std::cout << "\nRetraso en Salida (DEP_DELAY)\n";
    std::cout << "Registros: " << dataset_->size() << "\n\n";
    
    std::cout << "1. Retrasos\n";
    std::cout << "2. Adelantos\n";
    std::cout << "Opcion: ";
    
    std::string type_input;
    std::getline(std::cin, type_input);
    
    bool delay_type = (type_input != "2");
    
    std::cout << "\nUmbral (minutos): ";
    
    std::string threshold_input;
    std::getline(std::cin, threshold_input);
    
    float threshold;
    std::stringstream ss(threshold_input);
    
    if (!(ss >> threshold)) {
        std::cout << "Entrada invalida\n";
        waitForEnter();
        return;
    }
    
    std::cout << "\n";
    executeDepDelayAnalysis(*dataset_, threshold, delay_type);
    std::cout << "\n";
    waitForEnter();
}

void Menu::processArrivalDelay() {
    clearScreen();
    std::cout << "\n=== Analisis de Retraso en Llegada (ARR_DELAY) ===\n";
    std::cout << "Registros cargados: " << dataset_->size() << "\n\n";
    
    std::cout << "Tipo de analisis:\n";
    std::cout << "  1. Retrasos (vuelos que llegan tarde)\n";
    std::cout << "  2. Adelantos (vuelos que llegan temprano)\n";
    std::cout << "Opcion: ";
    
    std::string type_input;
    std::getline(std::cin, type_input);
    
    bool delay_type = (type_input != "2");
    
    if (delay_type) {
        std::cout << "\nUmbral de retraso (minutos positivos, ej: 1440 para 24 horas): ";
    } else {
        std::cout << "\nUmbral de adelanto (minutos negativos, ej: -15 para 15 min temprano): ";
    }
    
    std::string threshold_input;
    std::getline(std::cin, threshold_input);
    
    float threshold;
    std::stringstream ss(threshold_input);
    
    if (!(ss >> threshold)) {
        std::cout << "\nEntrada invalida. Debe ser un numero.\n";
        waitForEnter();
        return;
    }
    
    // Validar que el umbral tenga el signo correcto
    if (delay_type && threshold < 0) {
        std::cout << "\nAdvertencia: Para retrasos, el umbral debe ser positivo.\n";
        std::cout << "Convertido a: " << -threshold << " minutos.\n";
        threshold = -threshold;
    } else if (!delay_type && threshold > 0) {
        std::cout << "\nAdvertencia: Para adelantos, el umbral debe ser negativo.\n";
        std::cout << "Convertido a: " << -threshold << " minutos.\n";
        threshold = -threshold;
    }
    
    std::cout << "\n";
    executeArrDelayAnalysis(*dataset_, threshold, delay_type);
    waitForEnter();
}

void Menu::processDelayReduction() {
    clearScreen();
    std::cout << "\n=== Reduccion de Retraso (FASE 03) ===\n";
    std::cout << "Registros cargados: " << dataset_->size() << "\n\n";
    
    // Seleccionar columna
    std::cout << "Seleccione columna:\n";
    std::cout << "  0. DEP_DELAY (Retraso en salida)\n";
    std::cout << "  1. ARR_DELAY (Retraso en llegada)\n";
    std::cout << "  2. WEATHER_DELAY (Retraso por clima)\n";
    std::cout << "Opcion: ";
    
    std::string column_input;
    std::getline(std::cin, column_input);
    
    int column_index;
    std::stringstream ss1(column_input);
    
    if (!(ss1 >> column_index) || column_index < 0 || column_index > 2) {
        std::cout << "\nOpcion invalida. Debe ser 0, 1 o 2.\n";
        waitForEnter();
        return;
    }
    
    // Seleccionar tipo de reducción
    std::cout << "\nSeleccione tipo de reduccion:\n";
    std::cout << "  0. Minimo (Min)\n";
    std::cout << "  1. Maximo (Max)\n";
    std::cout << "Opcion: ";
    
    std::string type_input;
    std::getline(std::cin, type_input);
    
    int reduction_type;
    std::stringstream ss2(type_input);
    
    if (!(ss2 >> reduction_type) || reduction_type < 0 || reduction_type > 1) {
        std::cout << "\nOpcion invalida. Debe ser 0 o 1.\n";
        waitForEnter();
        return;
    }
    
    bool find_max = (reduction_type == 1);
    
    // Nombres de columnas
    const char* column_names[] = {"DEP_DELAY", "ARR_DELAY", "WEATHER_DELAY"};
    const char* operation_name = find_max ? "Max()" : "Min()";
    
    std::cout << "\n=== RESULTADOS ===\n";
    std::cout << "Columna: " << column_names[column_index] << "\n";
    std::cout << "Operacion: " << operation_name << "\n\n";
    
    // Ejecutar las 4 variantes
    int result_simple = dataset_->reduceDelaySimple(column_index, find_max);
    std::cout << "[Simple] " << operation_name << " " << column_names[column_index] 
              << " = " << result_simple << " minutos\n";
    
    int result_basic = dataset_->reduceDelayBasic(column_index, find_max);
    std::cout << "[Basica] " << operation_name << " " << column_names[column_index] 
              << " = " << result_basic << " minutos\n";
    
    int result_intermediate = dataset_->reduceDelayIntermediate(column_index, find_max);
    std::cout << "[Intermedia] " << operation_name << " " << column_names[column_index] 
              << " = " << result_intermediate << " minutos\n";
    
    int result_tree = dataset_->reduceDelayTreePattern(column_index, find_max);
    std::cout << "[Reduccion] " << operation_name << " " << column_names[column_index] 
              << " = " << result_tree << " minutos\n";
    
    waitForEnter();
}

void Menu::processAirportHistogram() {
    clearScreen();
    std::cout << "\nHistograma de Aeropuertos\n";
    std::cout << "[Por implementar]\n";
    waitForEnter();
}
