#include "../include/Menu.h"
#include <iostream>
#include <limits>
#include <cstdlib>
#include <sstream>

// Declaración de funciones externas implementadas en main.cu
extern void printGPUInfo();
extern void executeDepDelayAnalysis(const FlightDataset& dataset, float threshold, bool delay_type);
extern void executeArrDelayAnalysis(const FlightDataset& dataset, float threshold, bool delay_type);
extern void executeAirportHistogram(const FlightDataset& dataset, bool use_origin,
                                   int strategy, int threshold);

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
    std::cout << "\n=== Analisis de Retraso en Salida (DEP_DELAY) ===\n";
    std::cout << "Registros cargados: " << dataset_->size() << "\n\n";

    std::cout << "Tipo de analisis:\n";
    std::cout << "  1. Retrasos (vuelos que salen tarde)\n";
    std::cout << "  2. Adelantos (vuelos que salen temprano)\n";
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
    
    // Paso 1: Seleccionar columna
    std::cout << "Seleccione la columna a analizar:\n";
    std::cout << "  1. DEP_DELAY (Retraso en salida)\n";
    std::cout << "  2. ARR_DELAY (Retraso en llegada)\n";
    std::cout << "  3. WEATHER_DELAY (Retraso por clima)\n";
    std::cout << "Opcion: ";
    
    std::string column_input;
    std::getline(std::cin, column_input);
    
    int column_index;
    std::string column_name;
    
    if (column_input == "1") {
        column_index = 0;
        column_name = "DEP_DELAY";
    } else if (column_input == "2") {
        column_index = 1;
        column_name = "ARR_DELAY";
    } else if (column_input == "3") {
        column_index = 2;
        column_name = "WEATHER_DELAY";
    } else {
        std::cout << "\nOpcion invalida\n";
        waitForEnter();
        return;
    }
    
    // Paso 2: Seleccionar tipo de reducción
    std::cout << "\nSeleccione el tipo de reduccion:\n";
    std::cout << "  1. Maximo\n";
    std::cout << "  2. Minimo\n";
    std::cout << "Opcion: ";
    
    std::string type_input;
    std::getline(std::cin, type_input);
    
    bool find_max;
    std::string operation_name;
    
    if (type_input == "1") {
        find_max = true;
        operation_name = "Max()";
    } else if (type_input == "2") {
        find_max = false;
        operation_name = "Min()";
    } else {
        std::cout << "\nOpcion invalida\n";
        waitForEnter();
        return;
    }
    
    // Mostrar info GPU una sola vez antes de las 4 variantes
    printGPUInfo();

    // Ejecutar las 4 variantes y mostrar resultados
    std::cout << "=== Ejecutando Reducciones ===\n\n";

    // [3.1. Simple]
    std::cout << "Ejecutando variante Simple...\n";
    int result_simple = dataset_->reduceDelaySimple(column_index, find_max);
    
    // [3.2. Básica]
    std::cout << "Ejecutando variante Basica...\n";
    int result_basic = dataset_->reduceDelayBasic(column_index, find_max);
    
    // [3.3. Intermedia]
    std::cout << "Ejecutando variante Intermedia...\n";
    int result_intermediate = dataset_->reduceDelayIntermediate(column_index, find_max);
    
    // [3.4. Reducción con patrón de árbol]
    std::cout << "Ejecutando variante Reduccion con patron de arbol...\n";
    int result_tree = dataset_->reduceDelayTreePattern(column_index, find_max);
    
    // Mostrar resultados
    std::cout << "\n=== RESULTADOS ===\n\n";
    std::cout << "[Simple] " << operation_name << " " << column_name << " = " << result_simple << " minutos\n";
    std::cout << "[Basica] " << operation_name << " " << column_name << " = " << result_basic << " minutos\n";
    std::cout << "[Intermedia] " << operation_name << " " << column_name << " = " << result_intermediate << " minutos\n";
    std::cout << "[Reduccion] " << operation_name << " " << column_name << " = " << result_tree << " minutos\n";
    
    waitForEnter();
}

void Menu::processAirportHistogram() {
    clearScreen();
    std::cout << "\n=== Histograma de Aeropuertos (FASE 04) ===\n";
    std::cout << "Registros cargados: " << dataset_->size() << "\n\n";
    
    // Paso 1: Seleccionar tipo de histograma
    std::cout << "Seleccione el tipo de histograma:\n";
    std::cout << "  1. ORIGIN (Aeropuertos de origen - Salidas)\n";
    std::cout << "  2. DEST (Aeropuertos de destino - Llegadas)\n";
    std::cout << "Opcion: ";
    
    std::string type_input;
    std::getline(std::cin, type_input);
    
    bool use_origin;
    
    if (type_input == "1") {
        use_origin = true;
    } else if (type_input == "2") {
        use_origin = false;
    } else {
        std::cout << "\nOpcion invalida\n";
        waitForEnter();
        return;
    }
    
    // Paso 2: Solicitar umbral mínimo de ocurrencias
    std::cout << "\nIngrese el umbral minimo de ocurrencias para mostrar en el histograma\n";
    std::cout << "(ej: 30000 para mostrar solo aeropuertos con >= 30000 vuelos)\n";
    std::cout << "Umbral: ";
    
    std::string threshold_input;
    std::getline(std::cin, threshold_input);
    
    int threshold = 1;  // Por defecto, mostrar todos
    std::stringstream ss_threshold(threshold_input);
    
    if (!(ss_threshold >> threshold) || threshold < 0) {
        std::cout << "\nEntrada invalida, usando umbral minimo de 1\n";
        threshold = 1;
    }
    
    // Paso 3: Seleccionar estrategia de memoria
    std::cout << "\nSeleccione la estrategia:\n";
    std::cout << "  0. AUTOMATICA (recomendado)\n";
    std::cout << "  1. BASICA (solo memoria global)\n";
    std::cout << "  2. COMPARTIDA (shared memory por bloque)\n";
    std::cout << "  3. PRIVADA (histograma privado por bloque)\n";
    std::cout << "Opcion: ";
    
    std::string strategy_input;
    std::getline(std::cin, strategy_input);
    
    int strategy = 0;
    std::stringstream ss(strategy_input);
    
    if (!(ss >> strategy) || strategy < 0 || strategy > 3) {
        std::cout << "\nOpcion invalida, usando estrategia automatica\n";
        strategy = 0;
    }
    
    // Ejecutar el histograma en GPU
    std::cout << "\n";
    executeAirportHistogram(*dataset_, use_origin, strategy, threshold);
    
    waitForEnter();
}
