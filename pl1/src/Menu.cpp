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
    std::cout << "\nReduccion de Retraso\n";
    std::cout << "[Por implementar]\n";
    waitForEnter();
}

void Menu::processAirportHistogram() {
    clearScreen();
    std::cout << "\nHistograma de Aeropuertos\n";
    std::cout << "[Por implementar]\n";
    waitForEnter();
}
