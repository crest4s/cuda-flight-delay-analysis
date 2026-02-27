#include "../include/Menu.h"
#include <iostream>
#include <limits>
#include <cstdlib>

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
    std::cout << "\n=== Analisis de Vuelos ===\n\n";
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
    std::cout << "\n=== Carga de Dataset ===\n\n";
    std::cout << "Ingresa la ruta del archivo CSV\n";
    std::cout << "(presiona Enter para usar ruta por defecto)\n\n";
    std::cout << "Ruta por defecto: " << default_path << "\n\n";
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
    std::cout << "\n=== Retraso en Salida ===\n\n";
    std::cout << "Registros: " << dataset_->size() << "\n";
    std::cout << "\n[Por implementar - kernel CUDA]\n";
    waitForEnter();
}

void Menu::processArrivalDelay() {
    clearScreen();
    std::cout << "\n=== Retraso en Llegada ===\n\n";
    std::cout << "Registros: " << dataset_->size() << "\n";
    std::cout << "\n[Por implementar - kernel CUDA]\n";
    waitForEnter();
}

void Menu::processDelayReduction() {
    clearScreen();
    std::cout << "\n=== Reduccion de Retraso ===\n\n";
    std::cout << "Registros: " << dataset_->size() << "\n";
    std::cout << "\n[Por implementar - kernel CUDA]\n";
    waitForEnter();
}

void Menu::processAirportHistogram() {
    clearScreen();
    std::cout << "\n=== Histograma de Aeropuertos ===\n\n";
    std::cout << "Registros: " << dataset_->size() << "\n";
    std::cout << "\n[Por implementar - kernel CUDA]\n";
    waitForEnter();
}
