#include "../include/Menu.h"
#include <iostream>
#include <limits>
#include <cstdlib>

Menu::Menu() : dataset_(nullptr), dataset_loaded_(false) {}

Menu::~Menu() {}

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
    std::cout << "\n=== Reduccion de Retraso (FASE 03) ===\n\n";
    std::cout << "Registros totales: " << dataset_->size() << "\n\n";

    // --- Seleccion de columna ---
    std::cout << "--- Seleccione la columna de retraso ---\n";
    std::cout << "1. DEP_DELAY     (Retraso en salida)\n";
    std::cout << "2. ARR_DELAY     (Retraso en llegada)\n";
    std::cout << "3. WEATHER_DELAY (Retraso por clima)\n";
    std::cout << "x. Cancelar\n";
    std::cout << "\nOpcion: ";

    std::string column_choice = getInput();

    if (column_choice == "x" || column_choice == "X") {
        return;
    }

    int column_index = -1;
    std::string column_name;

    if (column_choice == "1") {
        column_index = 0;
        column_name = "DEP_DELAY";
    } else if (column_choice == "2") {
        column_index = 1;
        column_name = "ARR_DELAY";
    } else if (column_choice == "3") {
        column_index = 2;
        column_name = "WEATHER_DELAY";
    } else {
        std::cout << "\nOpcion invalida\n";
        waitForEnter();
        return;
    }

    // --- Seleccion de operacion ---
    std::cout << "\n--- Seleccione la operacion ---\n";
    std::cout << "1. MAXIMO\n";
    std::cout << "2. MINIMO\n";
    std::cout << "x. Cancelar\n";
    std::cout << "\nOpcion: ";

    std::string operation_choice = getInput();

    if (operation_choice == "x" || operation_choice == "X") {
        return;
    }

    bool find_max = false;
    std::string operation_label;

    if (operation_choice == "1") {
        find_max = true;
        operation_label = "Max()";
    } else if (operation_choice == "2") {
        find_max = false;
        operation_label = "Min()";
    } else {
        std::cout << "\nOpcion invalida\n";
        waitForEnter();
        return;
    }

    // --- Confirmacion y ejecucion de las 4 variantes ---
    clearScreen();
    std::cout << "\n=== Reduccion: " << operation_label << " " << column_name << " ===\n\n";
    std::cout << "Ejecutando las 4 variantes de kernel...\n\n";

    try {
        int r_simple = dataset_->reduceDelaySimple(column_index, find_max);
        std::cout << "[Simple]    " << operation_label << " " << column_name
                  << " = " << r_simple << " minutos\n";

        int r_basic = dataset_->reduceDelayBasic(column_index, find_max);
        std::cout << "[Basica]    " << operation_label << " " << column_name
                  << " = " << r_basic << " minutos\n";

        int r_inter = dataset_->reduceDelayIntermediate(column_index, find_max);
        std::cout << "[Intermedia]" << operation_label << " " << column_name
                  << " = " << r_inter << " minutos\n";

        int r_tree = dataset_->reduceDelayTreePattern(column_index, find_max);
        std::cout << "[Reduccion] " << operation_label << " " << column_name
                  << " = " << r_tree << " minutos\n";

    } catch (const std::exception& e) {
        std::cerr << "\nERROR durante la ejecucion: " << e.what() << "\n";
    }

    waitForEnter();
}

void Menu::processAirportHistogram() {
    clearScreen();
    std::cout << "\n=== Histograma de Aeropuertos ===\n\n";
    std::cout << "Registros: " << dataset_->size() << "\n";
    std::cout << "\n[Por implementar - kernel CUDA]\n";
    waitForEnter();
}
