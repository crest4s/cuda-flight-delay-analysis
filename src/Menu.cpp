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
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════╗\n";
    std::cout << "║     SISTEMA DE ANÁLISIS DE VUELOS - US AIRLINE DATASET   ║\n";
    std::cout << "║                   (CPU + GPU con CUDA)                    ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
    std::cout << "  Seleccione una opción:\n";
    std::cout << "\n";
    std::cout << "    1. Retraso en salida (DEP_DELAY)\n";
    std::cout << "    2. Retraso en llegada (ARR_DELAY)\n";
    std::cout << "    3. Reducción de retraso\n";
    std::cout << "    4. Histograma de aeropuertos\n";
    std::cout << "\n";
    std::cout << "    x. Salir\n";
    std::cout << "\n";
    std::cout << "═════════════════════════════════════════════════════════════\n";
    std::cout << "Opción: ";
}

std::string Menu::getInput() const {
    std::string input;
    std::getline(std::cin, input);
    return input;
}

void Menu::waitForEnter() const {
    std::cout << "\nPresione Enter para continuar...";
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
}

std::string Menu::promptForCSVPath(const std::string& default_path) const {
    clearScreen();
    std::cout << "\n";
    std::cout << "╔═══════════════════════════════════════════════════════════╗\n";
    std::cout << "║            CARGA DE DATASET DE VUELOS (CSV)              ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
    std::cout << "  Ingrese la ruta del archivo CSV.\n";
    std::cout << "  Presione Enter para usar la ruta por defecto.\n";
    std::cout << "\n";
    std::cout << "  Ruta por defecto: " << default_path << "\n";
    std::cout << "\n";
    std::cout << "═════════════════════════════════════════════════════════════\n";
    std::cout << "Ruta del archivo: ";
    
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
            std::cout << "\n¡Hasta pronto!\n";
            running = false;
        } else {
            std::cout << "\nOpción no válida. Por favor intente nuevamente.\n";
            waitForEnter();
        }
    }
    
    return 0;
}

void Menu::processDepartureDelay() {
    clearScreen();
    std::cout << "\n╔═══════════════════════════════════════════════════════════╗\n";
    std::cout << "║           FASE 01: RETRASO EN SALIDA (DEP_DELAY)        ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════╝\n\n";
    
    std::cout << "Registros totales: " << dataset_->size() << "\n\n";
    
    // Solicitar al usuario el umbral de retraso
    std::cout << "Ingrese el umbral de retraso en minutos:\n";
    std::cout << "(valores positivos detectan retrasos, negativos detectan adelantos)\n";
    std::cout << "Umbral: ";
    
    float threshold;
    std::cin >> threshold;
    
    // Limpiar el buffer de entrada
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    
    // Verificar si hay datos para procesar
    if (dataset_->empty()) {
        std::cout << "\nERROR: No hay datos para procesar.\n";
        waitForEnter();
        return;
    }
    
    // Obtener el vector de retrasos en despegue
    const std::vector<float>& dep_delay = dataset_->getDepDelay();
    
    // Llamar a la función CUDA para realizar el análisis en la GPU
    analyzeDepDelayGPU(dep_delay, threshold);
    
    waitForEnter();
}

void Menu::processArrivalDelay() {
    clearScreen();
    std::cout << "\n╔═══════════════════════════════════════════════════════════╗\n";
    std::cout << "║              ANÁLISIS DE RETRASO EN LLEGADA              ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════╝\n\n";
    
    std::cout << "Esta funcionalidad procesará los datos de ARR_DELAY\n";
    std::cout << "utilizando kernels CUDA para calcular estadísticas.\n\n";
    
    std::cout << "Estado: [PENDIENTE DE IMPLEMENTACIÓN]\n";
    std::cout << "\nEsta opción se integrará con el kernel CUDA correspondiente\n";
    std::cout << "en las siguientes fases del proyecto.\n\n";
    
    std::cout << "Dataset cargado:\n";
    std::cout << "  - Total de registros: " << dataset_->size() << "\n";
    std::cout << "  - Columna: ARR_DELAY (float)\n";
    
    waitForEnter();
}

void Menu::processDelayReduction() {
    clearScreen();
    std::cout << "\n╔═══════════════════════════════════════════════════════════╗\n";
    std::cout << "║              ANÁLISIS DE REDUCCIÓN DE RETRASO            ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════╝\n\n";
    
    std::cout << "Esta funcionalidad calculará la diferencia entre\n";
    std::cout << "DEP_DELAY y ARR_DELAY utilizando kernels CUDA.\n\n";
    
    std::cout << "Estado: [PENDIENTE DE IMPLEMENTACIÓN]\n";
    std::cout << "\nEsta opción se integrará con el kernel CUDA correspondiente\n";
    std::cout << "en las siguientes fases del proyecto.\n\n";
    
    std::cout << "Dataset cargado:\n";
    std::cout << "  - Total de registros: " << dataset_->size() << "\n";
    std::cout << "  - Columnas: DEP_DELAY, ARR_DELAY (float)\n";
    
    waitForEnter();
}

void Menu::processAirportHistogram() {
    clearScreen();
    std::cout << "\n╔═══════════════════════════════════════════════════════════╗\n";
    std::cout << "║            HISTOGRAMA DE AEROPUERTOS (ORIGEN)            ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════╝\n\n";
    
    std::cout << "Esta funcionalidad generará un histograma de frecuencias\n";
    std::cout << "de aeropuertos de origen utilizando kernels CUDA.\n\n";
    
    std::cout << "Estado: [PENDIENTE DE IMPLEMENTACIÓN]\n";
    std::cout << "\nEsta opción se integrará con el kernel CUDA correspondiente\n";
    std::cout << "en las siguientes fases del proyecto.\n\n";
    
    std::cout << "Dataset cargado:\n";
    std::cout << "  - Total de registros: " << dataset_->size() << "\n";
    std::cout << "  - Columnas: ORIGIN_AIRPORT_SEQ_ID, DEST_AIRPORT_SEQ_ID (int)\n";
    
    waitForEnter();
}
