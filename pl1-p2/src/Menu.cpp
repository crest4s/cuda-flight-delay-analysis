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
<<<<<<< HEAD
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

=======
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
    
>>>>>>> feature/fase1-despegues
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
