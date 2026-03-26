#include <iostream>
#include <cuda_runtime.h>
#include "include/FlightDataset.h"
#include "include/CSVParser.h"
#include "include/Menu.h"

const std::string DEFAULT_CSV_PATH = "./data/Airline_dataset.csv";

// Forward declaration of function defined in src/gpu_utils.cu
extern bool checkCudaAvailability();

int main() {
    std::cout << "\nAnalisis de Vuelos (CUDA)\n";

    bool cuda_available = checkCudaAvailability();

    Menu menu;
    std::string csv_path = menu.promptForCSVPath(DEFAULT_CSV_PATH);

    CSVParser parser(csv_path);
    FlightDataset dataset;

    if (!parser.fileExists()) {
        std::cerr << "\nERROR: Archivo no existe: " << csv_path << "\n";
        return 1;
    }

    if (!parser.parse(dataset)) {
        std::cerr << "ERROR: Fallo al cargar dataset\n";
        return 1;
    }

    dataset.printStats();
    menu.setDataset(&dataset);

    std::cout << "\nEnter para continuar...";
    std::cin.get();

    int exit_code = menu.run();

    if (cuda_available) {
        cudaDeviceReset();
    }

    return exit_code;
}
