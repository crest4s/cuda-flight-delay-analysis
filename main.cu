#include <iostream>
#include <cuda_runtime.h>
#include "include/FlightDataset.h"
#include "include/CSVParser.h"
#include "include/Menu.h"

// Ruta por defecto del dataset (puede ser modificada por el usuario)
const std::string DEFAULT_CSV_PATH = "./data/Airline_dataset.csv";

bool checkCudaAvailability() {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    
    if (error != cudaSuccess || device_count == 0) {
        std::cerr << "\nADVERTENCIA: No se detectaron dispositivos CUDA\n";
        std::cerr << "El programa continuara sin funcionalidades GPU\n\n";
        return false;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    std::cout << "\n=== Dispositivo GPU ===\n";
    std::cout << "Nombre: " << prop.name << "\n";
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Memoria Global: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB\n\n";
    
    return true;
}

int main() {
    std::cout << "\n=== Analisis de Vuelos (CUDA) ===\n";
    
    bool cuda_available = checkCudaAvailability();
    if (!cuda_available) {
        std::cout << "Presione Enter para continuar...";
        std::cin.get();
    }
    
    Menu menu;
    std::string csv_path = menu.promptForCSVPath(DEFAULT_CSV_PATH);
    
    CSVParser parser(csv_path);
    FlightDataset dataset;
    
    if (!parser.fileExists()) {
        std::cerr << "\nERROR: Archivo no existe: " << csv_path << "\n";
        return 1;
    }
    
    std::cout << "\n";
    if (!parser.parse(dataset)) {
        std::cerr << "\nERROR: Fallo al cargar dataset\n";
        return 1;
    }
    
    // Paso 4: Mostrar estadísticas del dataset cargado
    dataset.printStats();
    
    // Paso 5: Configurar el menú con el dataset y ejecutarlo
    menu.setDataset(&dataset);
    
    std::cout << "Enter para iniciar menu...";
    std::cin.get();
    
    int exit_code = menu.run();
    
    // Limpieza (el dataset se destruye automáticamente)
    if (cuda_available) {
        cudaDeviceReset(); // Limpia recursos CUDA
    }
    
    std::cout << "\nPrograma finalizado\n\n";
    
    return exit_code;
}
