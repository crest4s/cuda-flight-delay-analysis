/**
 * @file main.cu
 * @brief Punto de entrada principal del sistema de análisis de vuelos
 * 
 * Este archivo integra la lógica CPU (Host) con CUDA (Device) para el
 * procesamiento eficiente del US Airline Dataset.
 * 
 * Arquitectura:
 *   - Host (CPU): Carga de datos, parsing CSV, interfaz de usuario
 *   - Device (GPU): Procesamiento paralelo mediante kernels CUDA
 * 
 * @author Proyecto Paradigmas de Programación
 * @date 2026
 */

#include <iostream>
#include <cuda_runtime.h>
#include "include/FlightDataset.h"
#include "include/CSVParser.h"
#include "include/Menu.h"

// Ruta por defecto del dataset (puede ser modificada por el usuario)
const std::string DEFAULT_CSV_PATH = "./data/us_airline_dataset.csv";

/**
 * @brief Verifica si CUDA está disponible en el sistema
 * 
 * @return true si hay dispositivos CUDA disponibles, false en caso contrario
 */
bool checkCudaAvailability() {
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    
    if (error != cudaSuccess || device_count == 0) {
        std::cerr << "\n╔═══════════════════════════════════════════════════════════╗\n";
        std::cerr << "║                  ADVERTENCIA: CUDA                        ║\n";
        std::cerr << "╚═══════════════════════════════════════════════════════════╝\n";
        std::cerr << "  No se detectaron dispositivos CUDA en el sistema.\n";
        std::cerr << "  El programa continuará pero las funcionalidades GPU\n";
        std::cerr << "  no estarán disponibles.\n\n";
        return false;
    }
    
    // Mostrar información del dispositivo
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    std::cout << "\n╔═══════════════════════════════════════════════════════════╗\n";
    std::cout << "║                INFORMACIÓN DEL DISPOSITIVO GPU            ║\n";
    std::cout << "╚═══════════════════════════════════════════════════════════╝\n";
    std::cout << "  Dispositivo: " << prop.name << "\n";
    std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "  Memoria Global: " << (prop.totalGlobalMem / (1024 * 1024)) << " MB\n";
    std::cout << "  Multiprocessors: " << prop.multiProcessorCount << "\n";
    std::cout << "  CUDA Cores: ~" << (prop.multiProcessorCount * 128) << " (aproximado)\n";
    std::cout << "═════════════════════════════════════════════════════════════\n\n";
    
    return true;
}

/**
 * @brief Función principal
 * 
 * Flujo del programa:
 *   1. Verificar disponibilidad de CUDA
 *   2. Solicitar ruta del archivo CSV
 *   3. Cargar y parsear el dataset
 *   4. Mostrar estadísticas
 *   5. Ejecutar menú interactivo
 * 
 * @return Código de salida (0 = éxito, 1 = error)
 */
int main() {
    std::cout << "\n";
    std::cout << "███████╗██╗     ██╗ ██████╗ ██╗  ██╗████████╗███████╗\n";
    std::cout << "██╔════╝██║     ██║██╔════╝ ██║  ██║╚══██╔══╝██╔════╝\n";
    std::cout << "█████╗  ██║     ██║██║  ███╗███████║   ██║   ███████╗\n";
    std::cout << "██╔══╝  ██║     ██║██║   ██║██╔══██║   ██║   ╚════██║\n";
    std::cout << "██║     ███████╗██║╚██████╔╝██║  ██║   ██║   ███████║\n";
    std::cout << "╚═╝     ╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚══════╝\n";
    std::cout << "           ANÁLISIS DE DATASET (C++ & CUDA)            \n";
    std::cout << "═════════════════════════════════════════════════════════════\n";
    
    // Paso 1: Verificar disponibilidad de CUDA
    bool cuda_available = checkCudaAvailability();
    if (!cuda_available) {
        std::cout << "Presione Enter para continuar...";
        std::cin.get();
    }
    
    // Paso 2: Crear instancia del menú y solicitar ruta del archivo
    Menu menu;
    std::string csv_path = menu.promptForCSVPath(DEFAULT_CSV_PATH);
    
    // Paso 3: Crear parser y dataset
    CSVParser parser(csv_path);
    FlightDataset dataset;
    
    // Verificar que el archivo existe antes de intentar parsearlo
    if (!parser.fileExists()) {
        std::cerr << "\nERROR: El archivo especificado no existe.\n";
        std::cerr << "Ruta: " << csv_path << "\n\n";
        std::cerr << "Por favor, verifica la ruta e intenta nuevamente.\n";
        return 1;
    }
    
    // Parsear el archivo CSV
    std::cout << "\n";
    if (!parser.parse(dataset)) {
        std::cerr << "\nERROR: Fallo al cargar el dataset.\n";
        std::cerr << "Verifica que el archivo tenga el formato correcto.\n";
        return 1;
    }
    
    // Paso 4: Mostrar estadísticas del dataset cargado
    dataset.printStats();
    
    // Paso 5: Configurar el menú con el dataset y ejecutarlo
    menu.setDataset(&dataset);
    
    std::cout << "Iniciando menú interactivo...\n";
    std::cout << "Presione Enter para continuar...";
    std::cin.get();
    
    int exit_code = menu.run();
    
    // Limpieza (el dataset se destruye automáticamente)
    if (cuda_available) {
        cudaDeviceReset(); // Limpia recursos CUDA
    }
    
    std::cout << "\n¡Programa finalizado correctamente!\n\n";
    
    return exit_code;
}
