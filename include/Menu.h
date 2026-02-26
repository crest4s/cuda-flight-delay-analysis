#ifndef MENU_H
#define MENU_H

#include "FlightDataset.h"
#include <string>

/**
 * @brief Clase para gestionar la interfaz de usuario del programa
 * 
 * Proporciona un menú interactivo para acceder a las diferentes
 * funcionalidades de procesamiento del dataset de vuelos.
 */
class Menu {
private:
    FlightDataset* dataset_;  // Puntero al dataset cargado
    bool dataset_loaded_;     // Flag para verificar si hay datos cargados
    
    /**
     * @brief Muestra el menú principal
     */
    void displayMainMenu() const;
    
    /**
     * @brief Limpia la pantalla (compatible con Unix/Windows)
     */
    void clearScreen() const;
    
    /**
     * @brief Lee la entrada del usuario
     * 
     * @return String con la opción seleccionada
     */
    std::string getInput() const;
    
    /**
     * @brief Pausa hasta que el usuario presione Enter
     */
    void waitForEnter() const;

public:
    /**
     * @brief Constructor
     */
    Menu();
    
    /**
     * @brief Destructor
     */
    ~Menu();
    
    /**
     * @brief Solicita al usuario la ruta del archivo CSV
     * 
     * @param default_path Ruta por defecto si el usuario presiona Enter
     * @return Ruta del archivo seleccionada por el usuario
     */
    std::string promptForCSVPath(const std::string& default_path) const;
    
    /**
     * @brief Establece el dataset con el que trabajará el menú
     * 
     * @param dataset Puntero al dataset cargado
     */
    void setDataset(FlightDataset* dataset);
    
    /**
     * @brief Ejecuta el bucle principal del menú
     * 
     * @return Código de salida (0 = salida normal)
     */
    int run();
    
    /**
     * @brief Procesa la opción 1: Retraso en salida
     */
    void processDepartureDelay();
    
    /**
     * @brief Procesa la opción 2: Retraso en llegada
     */
    void processArrivalDelay();
    
    /**
     * @brief Procesa la opción 3: Reducción de retraso
     */
    void processDelayReduction();
    
    /**
     * @brief Procesa la opción 4: Histograma de aeropuertos
     */
    void processAirportHistogram();
};

#endif // MENU_H
