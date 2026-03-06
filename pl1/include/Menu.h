#ifndef MENU_H
#define MENU_H

#include "FlightDataset.h"
#include <string>

class Menu {
private:
    FlightDataset* dataset_;  // Puntero al dataset cargado
    bool dataset_loaded_;     // Flag para verificar si hay datos cargados
    
    void displayMainMenu() const;
    void clearScreen() const;
    std::string getInput() const;
    void waitForEnter() const;

public:
    Menu();
    ~Menu();
    
    std::string promptForCSVPath(const std::string& default_path) const;
    void setDataset(FlightDataset* dataset);
    int run();
    
    void processDepartureDelay();
    void processArrivalDelay();
    void processDelayReduction();
    void processAirportHistogram();
};

#endif // MENU_H
