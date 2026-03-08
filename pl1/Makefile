# Compiladores
NVCC := nvcc
CXX := g++

SRC_DIR := src
INC_DIR := include
OBJ_DIR := obj
BIN_DIR := bin

TARGET := $(BIN_DIR)/flights_analyzer

CXXFLAGS := -std=c++11 -Wall -O2 -I$(INC_DIR)
NVCCFLAGS := -std=c++11 -O2 -I$(INC_DIR) --compiler-options -Wall -arch=sm_50 -Xcompiler -fPIC

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    NVCCFLAGS += -Xcompiler -stdlib=libc++
endif

CPP_SOURCES := $(wildcard $(SRC_DIR)/*.cpp)
CU_SOURCES := main.cu
CPP_OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(CPP_SOURCES))
CU_OBJECTS := $(OBJ_DIR)/main.o
OBJECTS := $(CPP_OBJECTS) $(CU_OBJECTS)
all: directories $(TARGET)

directories:
	@mkdir -p $(OBJ_DIR) $(BIN_DIR) data

$(TARGET): $(OBJECTS)
	@echo "Enlazando ejecutable final..."
	$(NVCC) $(NVCCFLAGS) -o $@ $^
	@echo "Compilacion exitosa: $(TARGET)"

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo "Compilando: $<"
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: %.cu
	@echo "Compilando: $<"
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

run: all
	@$(TARGET)

clean:
	@echo "Limpiando..."
	@rm -rf $(OBJ_DIR) $(BIN_DIR)
	@echo "Completado"

rebuild: clean all

$(OBJ_DIR)/FlightDataset.o: $(INC_DIR)/FlightDataset.h
$(OBJ_DIR)/CSVParser.o: $(INC_DIR)/CSVParser.h $(INC_DIR)/FlightDataset.h
$(OBJ_DIR)/Menu.o: $(INC_DIR)/Menu.h $(INC_DIR)/FlightDataset.h
$(OBJ_DIR)/main.o: $(INC_DIR)/FlightDataset.h $(INC_DIR)/CSVParser.h $(INC_DIR)/Menu.h

.PHONY: all clean rebuild run directories
