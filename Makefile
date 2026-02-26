# ============================================================================
# Makefile para el Proyecto de Análisis de Vuelos (C++ & CUDA)
# ============================================================================
# Este Makefile gestiona la compilación del proyecto que combina código
# C++ estándar (Host) con código CUDA (Device).
#
# Uso:
#   make         - Compila el proyecto completo
#   make clean   - Limpia archivos objeto y ejecutable
#   make run     - Compila y ejecuta el programa
#   make rebuild - Limpia y recompila desde cero
# ============================================================================

# Compiladores
NVCC := nvcc
CXX := g++

# Directorios
SRC_DIR := src
INC_DIR := include
OBJ_DIR := obj
BIN_DIR := bin

# Nombre del ejecutable
TARGET := $(BIN_DIR)/flights_analyzer

# Flags de compilación para C++
CXXFLAGS := -std=c++11 -Wall -O2 -I$(INC_DIR)

# Flags de compilación para NVCC
NVCCFLAGS := -std=c++11 -O2 -I$(INC_DIR) \
             --compiler-options -Wall \
             -arch=sm_50 \
             -Xcompiler -fPIC

# Detectar el sistema operativo
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)  # macOS
    # En macOS, ajustar flags si es necesario
    NVCCFLAGS += -Xcompiler -stdlib=libc++
endif

# Archivos fuente
CPP_SOURCES := $(wildcard $(SRC_DIR)/*.cpp)
CU_SOURCES := main.cu

# Archivos objeto
CPP_OBJECTS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(CPP_SOURCES))
CU_OBJECTS := $(OBJ_DIR)/main.o

# Todos los objetos
OBJECTS := $(CPP_OBJECTS) $(CU_OBJECTS)

# ============================================================================
# Reglas principales
# ============================================================================

# Regla por defecto: compilar todo
all: directories $(TARGET)

# Crear directorios necesarios
directories:
	@mkdir -p $(OBJ_DIR)
	@mkdir -p $(BIN_DIR)
	@mkdir -p data

# Enlazar el ejecutable final
$(TARGET): $(OBJECTS)
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║               ENLAZANDO EJECUTABLE FINAL                   ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
	$(NVCC) $(NVCCFLAGS) -o $@ $^
	@echo "✓ Compilación exitosa: $(TARGET)"
	@echo ""

# Compilar archivos .cpp a .o
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo "Compilando [C++]: $<"
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compilar archivos .cu a .o
$(OBJ_DIR)/%.o: %.cu
	@echo "Compilando [CUDA]: $<"
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# ============================================================================
# Reglas de utilidad
# ============================================================================

# Ejecutar el programa
run: all
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║                  EJECUTANDO PROGRAMA                       ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
	@echo ""
	@$(TARGET)

# Limpiar archivos generados
clean:
	@echo "Limpiando archivos objeto y ejecutable..."
	@rm -rf $(OBJ_DIR)
	@rm -rf $(BIN_DIR)
	@echo "✓ Limpieza completada"

# Limpiar y recompilar
rebuild: clean all

# Mostrar información del sistema CUDA
info:
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║            INFORMACIÓN DEL ENTORNO DE DESARROLLO           ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
	@echo "Sistema operativo: $(UNAME_S)"
	@echo ""
	@echo "Compilador C++:"
	@$(CXX) --version | head -n 1
	@echo ""
	@echo "Compilador CUDA (nvcc):"
	@$(NVCC) --version | grep release
	@echo ""
	@echo "Dispositivos CUDA disponibles:"
	@nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader 2>/dev/null || echo "  No se detectaron dispositivos CUDA o nvidia-smi no está disponible"
	@echo "════════════════════════════════════════════════════════════"

# Ayuda
help:
	@echo "╔════════════════════════════════════════════════════════════╗"
	@echo "║              SISTEMA DE ANÁLISIS DE VUELOS                 ║"
	@echo "║                   Comandos Disponibles                     ║"
	@echo "╚════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "  make           - Compila el proyecto completo"
	@echo "  make run       - Compila y ejecuta el programa"
	@echo "  make clean     - Elimina archivos objeto y ejecutable"
	@echo "  make rebuild   - Limpia y recompila desde cero"
	@echo "  make info      - Muestra información del entorno CUDA"
	@echo "  make help      - Muestra esta ayuda"
	@echo ""
	@echo "════════════════════════════════════════════════════════════"

# Declarar reglas phony (que no crean archivos)
.PHONY: all clean rebuild run info help directories

# ============================================================================
# Dependencias de headers
# ============================================================================

# Los archivos objeto dependen de sus respectivos headers
$(OBJ_DIR)/FlightDataset.o: $(INC_DIR)/FlightDataset.h
$(OBJ_DIR)/CSVParser.o: $(INC_DIR)/CSVParser.h $(INC_DIR)/FlightDataset.h
$(OBJ_DIR)/Menu.o: $(INC_DIR)/Menu.h $(INC_DIR)/FlightDataset.h
$(OBJ_DIR)/main.o: $(INC_DIR)/FlightDataset.h $(INC_DIR)/CSVParser.h $(INC_DIR)/Menu.h
