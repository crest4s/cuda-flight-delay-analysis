# cuda-flight-delay-analysis

GPU analysis of the US Airline dataset (about 1.2 million flights) with CUDA C++. A console menu loads the CSV on the host and runs four analysis phases as CUDA kernels, comparing several implementations of the same problem (atomics, shared memory, constant memory, reduction trees and privatised histograms).

Lab project (PL1) for the *Paradigmas Avanzados de Programación* (Advanced Programming Paradigms) course at the University of Alcalá (UAH), 2025–26 academic year. The same four phases are implemented in functional Scala, with a cloud API, in [airport-data-cloud-viewer](https://github.com/crest4s/airport-data-cloud-viewer).

## Phases

| Phase | Kernels | Technique |
|-------|---------|-----------|
| 1 — Departure delay (`DEP_DELAY`) | `analyzeDepDelayKernel` | Threshold in `__constant__` memory; flights at or above the threshold (or at or below it, for negative thresholds) are printed from the GPU and collected with `atomicAdd` |
| 2 — Arrival delay (`ARR_DELAY`) | `analyzeArrDelayKernel` | Same approach; returns the tail number (`TAIL_NUM`) and delay of every matching flight plus the total count |
| 3 — Min/max reduction (`DEP_DELAY`, `ARR_DELAY` or `WEATHER_DELAY`) | `reduceSimpleKernel`, `reduceBasicKernel`, `reduceIntermediateKernel`, `reduceTreeKernel` | 3.1 one atomic per element · 3.2 each thread checks three positions · 3.3 three positions plus pairwise comparison · 3.4 tree reduction in shared memory |
| 4 — Airport histogram (origin or destination) | `airportHistogramKernel`, `airportHistogramSharedKernel`, `airportHistogramPrivateKernel`, `reduceHistogramsKernel` | 4.1 global-memory atomics · 4.2 shared-memory histogram per block · 4.3 private histograms per block merged by a final kernel |

`gpu_utils.cu` queries the device and picks the block size from its capabilities.

## Project structure

```
pl1/
├── main.cu                 # entry point: asks for the CSV path and shows the menu
├── include/                # CSVParser, FlightDataset, Menu, CUDA constants
├── src/
│   ├── CSVParser.cpp, FlightDataset.cpp, Menu.cpp   # host code
│   ├── gpu_utils.cu
│   └── phase1_dep_delay.cu, phase2_arr_delay.cu, phase3_reduction.cu, phase4_histogram.cu
├── data/                   # place the dataset here (see data/README.md)
├── Makefile                # Linux build with nvcc
└── pl1.sln, pl1.vcxproj    # Visual Studio solution (Windows)
```

## Build and run

Requirements: an NVIDIA GPU and the CUDA Toolkit (`nvcc`). Download the US Airline dataset (Kaggle) and save it as `pl1/data/Airline_dataset.csv`; the program also accepts any other path at start-up.

```bash
cd pl1
make
./bin/flights_analyzer
```

On Windows, open `pl1/pl1.sln` in Visual Studio with the CUDA integration installed.

## Authors

- Adrián Morales Rodríguez ([@crest4s](https://github.com/crest4s))
- [@BCA-Lucas](https://github.com/BCA-Lucas)
