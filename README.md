# Regresión logística binaria sobre MNIST: Scikit-Learn (Python) vs C++ con Eigen

Proyecto final de semestre, **Redes Neuronales**
Universidad Sergio Arboleda, Ciencias de la Computación e Inteligencia Artificial
Docente: Oscar Andrés Arias Moyano

**Integrantes:**
- Ana María Amador
- Daniela López Rincón

---

## Descripción

El proyecto clasifica imágenes de los dígitos **3 y 8** de MNIST con regresión logística y compara el rendimiento (tiempo de entrenamiento, consumo de memoria y métricas de clasificación) de cinco configuraciones:

| Implementación | Algoritmo | Descripción |
|---|---|---|
| **C++ Eigen RowMajor -O2** | Gradiente descendente | Almacenamiento fila-mayor, optimización O2 |
| **C++ Eigen RowMajor -O3** | Gradiente descendente | Fila-mayor, optimización O3 con `-march=native` |
| **C++ Eigen ColMajor -O3** | Gradiente descendente | Columna-mayor, optimización O3 con `-march=native` |
| **Python NumPy** | Gradiente descendente | Mismo algoritmo e hiperparámetros que C++ |
| **Python Scikit-Learn** | L-BFGS | *Baseline* de alto nivel |

La comparación de lenguajes se hace entre **C++ y NumPy**, porque usan el mismo algoritmo. Scikit-Learn usa otro optimizador, por lo que sirve como referencia.

## Dataset

- **MNIST**, dígito 3 (clase 0) contra dígito 8 (clase 1).
- Imágenes de 28x28 píxeles, aplanadas a 784 características, convertidas a `float32` y normalizadas dividiendo entre 255.
- **11 982** imágenes de entrenamiento (6131 de 3 y 5851 de 8) y **1984** de prueba (1010 de 3 y 974 de 8).
- Python exporta los datos a binario (`datos_train.bin`, 37,6 MB, y `datos_test.bin`, 6,2 MB) y C++ los lee, de modo que ambas implementaciones usan exactamente los mismos datos.

## Contenido del repositorio

```
.
├── ProyectoFinal_V2.ipynb   # Notebook de Google Colab con todo el desarrollo
├── PROY_FINAL.docx          # Informe técnico
└── README.md
```

El notebook genera durante su ejecución los demás archivos:

| Archivo | Cómo se genera |
|---|---|
| `main.cpp` | Celda con `%%writefile` (Fase B). Regresión logística con Eigen |
| `train_python.py` | Celda con `%%writefile` (Fase C). Entrenamiento con NumPy y Scikit-Learn |
| `datos_train.bin`, `datos_test.bin` | Función de exportación a binario (Fase B) |
| `lr_row_O2`, `lr_row_O3`, `lr_col_O3` | Compilación de `main.cpp` (Fase B) |

> Si se quiere que el repositorio contenga también el código fuente por separado, se pueden descargar `main.cpp` y `train_python.py` desde Colab y agregarlos a la raíz.

## Contenido por fases

### Fase A: análisis exploratorio y prototipo
- Distribución de clases y de intensidades de píxeles (78,54 % de los píxeles vale cero y 161 posiciones tienen varianza nula).
- Imágenes promedio, mapa de varianza y correlación píxel-etiqueta.
- Correlación entre píxeles vecinos (0,727) y distantes (0,084) en la fila central.
- Baseline con Scikit-Learn: Accuracy de 0,9672.

### Fase B: implementación en C++
- Programa organizado en tipos de datos, carga del conjunto, sigmoide, clase `RegresionLogistica`, métricas y `main()` con dos modos (`train` y `locality`).
- Sigmoide y gradiente descendente de lote completo con regularización L2, usando Eigen.
- Gestión de memoria con punteros (lectura directa al búfer y recorrido con aritmética de punteros), referencias constantes y `Eigen::Map`.
- Compilación por fases: preprocesamiento, compilación a ensamblador, ensamblado y enlazado.
- Tres variantes del ejecutable: `lr_row_O2`, `lr_row_O3` y `lr_col_O3`.

### Fase C: optimización y evaluación
- Benchmark de las cinco implementaciones (3 ejecuciones por implementación, un solo hilo).
- Medición de memoria con `VmHWM` en todos los casos.
- Estudio de localidad espacial y temporal (recorrido por filas vs columnas, gradiente en dos pasadas vs una sola).

## Requisitos

- **Google Colab** (entorno en el que se ejecutó el proyecto) o un sistema **Linux** equivalente. La medición de memoria usa `/proc/self/status`.
- **g++ con soporte C++17** y **Eigen 3** (`libeigen3-dev`).
- **Python** con `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn` y `tensorflow` (este último solo para descargar MNIST).

## Cómo ejecutarlo

### Opción 1: Google Colab (recomendada)
1. Subir `ProyectoFinal_V2.ipynb` a Google Colab.
2. Ejecutar las celdas en orden. El notebook instala Eigen, prepara los datos, genera y compila el C++, y corre el benchmark.

### Opción 2: ejecución manual
Con `main.cpp`, `train_python.py` y los archivos `.bin` ya generados:

```bash
# Compilar las tres variantes de C++
EIGEN=/usr/include/eigen3
g++ -std=c++17 -O2 -I$EIGEN main.cpp -o lr_row_O2
g++ -std=c++17 -O3 -march=native -I$EIGEN main.cpp -o lr_row_O3
g++ -std=c++17 -O3 -march=native -DLAYOUT_COLMAJOR -I$EIGEN main.cpp -o lr_col_O3

# Entrenar y evaluar en C++ (argumentos: tasa de aprendizaje, épocas, lambda)
./lr_row_O3 train datos_train.bin datos_test.bin 0.1 1000 1e-4

# Entrenar y evaluar en Python (un solo hilo)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 train_python.py numpy   datos_train.bin datos_test.bin 0.1 1000 1e-4
python3 train_python.py sklearn datos_train.bin datos_test.bin 0.1 1000 1e-4

# Micro-benchmark de localidad
./lr_row_O3 locality datos_train.bin
```

### Qt Creator (opcional)
Para compilar `main.cpp` en Qt Creator se puede usar un `CMakeLists.txt` como el siguiente (*File > Open File or Project*):

```cmake
cmake_minimum_required(VERSION 3.16)
project(RegresionLogisticaEigen LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_executable(lr_row main.cpp)
target_link_libraries(lr_row Eigen3::Eigen)
target_compile_options(lr_row PRIVATE -O3 -march=native)

add_executable(lr_col main.cpp)
target_link_libraries(lr_col Eigen3::Eigen)
target_compile_definitions(lr_col PRIVATE LAYOUT_COLMAJOR)
target_compile_options(lr_col PRIVATE -O3 -march=native)
```

En *Projects > Run > Command line arguments* escribir: `train datos_train.bin datos_test.bin 0.1 1000 1e-4`

## Hiperparámetros

Iguales en todas las implementaciones:

| Parámetro | Valor |
|---|---|
| Tasa de aprendizaje | 0,1 |
| Épocas (gradiente descendente) | 1000 |
| Regularización L2 (λ) | 1e-4 (equivale a C ≈ 0,835 en Scikit-Learn) |
| Semilla | 42 |
| Hilos | 1 |
| Ejecuciones por implementación | 3 (se reporta el promedio) |

## Resultados

Promedio de 3 ejecuciones en Google Colab (CPU Intel Xeon @ 2,20 GHz, g++ 13.3.0, Eigen 3.4.0), conjunto de prueba:

| Implementación | Entrenamiento (s) | Predicción (s) | Memoria pico (MB) | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| C++ Eigen RowMajor -O2 | 9,3655 | 0,0008 | 74,96 | 0,9672 | 0,9690 | 0,9641 | 0,9665 |
| C++ Eigen RowMajor -O3 | 7,0698 | 0,0007 | 75,04 | 0,9672 | 0,9690 | 0,9641 | 0,9665 |
| C++ Eigen ColMajor -O3 | 7,3678 | 0,0008 | 75,04 | 0,9672 | 0,9690 | 0,9641 | 0,9665 |
| Python NumPy (GD) | 7,4990 | 0,0006 | 218,19 | 0,9672 | 0,9690 | 0,9641 | 0,9665 |
| Python Scikit-Learn (L-BFGS) | 1,3286 | 0,0078 | 302,98 | 0,9672 | 0,9681 | 0,9651 | 0,9666 |

Estudio de localidad (promedio de 100 repeticiones, binario -O3):

| Experimento | Tiempo (ms) |
|---|---|
| Recorrido por filas (contiguo) | 14,41 |
| Recorrido por columnas (saltos de 3136 bytes) | 116,17 |
| Gradiente en dos pasadas | 7,41 |
| Gradiente fusionado (una pasada) | 29,79 |

### Conclusiones principales
- Las cinco implementaciones alcanzaron el mismo **Accuracy de 96,72 %**, con métricas casi idénticas. Las diferencias están en la eficiencia, no en la calidad.
- Con el mismo algoritmo, **C++ -O3 y NumPy son equivalentes** (1,06x), porque el entrenamiento son productos matriz-vector limitados por el ancho de banda de memoria.
- **Scikit-Learn fue el más rápido** (5,3x a 5,6x) por el optimizador L-BFGS, que convergió en 78 iteraciones frente a las 1000 épocas del gradiente descendente. La diferencia se debe al algoritmo y no al lenguaje.
- **C++ usó menos memoria**: ≈ 75 MB frente a 218 MB de NumPy y 303 MB de Scikit-Learn.
- **La localidad espacial es determinante**: recorrer la matriz por columnas fue 8,06 veces más lento que por filas.
- El gradiente fusionado en una pasada fue 4,0 veces más lento que las dos pasadas de Eigen, porque el bucle escrito a mano no se vectoriza. Toda optimización debe validarse con mediciones.
- La diferencia entre RowMajor y ColMajor (4 %) cambió de signo entre sesiones, por lo que no es concluyente.

## Limitaciones

- Tres repeticiones en una máquina virtual compartida limitan la precisión. Los tiempos absolutos varían entre sesiones de Colab (13 % a 37 %), por lo que solo son comparables las mediciones hechas en una misma sesión.
- La comparación de -O2 con -O3 mezcla dos factores, porque las versiones -O3 también usan `-march=native`.
- Las optimizaciones identificadas pero no aplicadas incluyen eliminar las 161 columnas constantes, fusión del gradiente por bloques, lectura directa al almacenamiento de Eigen, uso de `uint8` y paralelización con OpenMP.

## Licencia

Proyecto académico. Uso educativo.
