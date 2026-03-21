# 🎓 Smart Analytics IES

## 📋 Descripción del proyecto

Smart Analytics IES es un modelo analítico diseñado para transformar la gestión de las Instituciones de Educación Superior (IES) en Colombia, migrando de un análisis reactivo a una gestión estratégica basada en datos. El proyecto utiliza 10 años de registros históricos oficiales del SNIES para identificar patrones de matrícula y anticipar, mediante fundamento estadístico, qué programas académicos presentan una tendencia decreciente estructural que comprometa su sostenibilidad a mediano plazo.

El propósito central es crear valor estratégico al fundamentar las decisiones de planeación académica en evidencia robusta, permitiendo proyectar la matrícula a un horizonte de 5 años y reducir la incertidumbre institucional.


## Funcionalidades

- Extracción y consolidación de 11 archivos históricos oficiales del SNIES (formato Excel).
- Normalización y limpieza de variables del dataset.
- Unión de todos los datasets en un único archivo consolidado listo para análisis.
- Análisis Exploratorio de Datos (EDA) con visualizaciones interactivas.
- Implementación de algoritmos de series temporales ARIMA y Prophet para proyección de matrícula.
- Validación del desempeño de los modelos mediante métricas de error MAE y RMSE.
- Construcción del índice de riesgo y matriz académica institucional.
- Identificación de programas académicos con tendencia decreciente estructural.
- Proyección de matrícula a un horizonte de 5 años mediante modelos estadísticos.

## Cómo usar el proyecto localmente

### Prerrequisitos

Antes de comenzar, asegúrate de tener instalados:

- [Docker](https://www.docker.com/get-started) y [Docker Compose](https://docs.docker.com/compose/install/)
- Git

### Pasos de instalación

**1. Clona el repositorio**
```bash
git clone https://github.com/tu-usuario/smart-analytics-ies.git
cd smart-analytics-ies
```

**2. Levanta el entorno con Docker**
```bash
docker-compose up
```

> La primera vez tomará unos minutos mientras descarga la imagen base e instala las dependencias del `requirements.txt`.

**3. Accede a JupyterLab**

Abre tu navegador y ve a:
```
http://localhost:8888
```

No se requiere contraseña ni token.


**4. Ejecuta los notebooks en orden**

Dentro de JupyterLab, navega a la carpeta `work/notebooks/` y ejecuta los notebooks en el siguiente orden:

1. `final_project_data_extraction.ipynb` — Auditoría de los 11 archivos Excel fuente, normalización de nombres de columnas, mapeo hacia un esquema unificado de 11 variables y exportación del dataset consolidado (`data.csv`).

2. `final_project_eda.ipynb` — Carga y tipado del dataset consolidado, estandarización de la variable género, análisis exploratorio de datos (EDA): detección de nulos, duplicados y valores en cero, evolución histórica de matrículas, distribución por dimensión (institución, programa, municipio, género) y brecha de género por año. Exporta el archivo `data_historico.csv`  listo para el modelado.

3. `final_project_models.ipynb` — Carga `data_historico.csv`, agrupa los datos por institución, programa, municipio y género, y ajusta modelos de pronóstico (SARIMA y Prophet) por cada grupo en paralelo. Evalúa el desempeño mediante MAE, RMSE y MAPE, selecciona el mejor modelo por grupo y exporta las predicciones a 5 años en un archivo listo para consumo en Power BI.


**5. Detener el entorno**
```bash
docker-compose down
```

---

## Dependencias principales

| Librería | Uso |
|---|---|
| `pandas` / `numpy` | Manipulación y procesamiento de datos |
| `matplotlib` / `seaborn` / `plotly` | Visualización de datos |
| `statsmodels` / `prophet` | Modelado estadístico y proyección |
| `openpyxl` | Lectura de archivos Excel |
| `jupyterlab` | Entorno de desarrollo interactivo |
| `ipywidgets` | Widgets interactivos en notebooks |
| `joblib` | Serialización de modelos |

---

## Diccionario de datos

| Variable | Tipo | Descripción | Ejemplo |
|---|---|---|---|
| `codigo_institucion` | Numérico | Código identificador único de la institución de educación superior | `1101` |
| `institucion` | Texto | Nombre oficial de la institución de educación superior | `UNIVERSIDAD NACIONAL DE COLOMBIA` |
| `codigo_municipio` | Numérico | Código DANE del municipio donde se ubica la institución | `11001` |
| `municipio` | Texto | Nombre del municipio donde se ubica la institución | `BOGOTA D.C.` |
| `codigo_genero` | Numérico | Código identificador del género (`1` = Masculino, `2` = Femenino) | `1` |
| `genero` | Texto | Descripción del género del estudiante | `MASCULINO` |
| `codigo_programa` | Numérico | Código identificador único del programa académico | `3` |
| `programa` | Texto | Nombre del programa académico | `ZOOTECNIA` |
| `anio` | Numérico | Año académico de reporte | `2014` |
| `semestre` | Numérico | Semestre de reporte (`1` = primer semestre, `7` = segundo semestre) | `1` |
| `matriculados` | Numérico | Número de estudiantes matriculados por programa, institución, género y período | `446` |

> Fuente: Sistema Nacional de Información de la Educación Superior ([SNIES](https://snies.mineducacion.gov.co/))

---


## Autores

| Nombre | Rol |
|---|---|
| Andrés Fernando Puyo | Project Lead y Documentador |
| Andrés Felipe Hernández | Data Engineer |
| Joner Rodolfo Ruíz | Data Analyst |
| Daniel Otero Erazo | Modelador Predictivo |
