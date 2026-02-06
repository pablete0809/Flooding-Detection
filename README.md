# Proyecto de Detección de Inundaciones con Datos de Satélite

Este proyecto tiene como objetivo la creación de un dataset y pipeline para la detección de inundaciones utilizando datos de Sentinel-1 (Radar) y Sentinel-2 (Óptico) obtenidos a través de Google Earth Engine (GEE).

## Estructura del Proyecto

- `gee_pipeline.py`: Script principal que contiene la lógica para descargar y procesar datos de GEE.
- `main.ipynb`: Notebook principal para la ejecución del pipeline y experimentación.
- `dataset_sen12flood_v1/`: Carpeta (ignorada en git) que contiene los datos descargados.

## Documentación de `gee_pipeline.py`

Este módulo contiene funciones esenciales para la obtención y procesamiento de imágenes satelitales. A continuación se detalla cada función:

### `get_sentinel2_data(roi, start_date, end_date, cloud_threshold=60)`
Obtiene y procesa datos ópticos de Sentinel-2.
- **Filtros**: Selecciona imágenes dentro de la región de interés (`roi`) y el rango de fechas, filtrando aquellas con demasiada nubosidad.
- **Máscara de Nubes**: Aplica una máscara para eliminar píxeles de nubes, sombras y cirros utilizando la banda `SCL`.
- **Índices Espectrales**: Calcula y añade las bandas `S2_MNDWI` (Índice Diferencial de Agua Normalizado Modificado) y `S2_NDWI`.

### `get_sentinel1_data(roi, start_date, end_date)`
Obtiene y procesa datos de radar Sentinel-1.
- **Filtrado**: Selecciona imágenes en modo interferométrico (IW) y polarizaciones VV y VH.
- **Procesamiento**: Añade una banda de ratio `S1_VV_VH_ratio` (diferencia en dB entre VV y VH).

### `fuse_datasets(s2_collection, s1_collection, roi, start_date, end_date)`
Fusiona las colecciones de Sentinel-1 y Sentinel-2 en compuestos diarios.
- Itera día a día para encontrar coincidencias o imágenes disponibles de ambos satélites.
- Si ambos satélites tienen datos para un día específico, crea una imagen fusionada con todas las bandas.

### `add_weak_labels(image, threshold=0.0)`
Genera etiquetas "débiles" (weak labels) para la detección de agua/inundación.
- Utiliza el índice MNDWI con un umbral (por defecto 0.0) para clasificar píxeles como agua (inundación) o no agua.

### `download_tile(image, region, filename, scale=10)`
Descarga una única imagen procesada al disco local.

### `download_patches(image, roi, output_dir, scale=10, overwrite=False)`
Descarga una imagen dividiéndola en parches (tiles) y organiza los resultados en directorios separados, similar a la estructura del dataset SEN12FLOOD:
- `S1/`: Datos de Sentinel-1.
- `S2/`: Datos de Sentinel-2.
- `labels/`: Etiquetas de inundación generadas.
- Divide la región de interés en una cuadrícula y descarga cada tesela individualmente para manejar grandes volúmenes de datos.

## Instalación y Uso

1. Asegúrate de tener configurado el entorno de Python.
2. Instala las dependencias: `pip install -r requirements.txt` (si existe) o las necesarias (`earthengine-api`, `geemap`, `rasterio`, etc.).
3. Ejecuta `main.ipynb` para correr el flujo de trabajo.
