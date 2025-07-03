# 🧠 Maxweel Detection — Análisis de Trayectorias Circulares con Visión Artificial, CUDA y Diagnóstico Estadístico

Este prototipo simula la detección de trayectorias circulares mediante visión artificial, con soporte para aceleración por GPU mediante **CUDA**. Emplea detección con YOLOv8, seguimiento con SORT, ajuste de curvas cuadráticas y análisis estadístico con exportación a base de datos.

---

## ⚙️ Requisitos del Sistema

- Python 3.8+
- GPU NVIDIA compatible con CUDA (opcional pero recomendado)
- CUDA Toolkit + cuDNN instalados
- OpenCV y Ultralytics (YOLOv8)

---

## 📂 Estructura de Archivos

Maxweel_detection/
├── .venv/ ← Entorno virtual (opcional)
├── Data/
│ └── laboratorio.mp4 ← Video de entrada
├── .gitignore ← Exclusiones de git
├── Deteccion_Yolo.py ← Script principal de detección, seguimiento y análisis
├── output.avi ← Video procesado exportado
├── sort.py ← Algoritmo de seguimiento SORT
├── wheel.pt ← Modelo YOLOv8 entrenado
└── README.md ← Documentación del proyecto

---

## 🚀 ¿Cómo ejecutar?

1. **Clonar el repositorio o copiar la carpeta.**

2. **Activar entorno virtual (opcional pero recomendado):**

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

## 📦 Modelo entrenado

El archivo `wheel.pt` no se encuentra en este repositorio debido a su tamaño.

📁 Puedes descargarlo desde esta carpeta de Google Drive:

🔗 [Abrir carpeta en Google Drive](https://drive.google.com/drive/folders/1y4VoyEVsQyzBrG7d4gqcnFJ6SJHvPHq1?usp=drive_link)

## 📊 Comparación con método tradicional (R)

Esta sección presenta gráficos de diagnóstico generados en R para validar los resultados obtenidos con nuestro software.

> ⚠️ **Nota:**  
> Estas figuras no fueron generadas por el sistema principal, sino por un script externo en R como referencia estadística.

---

### Figuras de referencia:

#### QQ-Plot de residuales studentizados  
![QQ-Plot](https://raw.githubusercontent.com/ValentinoAlvarado/deteccion-centroides-ruedas-fisica/master/imagenes/qqplot.png)

#### Histograma de residuales  
![Histograma](https://raw.githubusercontent.com/ValentinoAlvarado/deteccion-centroides-ruedas-fisica/master/imagenes/hist_residuales.png)

#### Boxplot de residuales  
![Boxplot](https://raw.githubusercontent.com/ValentinoAlvarado/deteccion-centroides-ruedas-fisica/master/imagenes/boxplot.png)

#### Residuales estandarizados vs ajustados  
![Ajustados](https://raw.githubusercontent.com/ValentinoAlvarado/deteccion-centroides-ruedas-fisica/master/imagenes/residuales_ajustados.png)

#### Residuales estandarizados vs distancia d  
![Distancia d](https://raw.githubusercontent.com/ValentinoAlvarado/deteccion-centroides-ruedas-fisica/master/imagenes/residuales_distancia_d.png)

#### Residuales vs orden  
![Orden](https://raw.githubusercontent.com/ValentinoAlvarado/deteccion-centroides-ruedas-fisica/master/imagenes/residuales_orden.png)

#### Residuales studentizados vs leverage  
![Leverage](https://raw.githubusercontent.com/ValentinoAlvarado/deteccion-centroides-ruedas-fisica/master/imagenes/studentized_vs_leverage.png)

#### Residuales estandarizados vs distancia de Cook  
![CookD](https://raw.githubusercontent.com/ValentinoAlvarado/deteccion-centroides-ruedas-fisica/master/imagenes/cookd_vs_residuals.png)

#### Parabola con IC e IP  
![Parábola](https://raw.githubusercontent.com/ValentinoAlvarado/deteccion-centroides-ruedas-fisica/master/imagenes/ajuste_parabola_ic_ip.png)

