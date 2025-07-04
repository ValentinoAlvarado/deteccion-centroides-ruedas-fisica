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
```

### 📘 Diagrama de Clases del Proyecto

El siguiente diagrama representa la arquitectura del sistema, incluyendo la interacción entre las clases `App`, `ExperimentTab`, `VideoProcessor`, `DBClient`, entre otras.

![Diagrama de Clases UML](https://raw.githubusercontent.com/ValentinoAlvarado/deteccion-centroides-ruedas-fisica/master/docs/UML_class_diagram.svg)

> *Figura 1.* Diagrama UML generado con PlantUML.

## 📊 Visualización de resultados del entrenamiento (YOLOv8)

> Las siguientes imágenes fueron generadas automáticamente por el framework Ultralytics YOLOv8.  
> No forman parte directa del código fuente, sino que sirven como referencia visual  
> para comparar el rendimiento del modelo con métodos tradicionales.

## Video demostrativo



---

### 1. Curva F1  
Evalúa el equilibrio entre precisión y sensibilidad del modelo durante las épocas.

<img src="https://raw.githubusercontent.com/ValentinoAlvarado/deteccion-centroides-ruedas-fisica/master/train/F1_curve.png" width="520"/>

---

### 2. Curva de Precisión vs Recall (PR)  
Muestra cómo varía la precisión a medida que cambia el recall.

<img src="https://raw.githubusercontent.com/ValentinoAlvarado/deteccion-centroides-ruedas-fisica/master/train/PR_curve.png" width="520"/>

---

### 3. Curva de Precisión  
Indica la precisión del modelo a lo largo del entrenamiento.

<img src="https://raw.githubusercontent.com/ValentinoAlvarado/deteccion-centroides-ruedas-fisica/master/train/P_curve.png" width="520"/>

---

### 4. Curva de Recall  
Describe la capacidad del modelo para detectar correctamente los objetos reales.

<img src="https://raw.githubusercontent.com/ValentinoAlvarado/deteccion-centroides-ruedas-fisica/master/train/R_curve.png" width="520"/>

---

### 5. Matriz de Confusión Absoluta  
Representa el número exacto de verdaderos positivos, falsos negativos, etc.

<img src="https://raw.githubusercontent.com/ValentinoAlvarado/deteccion-centroides-ruedas-fisica/master/train/confusion_matrix.png" width="520"/>

---

### 6. Matriz de Confusión Normalizada  
Muestra las proporciones relativas por clase para evaluar la distribución del rendimiento.

<img src="https://raw.githubusercontent.com/ValentinoAlvarado/deteccion-centroides-ruedas-fisica/master/train/confusion_matrix_normalized.png" width="520"/>

---

### 7. Distribución de Etiquetas  
Frecuencia de aparición de cada clase en el conjunto de entrenamiento.

<img src="https://raw.githubusercontent.com/ValentinoAlvarado/deteccion-centroides-ruedas-fisica/master/train/labels.jpg" width="520"/>

---

### 8. Resultados
En conjunto, los gráficos evidencian un entrenamiento estable, donde las pérdidas disminuyen y las métricas de desempeño aumentan, sin señales claras de sobreajuste.

<img src="https://raw.githubusercontent.com/ValentinoAlvarado/deteccion-centroides-ruedas-fisica/master/train/results.png" width="520"/>

---

### 9. Predicción en lotes de entrenamiento
Visualización de las detecciones en entrenamiento.

<img src="https://raw.githubusercontent.com/ValentinoAlvarado/deteccion-centroides-ruedas-fisica/master/train/train_batch0.jpg" width="520"/>
<img src="https://raw.githubusercontent.com/ValentinoAlvarado/deteccion-centroides-ruedas-fisica/master/train/train_batch1.jpg" width="520"/>
<img src="https://raw.githubusercontent.com/ValentinoAlvarado/deteccion-centroides-ruedas-fisica/master/train/train_batch2.jpg" width="520"/>
<img src="https://raw.githubusercontent.com/ValentinoAlvarado/deteccion-centroides-ruedas-fisica/master/train/train_batch579600.jpg" width="520"/>
<img src="https://raw.githubusercontent.com/ValentinoAlvarado/deteccion-centroides-ruedas-fisica/master/train/train_batch579601.jpg" width="520"/>
<img src="https://raw.githubusercontent.com/ValentinoAlvarado/deteccion-centroides-ruedas-fisica/master/train/train_batch579602.jpg" width="520"/>
---

### 10. Visualización del conjunto de validación – Etiquetas 
Muestra las anotaciones reales sobre las imágenes de entrenamiento.

<img src="https://raw.githubusercontent.com/ValentinoAlvarado/deteccion-centroides-ruedas-fisica/master/train/val_batch0_labels.jpg" width="520"/>
<img src="https://raw.githubusercontent.com/ValentinoAlvarado/deteccion-centroides-ruedas-fisica/master/train/val_batch1_labels.jpg" width="520"/>
<img src="https://raw.githubusercontent.com/ValentinoAlvarado/deteccion-centroides-ruedas-fisica/master/train/val_batch2_labels.jpg" width="520"/>
---

### 11. Visualización del conjunto de validación – Predicciones
Ejemplos visuales con las detecciones generadas por el modelo.

<img src="https://raw.githubusercontent.com/ValentinoAlvarado/deteccion-centroides-ruedas-fisica/master/train/val_batch0_pred.jpg" width="520"/>
<img src="https://raw.githubusercontent.com/ValentinoAlvarado/deteccion-centroides-ruedas-fisica/master/train/val_batch1_pred.jpg" width="520"/>
<img src="https://raw.githubusercontent.com/ValentinoAlvarado/deteccion-centroides-ruedas-fisica/master/train/val_batch2_pred.jpg" width="520"/>
---

---
## 📊 Comparación con método tradicional (R)

Esta sección presenta gráficos de diagnóstico generados en R para validar los resultados obtenidos con nuestro software.

> ⚠️ **Nota:**  
> Estas figuras no fueron generadas por el sistema principal, sino por un script externo en R como referencia estadística.

---

### Figuras de referencia:

#### QQ-Plot de residuales studentizados  
<img src="https://raw.githubusercontent.com/ValentinoAlvarado/deteccion-centroides-ruedas-fisica/master/imagenes/qqplot.png" width="500"/>

**Interpretación:**  
Este gráfico evalúa si los residuales siguen una distribución normal. Como los puntos se alinean aproximadamente con la línea de referencia (45°), se sugiere que los errores del modelo se distribuyen normalmente, lo cual valida el supuesto de normalidad.

---

#### Histograma de residuales  
<img src="https://raw.githubusercontent.com/ValentinoAlvarado/deteccion-centroides-ruedas-fisica/master/imagenes/hist_residuales.png" width="450"/>

**Interpretación:**  
El histograma muestra una forma aproximadamente simétrica y de campana, indicando una distribución cercana a la normal para los residuales. Esto apoya el uso de inferencia basada en mínimos cuadrados.

---

#### Boxplot de residuales  
<img src="https://raw.githubusercontent.com/ValentinoAlvarado/deteccion-centroides-ruedas-fisica/master/imagenes/boxplot.png" width="400"/>

**Interpretación:**  
El boxplot confirma que los residuales están centrados cerca de cero y no hay presencia significativa de outliers extremos. Esto es indicativo de una distribución razonablemente controlada.

---

#### Residuales estandarizados vs ajustados  
<img src="https://raw.githubusercontent.com/ValentinoAlvarado/deteccion-centroides-ruedas-fisica/master/imagenes/residuales_ajustados.png" width="520"/>

**Interpretación:**  
Este gráfico sirve para detectar heterocedasticidad. Dado que los puntos no muestran un patrón cónico o curvo evidente, se sugiere que la varianza de los errores es aproximadamente constante a lo largo de los valores ajustados.

---

#### Residuales estandarizados vs distancia d  
<img src="https://raw.githubusercontent.com/ValentinoAlvarado/deteccion-centroides-ruedas-fisica/master/imagenes/residuales_distancia_d.png" width="520"/>

**Interpretación:**  
Aquí se explora si hay una relación entre la distancia multivariada de cada observación y sus residuos. Un patrón aleatorio sugiere que no hay dependencia estructural relacionada con la posición en el espacio de diseño.

---

#### Residuales vs orden  
<img src="https://raw.githubusercontent.com/ValentinoAlvarado/deteccion-centroides-ruedas-fisica/master/imagenes/residuales_orden.png" width="450"/>

**Interpretación:**  
Este gráfico permite detectar autocorrelación temporal. Si los puntos fluctúan aleatoriamente alrededor de cero, como en este caso, no hay evidencia clara de dependencia serial.

---

#### Residuales studentizados vs leverage  
<img src="https://raw.githubusercontent.com/ValentinoAlvarado/deteccion-centroides-ruedas-fisica/master/imagenes/studentized_vs_leverage.png" width="520"/>

**Interpretación:**  
Este gráfico combina información sobre la magnitud de los residuos y el apalancamiento. Observaciones alejadas horizontalmente (alto leverage) y verticalmente (residuales extremos) pueden ser influyentes. Aquí se observan algunos puntos con mayor leverage que deben ser revisados.

---

#### Residuales estandarizados vs distancia de Cook  
<img src="https://raw.githubusercontent.com/ValentinoAlvarado/deteccion-centroides-ruedas-fisica/master/imagenes/cookd_vs_residuals.png" width="520"/>

**Interpretación:**  
Este gráfico revela el posible impacto de cada observación sobre los coeficientes del modelo. La forma parabólica invertida observada es típica y esperada; sin embargo, puntos extremos en el eje de Cook pueden indicar observaciones influyentes.

---

#### Parabola con IC e IP  
<img src="https://raw.githubusercontent.com/ValentinoAlvarado/deteccion-centroides-ruedas-fisica/master/imagenes/ajuste_parabola_ic_ip.png" width="600"/>

**Interpretación:**  
La curva roja representa el ajuste cuadrático del modelo. Las bandas rojas punteadas son los intervalos de confianza del 95% (IC) para el valor promedio, mientras que las bandas verdes representan los intervalos de predicción del 95% (IP) para nuevas observaciones. El modelo parece ajustarse bien al conjunto de datos.
