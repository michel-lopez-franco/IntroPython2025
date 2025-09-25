# Práctica de Visión por Computadora para Robótica con scikit-image

## Objetivo

Desarrollar habilidades en procesamiento de imágenes aplicadas a robótica usando scikit-image, simulando tareas típicas que un robot necesita realizar.

## Configuración Inicial

```python
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, filters, measure, morphology, segmentation
from skimage.color import rgb2gray, rgb2hsv
from skimage.feature import canny, corner_harris, corner_peaks
from skimage.transform import hough_line, hough_line_peaks, hough_circle
from skimage.util import random_noise
import cv2  # Para algunas funciones complementarias
```

---

## Ejercicio 1: Detección de Objetos por Color (Clasificación de Piezas)

**Escenario**: Un robot debe clasificar piezas de diferentes colores en una banda transportadora.

**Tareas**:

1. Carga una imagen con objetos de diferentes colores
2. Convierte a espacio de color HSV
3. Crea máscaras para detectar objetos rojos, azules y verdes
4. Cuenta el número de objetos de cada color
5. Calcula el centroide de cada objeto

```python
def detectar_objetos_por_color(imagen):
    """
    Detecta y clasifica objetos por color
    """
    # Tu código aquí
    pass

# Imagen de prueba (puedes usar una imagen sintética o real)
# imagen = io.imread('piezas_colores.jpg')
```

**Criterios de evaluación**:

- Precisión en la detección (>90%)
- Robustez ante variaciones de iluminación
- Cálculo correcto de centroides

---

## Ejercicio 2: Navegación por Líneas (Seguimiento de Trayectoria)

**Escenario**: Un robot móvil debe seguir una línea en el suelo para navegar.

**Tareas**:

1. Detecta líneas usando la transformada de Hough
2. Calcula el ángulo de la línea principal
3. Determina si el robot debe girar izquierda/derecha
4. Estima la distancia al centro de la línea

```python
def seguir_linea(imagen):
    """
    Procesa imagen para seguimiento de línea
    Retorna: ángulo_giro, distancia_centro
    """
    # Preprocesamiento
    gray = rgb2gray(imagen)

    # Detección de bordes
    edges = canny(gray, sigma=2, low_threshold=0.1, high_threshold=0.2)

    # Tu código para Hough Transform aquí

    return angulo_giro, distancia_centro
```

**Reto adicional**: Maneja intersecciones y bifurcaciones.

---

## Ejercicio 3: Detección y Medición de Objetos (Control de Calidad)

**Escenario**: Un robot industrial debe medir dimensiones de piezas manufacturadas.

**Tareas**:

1. Segmenta objetos del fondo
2. Calcula área, perímetro y dimensiones principales
3. Detecta defectos (agujeros, irregularidades)
4. Clasifica piezas como "aprobadas" o "defectuosas"

```python
def control_calidad(imagen, tolerancia_area=0.1):
    """
    Analiza piezas para control de calidad
    """
    # Segmentación
    gray = rgb2gray(imagen)
    thresh = filters.threshold_otsu(gray)
    binary = gray > thresh

    # Análisis morfológico
    cleaned = morphology.remove_small_objects(binary, min_size=100)

    # Tu código para mediciones aquí

    return resultados_medicion
```

**Métricas a calcular**:

- Área en píxeles y mm²
- Relación aspecto (largo/ancho)
- Circularidad: 4π×área/perímetro²
- Número de agujeros

---

## Ejercicio 4: Reconocimiento de Formas Geométricas

**Escenario**: Un robot debe clasificar y manipular objetos según su forma.

**Tareas**:

1. Detecta contornos de objetos
2. Clasifica formas: círculo, cuadrado, triángulo, rectángulo
3. Calcula orientación para agarre robótico
4. Determina puntos de agarre óptimos

```python
def clasificar_formas(imagen):
    """
    Clasifica objetos por su forma geométrica
    """
    # Detección de contornos
    gray = rgb2gray(imagen)
    edges = canny(gray, sigma=1)

    # Análisis de forma
    contours = measure.find_contours(edges, 0.5)

    formas_detectadas = []

    for contour in contours:
        # Tu código para clasificación aquí
        # Usa aproximación poligonal, análisis de curvatura, etc.
        pass

    return formas_detectadas
```

**Algoritmos sugeridos**:

- Aproximación de Douglas-Peucker
- Análisis de momentos de Hu
- Descriptores de Fourier

---

## Ejercicio 5: Localización y Mapeo Visual (SLAM Básico)

**Escenario**: Un robot debe crear un mapa del entorno usando características visuales.

**Tareas**:

1. Detecta esquinas y puntos de interés
2. Calcula descriptores locales
3. Realiza correspondencia entre frames
4. Estima movimiento de la cámara

```python
def detectar_caracteristicas(imagen):
    """
    Detecta y describe características visuales
    """
    gray = rgb2gray(imagen)

    # Detección de esquinas Harris
    corners = corner_harris(gray)
    coords = corner_peaks(corners, min_distance=20, threshold_rel=0.1)

    # Tu código para descriptores aquí

    return coords, descriptores

def estimar_movimiento(imagen1, imagen2):
    """
    Estima el movimiento entre dos frames
    """
    # Tu implementación aquí
    pass
```

---

## Ejercicio 6: Manipulación Robótica Guiada por Visión

**Escenario**: Un brazo robótico debe recoger objetos específicos de una mesa desordenada.

**Tareas**:

1. Segmenta objetos individuales
2. Identifica el objeto objetivo
3. Calcula pose 3D aproximada
4. Determina estrategia de agarre

```python
def planificar_agarre(imagen, objeto_objetivo):
    """
    Planifica el agarre de un objeto específico
    """
    # Segmentación de instancias
    labels = segmentation.watershed(...)

    # Identificación del objeto
    objeto_id = identificar_objeto(labels, objeto_objetivo)

    # Cálculo de pose y agarre
    pose_3d = calcular_pose(objeto_id)
    puntos_agarre = calcular_agarre(objeto_id)

    return pose_3d, puntos_agarre
```

---

## Ejercicio Integrador: Sistema de Visión Completo

**Escenario**: Desarrolla un sistema completo que combine múltiples capacidades.

**Requisitos**:

1. Interfaz gráfica para visualización en tiempo real
2. Calibración automática de cámara
3. Detección robusta ante cambios de iluminación
4. Logging de eventos y errores
5. Optimización para tiempo real

```python
class SistemaVisionRobotica:
    def __init__(self):
        self.calibracion = None
        self.parametros = {}

    def calibrar_camara(self, imagenes_calibracion):
        """Calibra parámetros intrínsecos de la cámara"""
        pass

    def procesar_frame(self, imagen):
        """Procesa un frame completo"""
        resultados = {
            'objetos_detectados': [],
            'lineas_navegacion': [],
            'calidad_piezas': [],
            'caracteristicas_slam': []
        }
        return resultados

    def ejecutar_mision(self, tipo_mision):
        """Ejecuta una misión específica"""
        pass
```

---

## Criterios de Evaluación General

### Funcionalidad (40%)

- Correcta implementación de algoritmos
- Robustez ante condiciones variables
- Precisión en detecciones y mediciones

### Código (30%)

- Estructura y organización
- Documentación y comentarios
- Manejo de errores

### Innovación (20%)

- Optimizaciones propuestas
- Características adicionales
- Uso creativo de técnicas

### Presentación (10%)

- Visualizaciones claras
- Análisis de resultados
- Conclusiones técnicas

---

## Recursos Adicionales

### Datasets Sugeridos

- Piezas industriales: [Enlace a dataset]
- Navegación indoor: [Enlace a dataset]
- Objetos domésticos: [Enlace a dataset]

### Herramientas Complementarias

- OpenCV para funciones avanzadas
- PCL para nubes de puntos
- ROS para integración robótica

### Lecturas Recomendadas

- "Computer Vision for Robotics" - Corke
- "Multiple View Geometry" - Hartley & Zisserman
- Documentación oficial de scikit-image

---

## Entrega

**Formato**: Jupyter Notebook con código, visualizaciones y análisis
**Fecha límite**: [Especificar fecha]
**Presentación**: Demo en vivo de 10 minutos por equipo

¡Buena suerte con el desarrollo de sus sistemas de visión robótica!
