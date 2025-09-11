"""
Ejemplo Práctico: Análisis de Imágenes Médicas - Detección y Análisis de Células
===============================================================================

Este ejemplo demuestra cómo las operaciones morfológicas se aplican en el mundo real
para el análisis de imágenes médicas, específicamente para:
1. Limpiar ruido en imágenes de microscopía
2. Separar células que están tocándose
3. Rellenar huecos en estructuras celulares
4. Detectar bordes y contornos de células

Aplicaciones reales:
- Diagnóstico médico automatizado
- Conteo automático de células
- Detección de anomalías celulares
- Análisis de tejidos cancerosos
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology, measure, segmentation, filters
from scipy import ndimage
import cv2


def crear_imagen_simulada_celulas():
    """
    Crea una imagen simulada que representa células vistas bajo microscopio
    con ruido y artefactos típicos de imágenes médicas reales
    """
    # Crear imagen base de 400x400 píxeles
    img = np.zeros((400, 400), dtype=np.uint8)

    # Crear células circulares de diferentes tamaños
    centros_celulas = [
        (100, 100, 30),  # (x, y, radio)
        (200, 150, 25),
        (320, 80, 35),
        (150, 250, 28),
        (300, 280, 32),
        (80, 320, 22),
        (250, 300, 26),
    ]

    # Dibujar células como círculos
    for x, y, radio in centros_celulas:
        cv2.circle(img, (x, y), radio, 255, -1)

    # Crear algunas células que se tocan (problema común en microscopía)
    cv2.circle(img, (160, 180), 25, 255, -1)
    cv2.circle(img, (180, 190), 23, 255, -1)

    # Añadir núcleos dentro de las células
    for x, y, radio in centros_celulas:
        cv2.circle(img, (x + 5, y - 3), radio // 3, 128, -1)

    # Añadir ruido típico de imágenes médicas
    ruido = np.random.normal(0, 15, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + ruido, 0, 255).astype(np.uint8)

    # Añadir algunos píxeles dispersos (artefactos)
    for _ in range(50):
        x, y = np.random.randint(0, 400, 2)
        img[y, x] = 255

    # Crear algunos huecos en las células
    for x, y, _ in centros_celulas[:3]:
        cv2.circle(img, (x - 8, y + 5), 3, 0, -1)

    return img


def procesar_imagen_medica(imagen_original):
    """
    Aplica una serie de operaciones morfológicas para mejorar
    la imagen médica y prepararla para análisis
    """
    # Binarizar la imagen usando umbralización de Otsu
    umbral = filters.threshold_otsu(imagen_original)
    imagen_binaria = imagen_original > umbral

    # PASO 1: ELIMINACIÓN DE RUIDO
    # Usar opening para eliminar píxeles aislados y ruido pequeño
    elemento_estructural_pequeño = morphology.disk(2)
    sin_ruido = morphology.opening(imagen_binaria, elemento_estructural_pequeño)

    # PASO 2: RELLENAR HUECOS
    # Usar closing para rellenar pequeños huecos dentro de las células
    elemento_estructural_medio = morphology.disk(4)
    sin_huecos = morphology.closing(sin_ruido, elemento_estructural_medio)

    # PASO 3: SEPARACIÓN DE CÉLULAS QUE SE TOCAN
    # Usar erosión seguida de watershed para separar objetos conectados
    erodida = morphology.erosion(sin_huecos, morphology.disk(3))

    # Encontrar marcadores para watershed
    distancia = ndimage.distance_transform_edt(erodida)
    # Usar peak_local_max con min_distance
    from skimage.feature import peak_local_max

    coordenadas_maximos = peak_local_max(
        distancia, min_distance=20, threshold_abs=0.3 * distancia.max()
    )

    # Crear imagen de marcadores
    marcadores_img = np.zeros_like(distancia, dtype=bool)
    marcadores_img[tuple(coordenadas_maximos.T)] = True
    marcadores = measure.label(marcadores_img)

    # Aplicar watershed para separar células
    celulas_separadas = segmentation.watershed(-distancia, marcadores, mask=sin_huecos)

    return {
        "original": imagen_original,
        "binaria": imagen_binaria,
        "sin_ruido": sin_ruido,
        "sin_huecos": sin_huecos,
        "separadas": celulas_separadas > 0,
        "etiquetadas": celulas_separadas,
    }


def analizar_celulas(imagen_etiquetada):
    """
    Realiza análisis cuantitativo de las células detectadas
    """
    propiedades = measure.regionprops(imagen_etiquetada.astype(int))

    resultados = {
        "num_celulas": len(propiedades),
        "areas": [prop.area for prop in propiedades],
        "perimetros": [prop.perimeter for prop in propiedades],
        "circularidades": [],
        "centroides": [prop.centroid for prop in propiedades],
    }

    # Calcular circularidad (4π × área / perímetro²)
    for prop in propiedades:
        if prop.perimeter > 0:
            circularidad = 4 * np.pi * prop.area / (prop.perimeter**2)
            resultados["circularidades"].append(circularidad)
        else:
            resultados["circularidades"].append(0)

    return resultados


def detectar_bordes_celulares(imagen):
    """
    Usa operaciones morfológicas para detectar bordes de células
    """
    # Convertir imagen booleana a enteros para poder hacer la resta
    imagen_int = imagen.astype(np.uint8)

    # Gradiente morfológico para detectar bordes
    dilatada = morphology.dilation(imagen_int)
    erodida = morphology.erosion(imagen_int)
    bordes = dilatada - erodida

    return bordes


def ejemplo_aplicaciones_especificas():
    """
    Ejemplos de aplicaciones específicas de operaciones morfológicas
    en imágenes médicas
    """
    # Crear imagen base
    img = crear_imagen_simulada_celulas()

    # Binarizar
    umbral = filters.threshold_otsu(img)
    img_bin = img > umbral

    # Top-hat blanco: detectar pequeñas estructuras brillantes
    # Útil para detectar micocalcificaciones en mamografías
    tophat_blanco = morphology.white_tophat(img, morphology.disk(10))

    # Top-hat negro: detectar pequeñas estructuras oscuras
    # Útil para detectar vasos sanguíneos
    tophat_negro = morphology.black_tophat(img, morphology.disk(10))

    # Esqueletización: reducir estructuras a su forma más simple
    # Útil para análisis de vasos sanguíneos o redes neuronales
    esqueleto = morphology.skeletonize(img_bin)

    return {
        "original": img,
        "tophat_blanco": tophat_blanco,
        "tophat_negro": tophat_negro,
        "esqueleto": esqueleto,
    }


def visualizar_resultados():
    """
    Función principal que ejecuta todo el ejemplo y muestra los resultados
    """
    print("🔬 ANÁLISIS DE IMÁGENES MÉDICAS CON OPERACIONES MORFOLÓGICAS")
    print("=" * 65)

    # Crear imagen simulada
    imagen = crear_imagen_simulada_celulas()

    # Procesar la imagen
    resultados = procesar_imagen_medica(imagen)

    # Analizar células
    analisis = analizar_celulas(resultados["etiquetadas"])

    # Detectar bordes
    bordes = detectar_bordes_celulares(resultados["separadas"])

    # Aplicaciones específicas
    aplicaciones = ejemplo_aplicaciones_especificas()

    # Crear visualización
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(
        "Operaciones Morfológicas en Análisis de Imágenes Médicas", fontsize=16
    )

    # Fila 1: Proceso de limpieza
    axes[0, 0].imshow(resultados["original"], cmap="gray")
    axes[0, 0].set_title("1. Imagen Original\n(Microscopía simulada)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(resultados["sin_ruido"], cmap="gray")
    axes[0, 1].set_title("2. Después de Opening\n(Ruido eliminado)")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(resultados["sin_huecos"], cmap="gray")
    axes[0, 2].set_title("3. Después de Closing\n(Huecos rellenados)")
    axes[0, 2].axis("off")

    # Fila 2: Separación y análisis
    axes[1, 0].imshow(resultados["separadas"], cmap="gray")
    axes[1, 0].set_title("4. Células Separadas\n(Watershed)")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(resultados["etiquetadas"], cmap="tab20")
    axes[1, 1].set_title(
        f"5. Células Etiquetadas\n({analisis['num_celulas']} células detectadas)"
    )
    axes[1, 1].axis("off")

    axes[1, 2].imshow(bordes, cmap="gray")
    axes[1, 2].set_title("6. Bordes Detectados\n(Gradiente morfológico)")
    axes[1, 2].axis("off")

    # Fila 3: Aplicaciones específicas
    axes[2, 0].imshow(aplicaciones["tophat_blanco"], cmap="gray")
    axes[2, 0].set_title("7. Top-hat Blanco\n(Estructuras pequeñas)")
    axes[2, 0].axis("off")

    axes[2, 1].imshow(aplicaciones["tophat_negro"], cmap="gray")
    axes[2, 1].set_title("8. Top-hat Negro\n(Vasos/canales)")
    axes[2, 1].axis("off")

    axes[2, 2].imshow(aplicaciones["esqueleto"], cmap="gray")
    axes[2, 2].set_title("9. Esqueletización\n(Estructura simplificada)")
    axes[2, 2].axis("off")

    plt.tight_layout()
    plt.show()

    # Mostrar estadísticas del análisis
    print("\n📊 RESULTADOS DEL ANÁLISIS:")
    print(f"• Células detectadas: {analisis['num_celulas']}")
    print(f"• Área promedio: {np.mean(analisis['areas']):.1f} píxeles²")
    print(f"• Área mínima: {np.min(analisis['areas']):.1f} píxeles²")
    print(f"• Área máxima: {np.max(analisis['areas']):.1f} píxeles²")
    print(f"• Circularidad promedio: {np.mean(analisis['circularidades']):.3f}")

    print("\n🏥 APLICACIONES EN EL MUNDO REAL:")
    print("• Diagnóstico de cáncer: Detectar células anormales")
    print("• Hematología: Conteo automático de glóbulos rojos/blancos")
    print("• Cardiología: Análisis de tejido cardíaco")
    print("• Neurología: Estudio de neuronas y conexiones")
    print("• Oftalmología: Análisis de retina y vasos sanguíneos")
    print("• Patología: Clasificación automática de tejidos")


def ejemplo_caso_clinico():
    """
    Simula un caso clínico real: detección de células cancerosas
    """
    print("\n🩺 CASO CLÍNICO SIMULADO: Detección de Células Anormales")
    print("=" * 60)

    # Crear imagen con células normales y algunas anormales
    img = crear_imagen_simulada_celulas()

    # Procesar
    resultados = procesar_imagen_medica(img)
    analisis = analizar_celulas(resultados["etiquetadas"])

    # Clasificar células basándose en área y circularidad
    celulas_normales = 0
    celulas_sospechosas = 0

    for i, (area, circularidad) in enumerate(
        zip(analisis["areas"], analisis["circularidades"])
    ):
        # Criterios simplificados para ejemplo
        if area > 1000 and circularidad < 0.7:  # Células grandes e irregulares
            celulas_sospechosas += 1
            print(
                f"  ⚠️  Célula {i + 1}: SOSPECHOSA (Área: {area:.0f}, Circularidad: {circularidad:.3f})"
            )
        else:
            celulas_normales += 1
            print(
                f"  ✅ Célula {i + 1}: Normal (Área: {area:.0f}, Circularidad: {circularidad:.3f})"
            )

    print("\n📋 RESUMEN DEL DIAGNÓSTICO:")
    print(f"• Células normales: {celulas_normales}")
    print(f"• Células sospechosas: {celulas_sospechosas}")

    if celulas_sospechosas > 0:
        print("• RECOMENDACIÓN: Requiere revisión por especialista")
    else:
        print("• RECOMENDACIÓN: Muestra dentro de parámetros normales")


if __name__ == "__main__":
    # Ejecutar el ejemplo completo
    visualizar_resultados()
    ejemplo_caso_clinico()
