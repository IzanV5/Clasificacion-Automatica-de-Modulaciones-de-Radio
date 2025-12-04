# Clasificador de Modulación en Cascada para Radioaficionados (HF/UHF)

Este proyecto implementa un sistema de clasificación jerárquica ("Cascada") para identificar modos digitales de radioaficionado 
(Olivia, RTTY, PSK, DominoEX, etc.) a partir de señales I/Q crudas.

## Arquitectura

El sistema utiliza una estrategia de "Divide y Vencerás" con **7 modelos XGBoost** especializados:

1.  **Portero:** Separa señales simples (Grupo A) de complejas (Grupo B).
2.  **Especialistas:** Modelos dedicados para familias (PSK, Olivia) usando transformadas específicas (FFT Log, FFT Compleja, Envolvente).
3.  **Micro-Clasificadores:** Distinguen desplazamientos sutiles (ej: RTTY 45 vs 50 baudios).

## Estructura del Proyecto

- `src/`: Código fuente (Preprocesado DSP, Lógica de Inferencia).
- `models/`: Archivos .joblib serializados (XGBoost + LabelEncoders).
- `main.py`: Script principal de inferencia.
- `crear_dataset_sigidwiki.py`: Herramienta para importar audios .wav reales.

## Dataset

Debido al tamaño de lo archivos, el dataset no está incluido en el repositorio. Puedes descargarlo aquí:

**[Descargar Dataset (Google Drive)](https://drive.google.com/file/d/1Pf0nClO-mD01J43OlPht1l6oy6Ft2fAd/view?usp=sharing)**

## Instalación

1. Clonar el repositorio:
   ```bash
   git clone [https://github.com/IzanV5/Clasificacion-Automatica-de-Modulaciones-de-Radio.git](https://github.com/IzanV5/Clasificacion-Automatica-de-Modulaciones-de-Radio.git)

2. Instalar dependencias
   ```bash
   pip install -r requirements.txt

## Uso 

Coloca tu dataset (.pkl) en la carpeta data/ en la raiz del proyecto y ejecuta el comando:
   ```bash
   python main.py

## Resultados

El modelo alcanza un accuracy global del 81% en el dataset de prueba, resolviendo confusiones comunes entre modos de la misma 
familia gracias a la extracción de características específica por rama (Envelope FFT vs Complex FFT).