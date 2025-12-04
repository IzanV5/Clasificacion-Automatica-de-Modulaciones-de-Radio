import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# Importaciones locales
from src.config import DATA_FILE, LABELS_ORDER
from src.inference import CascadingSystem

# --- FUNCIÓN DE NORMALIZACIÓN ---
def normalizar(t):
    """Unifica nombres de etiquetas para la evaluación"""
    t = str(t).strip().lower()
    if 'domino' in t and 'familia' in t: return 'dominoex11'
    return t

def main():
    print("=== EJECUTANDO MODELO FINAL (CASCADA) + REPORTE CSV ===")
    
    # 1. Instanciar y Cargar Sistema
    brain = CascadingSystem()
    try:
        brain.load_models()
    except Exception as e:
        print(f"❌ Error crítico cargando modelos: {e}")
        return

    # 2. Cargar Datos
    print(f"📂 Leyendo dataset: {DATA_FILE}...")
    try:
        df_full = pd.read_pickle(DATA_FILE)
    except FileNotFoundError:
        print(f"❌ No se encuentra el archivo: {DATA_FILE}")
        return

    # 3. Preparar Datos de Test
    # Usamos las mismas columnas que el entrenamiento
    signal_cols = [c for c in df_full.columns if str(c).startswith('sample_')]
    X_raw = df_full[signal_cols]
    
    # Detección robusta de columna SNR
    snr_col_name = ' snr' if ' snr' in df_full.columns else 'snr'
    snr_col = df_full[snr_col_name]
    
    y_true = df_full[' mode']

    print("✂️ Separando conjunto de Test (20%)...")
    # Importante: random_state=42 para que sea el mismo split que en el entrenamiento
    indices_train, indices_test = train_test_split(
        df_full.index, test_size=0.20, stratify=y_true, random_state=42
    )

    X_test_raw = X_raw.loc[indices_test]
    snr_test = snr_col.loc[indices_test]
    y_test = y_true.loc[indices_test]

    # 4. EJECUCIÓN (Inferencia Rápida)
    print(f"🚀 Clasificando {len(X_test_raw)} muestras...")
    start_time = time.time()
    
    y_pred_final = brain.predict(X_test_raw, snr_test)
    
    elapsed = time.time() - start_time
    print(f"⏱️ Tiempo total: {elapsed:.2f}s ({len(X_test_raw)/elapsed:.1f} muestras/s)")

    # 5. POST-PROCESADO
    y_test_norm = y_test.apply(normalizar)
    y_pred_norm = [normalizar(p) for p in y_pred_final]

    # 6. GENERAR Y GUARDAR CSV (Lo que pediste)
    print("💾 Guardando resultados detallados en CSV...")
    
    # Creamos un DataFrame con los resultados para exportar
    df_resultados = pd.DataFrame({
        'Indice_Original': indices_test,
        'Etiqueta_Real': y_test,
        'Etiqueta_Real_Norm': y_test_norm,
        'Prediccion_Sistema': y_pred_final,
        'Prediccion_Norm': y_pred_norm,
        'SNR': snr_test.values,
        'Acierto': (np.array(y_test_norm) == np.array(y_pred_norm))
    })
    
    csv_filename = "resultados_modelo_final.csv"
    df_resultados.to_csv(csv_filename, index=False)
    print(f"✅ CSV guardado: {csv_filename}")

    # 7. METRICAS Y MATRIZ
    print("\n--- REPORTE DE CLASIFICACIÓN ---")
    print(classification_report(y_test_norm, y_pred_norm))

    print("🎨 Generando Matriz de Confusión...")
    fig, ax = plt.subplots(figsize=(20, 20))
    
    ConfusionMatrixDisplay.from_predictions(
        y_test_norm, y_pred_norm, 
        labels=LABELS_ORDER,
        cmap='viridis', 
        normalize='true', # Muestra porcentajes (0 a 1)
        values_format='.2f', 
        xticks_rotation='vertical', 
        ax=ax
    )
    
    plt.title(f"Matriz de Confusión - Modelo Final (Test Set n={len(y_test)})")
    plt.tight_layout()
    plt.savefig("matriz_confusion_final.png")
    print("✅ Gráfico guardado: matriz_confusion_final.png")
    plt.show()

if __name__ == "__main__":
    main()