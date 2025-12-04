import joblib
import numpy as np
from .config import MODELS_DIR
from .preprocessing import (
    calc_fft_log_b, 
    calc_fft_log_olivia, 
    calc_envelope_fft_psk, 
    calc_envelope_fft_rtty
)

class CascadingSystem:
    def __init__(self):
        self.loaded = False
        self.models = {}
        self.encoders = {}

    def load_models(self):
        print("🏗️ CARGANDO SISTEMA FINAL (INTEGRACIÓN EQUIPO)...")
        
        # 1. Portero
        pkg_port = joblib.load(MODELS_DIR / 'modelo_xgboost_local_categorias.joblib')
        self.models['portero'] = pkg_port['modelo']

        # 2. Rama A (Raw)
        pkg_A = joblib.load(MODELS_DIR / 'modelo_xgboost_local_cat_A.joblib')
        self.models['A'] = pkg_A['modelo']
        self.encoders['A'] = pkg_A['encoder']

        # 3. Rama B (Distribuidor - FFT Log Mitad)
        pkg_B = joblib.load(MODELS_DIR / 'modelo_unificado_fft.joblib')
        self.models['B'] = pkg_B['modelo']
        self.encoders['B'] = pkg_B['encoder']

        # 4. Especialista Olivia (FFT Log COMPLETA - 4096 cols)
        # Nota: Estos se cargan directo, no son dicts según tu snippet
        self.models['olivia'] = joblib.load(MODELS_DIR / 'entre_olivia_modelo.joblib')
        self.encoders['olivia'] = joblib.load(MODELS_DIR / 'entre_olivia_encoder.joblib')

        # 5. Especialista PSK (Envelope FFT)
        self.models['psk'] = joblib.load(MODELS_DIR / 'entre_psk_modelo.joblib')
        self.encoders['psk'] = joblib.load(MODELS_DIR / 'entre_psk_encoder.joblib')

        # 6. Decisor Resto (FFT Log Mitad + SNR)
        pkg_resto = joblib.load(MODELS_DIR / 'modelo_decisor_resto.joblib')
        self.models['resto'] = pkg_resto['modelo']
        self.encoders['resto'] = pkg_resto['encoder']

        # 7. Micro RTTY (Envelope FFT + SNR)
        pkg_micro_rtty = joblib.load(MODELS_DIR / 'modelo_micro_rtty_snr.joblib')
        self.models['micro_rtty'] = pkg_micro_rtty['modelo']

        self.loaded = True
        print("✅ Modelos cargados.")

    def predict(self, X_in, snr_in):
        """
        Versión vectorizada de 'ejecutar_sistema_final'
        """
        if not self.loaded:
            self.load_models()
            
        X_np = X_in.values
        snr_np = snr_in.values.reshape(-1, 1)
        X_con_snr = np.hstack((snr_np, X_np)) 
        
        # Array inicial
        preds = np.array(["PENDIENTE"] * len(X_np), dtype=object)
        
        # NIVEL 1: PORTERO
        decisiones_portero = self.models['portero'].predict(X_con_snr)
        idx_A = np.where(decisiones_portero == 1)[0]
        idx_B = np.where(decisiones_portero == 0)[0]
        
        # NIVEL 2-A: RAW
        if len(idx_A) > 0:
            p_num = self.models['A'].predict(X_con_snr[idx_A])
            preds[idx_A] = self.encoders['A'].inverse_transform(p_num)

        # NIVEL 2-B: DIGITALES (Distribuidor)
        if len(idx_B) > 0:
            X_fft_log_B = calc_fft_log_b(X_np[idx_B])
            p_num_B = self.models['B'].predict(X_fft_log_B)
            p_nom_B = self.encoders['B'].inverse_transform(p_num_B)
            preds[idx_B] = p_nom_B
            
            # --- RAMA OLIVIA (FFT COMPLETA) ---
            # Ojo: Usamos mask sobre p_nom_B que es un subset, así que mapeamos a índices globales
            mask_olivia = (p_nom_B == 'Familia Olivia')
            idx_global_olivia = idx_B[mask_olivia]
            
            if len(idx_global_olivia) > 0:
                feats_olivia = calc_fft_log_olivia(X_np[idx_global_olivia])
                p_num = self.models['olivia'].predict(feats_olivia)
                preds[idx_global_olivia] = self.encoders['olivia'].inverse_transform(p_num)

            # --- RAMA PSK (ENVELOPE FFT) ---
            mask_psk = (p_nom_B == 'Familia PSK')
            idx_global_psk = idx_B[mask_psk]
            
            if len(idx_global_psk) > 0:
                feats_psk = calc_envelope_fft_psk(X_np[idx_global_psk])
                p_num = self.models['psk'].predict(feats_psk)
                preds[idx_global_psk] = self.encoders['psk'].inverse_transform(p_num)

            # --- RAMA DOMINO ---
            mask_domino = (p_nom_B == 'Familia Domino')
            if np.any(mask_domino):
                # Asignación directa
                preds[idx_B[mask_domino]] = 'dominoex11'

            # --- RAMA RESTO ---
            mask_resto = np.char.find(p_nom_B.astype(str), 'RESTO') >= 0
            idx_global_resto = idx_B[mask_resto]
            
            if len(idx_global_resto) > 0:
                X_fft_resto = calc_fft_log_b(X_np[idx_global_resto])
                feats_resto = np.hstack((snr_np[idx_global_resto], X_fft_resto))
                
                p_num_R = self.models['resto'].predict(feats_resto)
                p_nom_R = self.encoders['resto'].inverse_transform(p_num_R)
                preds[idx_global_resto] = p_nom_R
                
                # --- MICRO RTTY ---
                mask_rtty = (p_nom_R == 'Familia RTTY')
                idx_global_rtty = idx_global_resto[mask_rtty]
                
                if len(idx_global_rtty) > 0:
                    x_env = calc_envelope_fft_rtty(X_np[idx_global_rtty])
                    feats_rtty = np.hstack((snr_np[idx_global_rtty], x_env))
                    
                    p_num_M = self.models['micro_rtty'].predict(feats_rtty)
                    p_nom_M = np.where(p_num_M == 1, 'rtty50_170', 'rtty45_170')
                    preds[idx_global_rtty] = p_nom_M

        return preds