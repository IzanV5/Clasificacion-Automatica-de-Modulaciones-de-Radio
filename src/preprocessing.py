import numpy as np

def calc_fft_log_b(X_raw):
    # Para Modelo B (Distribuidor): Usa la mitad (2048)
    freq = np.abs(np.fft.fft(X_raw, axis=1))[:, :2048]
    return np.log1p(freq)

def calc_fft_log_olivia(X_raw):
    # Para Modelo Olivia: Usa FFT COMPLETA (4096) + Epsilon
    # np.log(np.abs(np.fft.fft(signals)) + 1e-127)
    N = 2048
    x_r = X_raw[:, :N]
    x_i = X_raw[:, N:2*N]
    freq = np.abs(np.fft.fft(x_r + 1j * x_i, axis=1))
    return np.log(freq + 1e-127)

def calc_envelope_fft_psk(X_raw):
    # Para Modelo PSK: Envelope
    N = 2048
    x_r = X_raw[:, :N]
    x_i = X_raw[:, N:2*N]
    x_mag = np.sqrt(x_r**2 + x_i**2)
    x_mag_cent = x_mag - np.mean(x_mag, axis=1, keepdims=True)
    return x_mag_cent

def calc_envelope_fft_rtty(X_raw):
    # Para RTTY: Envelope FFT recortada (bajas frecuencias)
    mid = X_raw.shape[1] // 2
    x_r = X_raw[:, :mid]
    x_i = X_raw[:, mid:]
    x_mag = np.sqrt(x_r**2 + x_i**2)
    x_mag_cent = x_mag - np.mean(x_mag, axis=1, keepdims=True)
    x_env = np.abs(np.fft.fft(x_mag_cent, axis=1))[:, :200]
    return x_env