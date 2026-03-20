"""
modelos_forecasting.py
──────────────────────
Módulo con la lógica de SARIMA y Prophet simplificados.

Este archivo debe estar en la misma carpeta que el cuaderno .ipynb.
Al tenerlo como módulo separado, macOS puede importarlo correctamente
en los procesos paralelos (soluciona el error de spawn en Jupyter).
"""

import numpy as np
import pandas as pd
from scipy.linalg import lstsq

# ─────────────────────────────────────────────────────────────
# Configuración global (editar aquí si es necesario)
# ─────────────────────────────────────────────────────────────
PERIODOS_FUTURO   = 10   # 5 años × 2 semestres
N_TEST_VALIDACION = 4    # Últimos 4 semestres para calcular métricas


# ─────────────────────────────────────────────────────────────
# Funciones auxiliares
# ─────────────────────────────────────────────────────────────

def calcular_metricas(real, pred):
    """Calcula MAE, MSE y MAPE entre valores reales y predichos."""
    real  = np.array(real,  dtype=float)
    pred  = np.array(pred,  dtype=float)
    denom = np.where(real != 0, real, 1e-6)  # evita división por cero en MAPE

    mae  = float(np.mean(np.abs(real - pred)))
    mse  = float(np.mean((real - pred) ** 2))
    mape = float(np.mean(np.abs((real - pred) / denom)) * 100)
    return mae, mse, mape


def fechas_futuras(ultima_fecha, pasos=10):
    """Genera las fechas de los próximos `pasos` semestres a partir de ultima_fecha."""
    fechas = []
    year   = ultima_fecha.year
    month  = ultima_fecha.month
    for _ in range(pasos):
        if month == 1:
            month = 7
        else:
            year += 1
            month = 1
        fechas.append(pd.Timestamp(f'{year}-{month:02d}-01'))
    return fechas


# ─────────────────────────────────────────────────────────────
# Modelo 1 — SARIMA simplificado
# ─────────────────────────────────────────────────────────────

class SimpleSARIMA:
    """
    SARIMA simplificado con periodo estacional = 2 (semestral).

    Descomposición:
        y(t) = tendencia(t) + estacionalidad(t) + residual(t)

    Tendencia    → regresión lineal: y = a + b·t
    Estacionalidad → promedio de residuales por semestre (periodo 2)
    Residuales   → componente AR(2) que se amortigua en el futuro
    """

    def __init__(self, series):
        self.series = np.array(series, dtype=float)
        # atributos que se llenan al descomponer
        self.coef_tendencia = None
        self.seasonal       = None
        self.residuales     = None

    # ── Descomposición ────────────────────────────────────────────────
    def _descomponer(self):
        y = self.series
        n = len(y)
        t = np.arange(n)

        # 1. Tendencia lineal: y ≈ a + b·t
        A = np.column_stack([np.ones(n), t])
        try:
            self.coef_tendencia, _, _, _ = lstsq(A, y)
        except Exception:
            self.coef_tendencia = [np.mean(y), 0.0]

        tendencia = self.coef_tendencia[0] + self.coef_tendencia[1] * t

        # 2. Componente estacional semestral (periodo = 2)
        detrended = y - tendencia
        seasonal  = np.zeros(2)
        counts    = np.zeros(2)
        for i in range(n):
            seasonal[i % 2] += detrended[i]
            counts[i % 2]   += 1
        self.seasonal  = np.where(counts > 0, seasonal / counts, 0.0)
        self.seasonal -= self.seasonal.mean()  # centrar en cero

        # 3. Residuales
        self.residuales = detrended - np.array([self.seasonal[i % 2] for i in range(n)])

    # ── Componente AR sobre residuales ────────────────────────────────
    def _ar_residual(self, p=2):
        """Devuelve la siguiente predicción AR(p) de los residuales."""
        r = self.residuales
        p = min(p, len(r) - 1)
        if p < 1:
            return 0.0
        X = np.column_stack([r[i: len(r) - p + i] for i in range(p)])
        y_r = r[p:]
        try:
            coefs, _, _, _ = lstsq(X, y_r)
        except Exception:
            return 0.0
        return float(np.dot(r[-p:], coefs))

    # ── Predicción ────────────────────────────────────────────────────
    def predecir(self, pasos=PERIODOS_FUTURO):
        y = self.series
        n = len(y)

        # Series muy cortas → solo tendencia simple
        if n < 6:
            pendiente = (y[-1] - y[0]) / max(n - 1, 1)
            return np.maximum([y[-1] + pendiente * (i + 1) for i in range(pasos)], 0)

        self._descomponer()
        ar_base = self._ar_residual()

        predicciones = []
        for i in range(pasos):
            t_fut      = n + i
            tendencia  = self.coef_tendencia[0] + self.coef_tendencia[1] * t_fut
            estacional = self.seasonal[t_fut % 2]
            # El componente AR se amortigua exponencialmente: no explota en el largo plazo
            ar_decay   = ar_base * (0.7 ** (i + 1))
            predicciones.append(tendencia + estacional + ar_decay)

        return np.maximum(predicciones, 0)

    # ── Validación cruzada (hold-out) ─────────────────────────────────
    def metricas_cv(self):
        """Entrena con los primeros n-N_TEST puntos y evalúa en los últimos N_TEST."""
        n      = len(self.series)
        n_test = min(N_TEST_VALIDACION, max(2, n // 3))
        train  = self.series[:-n_test]
        test   = self.series[-n_test:]
        preds  = SimpleSARIMA(train).predecir(pasos=n_test)
        return calcular_metricas(test, preds[:n_test])


# ─────────────────────────────────────────────────────────────
# Modelo 2 — Prophet simplificado
# ─────────────────────────────────────────────────────────────

class SimpleProphet:
    """
    Prophet simplificado: regresión lineal + estacionalidad de Fourier.

    La estacionalidad de Fourier usa senos y cosenos para capturar
    patrones periódicos de forma más suave que el promedio por semestre.

        y(t) = a + b·t + Σ [α_k·sin(2πkt/T) + β_k·cos(2πkt/T)] + ε
    """

    def __init__(self, series):
        self.series = np.array(series, dtype=float)
        self.coefs  = None

    # ── Features de Fourier ───────────────────────────────────────────
    @staticmethod
    def _fourier(t, periodo=2, n_terminos=1):
        """Genera columnas sin/cos para cada armónico k=1..n_terminos."""
        cols = []
        for k in range(1, n_terminos + 1):
            cols.append(np.sin(2 * np.pi * k * t / periodo))
            cols.append(np.cos(2 * np.pi * k * t / periodo))
        return np.column_stack(cols)

    # ── Predicción ────────────────────────────────────────────────────
    def predecir(self, pasos=PERIODOS_FUTURO):
        y = self.series
        n = len(y)

        # Series muy cortas → solo tendencia simple
        if n < 4:
            pendiente = (y[-1] - y[0]) / max(n - 1, 1)
            return np.maximum([y[-1] + pendiente * (i + 1) for i in range(pasos)], 0)

        t          = np.arange(n, dtype=float)
        n_terminos = 1 if n < 10 else 2   # más términos con más datos

        # Construir matriz de features: [1, t, sin1, cos1, ...]
        fourier = self._fourier(t, n_terminos=n_terminos)
        X = np.column_stack([np.ones(n), t, fourier])

        # Ajuste por mínimos cuadrados
        try:
            self.coefs, _, _, _ = lstsq(X, y)
        except Exception:
            self.coefs = np.zeros(X.shape[1])
            self.coefs[0] = np.mean(y)

        # Proyectar al futuro
        t_fut     = np.arange(n, n + pasos, dtype=float)
        fourier_f = self._fourier(t_fut, n_terminos=n_terminos)
        X_fut     = np.column_stack([np.ones(pasos), t_fut, fourier_f])

        return np.maximum(X_fut @ self.coefs, 0)

    # ── Validación cruzada (hold-out) ─────────────────────────────────
    def metricas_cv(self):
        n      = len(self.series)
        n_test = min(N_TEST_VALIDACION, max(2, n // 3))
        train  = self.series[:-n_test]
        test   = self.series[-n_test:]
        preds  = SimpleProphet(train).predecir(pasos=n_test)
        return calcular_metricas(test, preds[:n_test])


# ─────────────────────────────────────────────────────────────
# Función por grupo — unidad de trabajo paralelo
# ─────────────────────────────────────────────────────────────

def procesar_grupo(args):
    """
    Recibe (clave_grupo, sub_dataframe) y devuelve un dict con:
      - predicciones de SARIMA y Prophet
      - métricas de validación cruzada de cada modelo
      - el mejor modelo según score compuesto (MAE + RMSE + MAPE/100)

    Esta función vive en un módulo .py para que macOS pueda
    importarla correctamente en los procesos hijo (spawn).
    """
    clave, sub_df = args

    # Ordenar y extraer la serie temporal
    sub_df       = sub_df.sort_values('fecha')
    serie        = sub_df['matriculados'].values.astype(float)
    ultima_fecha = pd.Timestamp(sub_df['fecha'].iloc[-1])
    fdates       = fechas_futuras(ultima_fecha, pasos=PERIODOS_FUTURO)

    # ── SARIMA ────────────────────────────────────────────────────────
    sarima      = SimpleSARIMA(serie)
    pred_s      = sarima.predecir(pasos=PERIODOS_FUTURO)
    mae_s, mse_s, mape_s = sarima.metricas_cv()

    # ── Prophet ───────────────────────────────────────────────────────
    prophet     = SimpleProphet(serie)
    pred_p      = prophet.predecir(pasos=PERIODOS_FUTURO)
    mae_p, mse_p, mape_p = prophet.metricas_cv()

    # ── Seleccionar mejor modelo ───────────────────────────────────────
    # Score compuesto: combina las 3 métricas de forma balanceada
    score_s = mae_s + np.sqrt(mse_s) + mape_s / 100
    score_p = mae_p + np.sqrt(mse_p) + mape_p / 100

    mejor_modelo = 'SARIMA'  if score_s <= score_p else 'PROPHET'
    mejor_pred   = pred_s    if mejor_modelo == 'SARIMA' else pred_p

    return {
        'clave'         : clave,
        'serie'         : serie,
        'fechas_futuras': fdates,
        'pred_sarima'   : pred_s,
        'pred_prophet'  : pred_p,
        'mejor_pred'    : mejor_pred,
        'mejor_modelo'  : mejor_modelo,
        'metricas': {
            'sarima' : {'MAE': mae_s, 'MSE': mse_s, 'MAPE': mape_s},
            'prophet': {'MAE': mae_p, 'MSE': mse_p, 'MAPE': mape_p},
        }
    }
