import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Callable, Tuple
from scipy.optimize import curve_fit

DATA_PATH = '../data/dataset/AirQuality_procesado.csv'
OUT_DIR = '../data/modelos/modelamiento1'
os.makedirs(OUT_DIR, exist_ok=True)

def _load_df(path=DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, index_col='DateTime', parse_dates=True)
    for c in df.columns:
        if df[c].dtype == 'object':
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def linear(x, a, b):
    return a * x + b

def poly2(x, a, b, c):
    return a * x**2 + b * x + c

def fit_and_eval(x: np.ndarray, y: np.ndarray, f: Callable, name: str) -> Tuple[np.ndarray, float, float]:
   
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_fit, y_fit = x[mask], y[mask]
    if len(x_fit) < 20:
        raise ValueError(f"Insuficientes datos para {name}: {len(x_fit)}")
    
    popt, _ = curve_fit(f, x_fit, y_fit, maxfev=10000)
    
    y_pred = f(x_fit, *popt)
    rmse = float(np.sqrt(np.mean((y_pred - y_fit) ** 2)))
    r2 = float(1 - np.sum((y_pred - y_fit)**2) / np.sum((y_fit - np.mean(y_fit))**2))
    return popt, rmse, r2

def plot_calibration(x, y, f: Callable, popt, title: str, out_path: str):
    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, s=12, alpha=0.4, label='Datos')
    xs = np.linspace(np.nanmin(x), np.nanmax(x), 200)
    plt.plot(xs, f(xs, *popt), color='red', label='Ajuste')
    plt.title(title)
    plt.xlabel('Señal sensor (MOX)')
    plt.ylabel('Concentración real')
    plt.grid(True); plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def save_params(name: str, f_name: str, popt, rmse: float, r2: float):
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(OUT_DIR, f'{name}_params.txt'), 'w', encoding='utf-8') as fh:
        fh.write(f'modelo: {f_name}\ncoeficientes: {list(map(float, popt))}\nRMSE: {rmse:.3f}\nR2: {r2:.3f}\n')

def univariate_models(df: pd.DataFrame):
    # 1) PT08.S3(NOx) -> NOx(GT)
    if 'PT08.S3(NOx)' in df.columns and 'NOx(GT)' in df.columns:
        x = df['PT08.S3(NOx)'].to_numpy()
        y = df['NOx(GT)'].to_numpy()
        try:
            p_lin, rmse_lin, r2_lin = fit_and_eval(x, y, linear, 'NOx_linear')
        except Exception:
            p_lin, rmse_lin, r2_lin = None, np.inf, -np.inf
        try:
            p_p2, rmse_p2, r2_p2 = fit_and_eval(x, y, poly2, 'NOx_poly2')
        except Exception:
            p_p2, rmse_p2, r2_p2 = None, np.inf, -np.inf
        if r2_p2 > r2_lin:
            plot_calibration(x, y, poly2, p_p2, 'Calibración NOx: PT08.S3(NOx) → NOx(GT) (poly2)', os.path.join(OUT_DIR, 'calib_nox_poly2.png'))
            save_params('calib_nox', 'poly2', p_p2, rmse_p2, r2_p2)
        else:
            plot_calibration(x, y, linear, p_lin, 'Calibración NOx: PT08.S3(NOx) → NOx(GT) (lineal)', os.path.join(OUT_DIR, 'calib_nox_lineal.png'))
            save_params('calib_nox', 'linear', p_lin, rmse_lin, r2_lin)

    # 2) PT08.S2(NMHC) -> NMHC(GT)
    if 'PT08.S2(NMHC)' in df.columns and 'NMHC(GT)' in df.columns:
        x = df['PT08.S2(NMHC)'].to_numpy()
        y = df['NMHC(GT)'].to_numpy()
        p_p2, rmse_p2, r2_p2 = fit_and_eval(x, y, poly2, 'NMHC_poly2')
        plot_calibration(x, y, poly2, p_p2, 'Calibración NMHC: PT08.S2(NMHC) → NMHC(GT) (poly2)', os.path.join(OUT_DIR, 'calib_nmhc_poly2.png'))
        save_params('calib_nmhc', 'poly2', p_p2, rmse_p2, r2_p2)

    # 3) PT08.S4(NO2) -> NO2(GT)
    if 'PT08.S4(NO2)' in df.columns and 'NO2(GT)' in df.columns:
        x = df['PT08.S4(NO2)'].to_numpy()
        y = df['NO2(GT)'].to_numpy()
        try:
            p_lin, rmse_lin, r2_lin = fit_and_eval(x, y, linear, 'NO2_linear')
        except Exception:
            p_lin, rmse_lin, r2_lin = None, np.inf, -np.inf
        try:
            p_p2, rmse_p2, r2_p2 = fit_and_eval(x, y, poly2, 'NO2_poly2')
        except Exception:
            p_p2, rmse_p2, r2_p2 = None, np.inf, -np.inf
        if r2_p2 > r2_lin:
            plot_calibration(x, y, poly2, p_p2, 'Calibración NO2: PT08.S4(NO2) → NO2(GT) (poly2)', os.path.join(OUT_DIR, 'calib_no2_poly2.png'))
            save_params('calib_no2', 'poly2', p_p2, rmse_p2, r2_p2)
        else:
            plot_calibration(x, y, linear, p_lin, 'Calibración NO2: PT08.S4(NO2) → NO2(GT) (lineal)', os.path.join(OUT_DIR, 'calib_no2_lineal.png'))
            save_params('calib_no2', 'linear', p_lin, rmse_lin, r2_lin)

def multivariable_co(df: pd.DataFrame):
    # Objetivo CO(GT)
    target = 'CO(GT)'
    sensors = [c for c in ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)'] if c in df.columns]
    if target not in df.columns or len(sensors) == 0:
        return
    d = df[sensors + [target]].dropna()
    X = d[sensors].values.astype(float)
    y = d[target].values.astype(float)
    
    X1 = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    y_hat = X1 @ beta
    rmse = float(np.sqrt(np.mean((y_hat - y) ** 2)))
    r2 = float(1 - np.sum((y_hat - y)**2) / np.sum((y - np.mean(y))**2))
    
    coef = dict(intercept=float(beta[0]))
    for i, s in enumerate(sensors, start=1):
        coef[s] = float(beta[i])
    with open(os.path.join(OUT_DIR, 'co_multivariable_params.txt'), 'w', encoding='utf-8') as fh:
        fh.write(f'RMSE: {rmse:.3f}\nR2: {r2:.3f}\nCoeficientes:\n')
        for k, v in coef.items():
            fh.write(f'- {k}: {v:.6f}\n')
    
    plt.figure(figsize=(7, 5))
    plt.scatter(y, y_hat, s=12, alpha=0.5)
    plt.plot([min(y), max(y)], [min(y), max(y)], 'r--', label='y = ŷ')
    plt.xlabel('CO(GT) real'); plt.ylabel('CO(GT) predicho')
    plt.title('Modelo multivariable para CO(GT) con MOX (OLS)')
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'co_multivariable_scatter.png'), dpi=200)
    plt.close()

def main():
    df = _load_df()
    univariate_models(df)
    multivariable_co(df)
    print("Modelos generados en:", OUT_DIR)

if __name__ == '__main__':
    main()