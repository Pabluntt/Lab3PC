import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = '../data/dataset/AirQuality_procesado.csv'
OUT_DIR = '../data/modelos/modelamiento2'
os.makedirs(OUT_DIR, exist_ok=True)

def _load_df(path=DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, index_col='DateTime', parse_dates=True)
    for c in df.columns:
        if df[c].dtype == 'object':
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def fit_linear(x: np.ndarray, y: np.ndarray):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    m = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[m], y[m]
    if len(x) < 10:
        return np.array([np.nan, np.nan])
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta 

def main():
    df = _load_df()
    if not {'PT08.S2(NMHC)', 'NMHC(GT)'}.issubset(df.columns):
        print("Faltan columnas PT08.S2(NMHC) o NMHC(GT)")
        return

    daily = df[['PT08.S2(NMHC)', 'NMHC(GT)']].dropna()
    if isinstance(daily.index, pd.DatetimeIndex):
        daily = daily.resample('D').mean().dropna()

    b0, b1 = fit_linear(daily['PT08.S2(NMHC)'], daily['NMHC(GT)'])
    y_hat = b0 + b1 * daily['PT08.S2(NMHC)']
    resid = daily['NMHC(GT)'] - y_hat

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(daily.index, resid, alpha=0.5, label='Residuales')
    ma = resid.rolling(30, min_periods=5, center=True).mean()
    ax.plot(daily.index, ma, color='red', label='Media móvil 30 días')
    ax.axhline(0, color='k', linestyle='--', linewidth=1)
    ax.set_title('Deriva del sensor NMHC: residuales vs tiempo')
    ax.set_xlabel('Fecha'); ax.set_ylabel('NMHC(GT) - predicción')
    ax.grid(True); ax.legend()
    fig.tight_layout()
    p1 = os.path.join(OUT_DIR, 'nmhc_residuales_mov30.png')
    fig.savefig(p1, dpi=220); plt.close(fig)

    coeffs_m = []
    for g, sub in daily.groupby(pd.Grouper(freq='M')):
        beta = fit_linear(sub['PT08.S2(NMHC)'], sub['NMHC(GT)'])
        coeffs_m.append((g, *beta))
    coeffs_m = pd.DataFrame(coeffs_m, columns=['fecha', 'intercepto', 'pendiente']).dropna()

    coeffs_q = []
    for g, sub in daily.groupby(pd.Grouper(freq='Q')):
        beta = fit_linear(sub['PT08.S2(NMHC)'], sub['NMHC(GT)'])
        coeffs_q.append((g, *beta))
    coeffs_q = pd.DataFrame(coeffs_q, columns=['fecha', 'intercepto', 'pendiente']).dropna()

    for name, coeffs in [('mensual', coeffs_m), ('trimestral', coeffs_q)]:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(coeffs['fecha'], coeffs['pendiente'], marker='o')
        ax.set_title(f'Sensibilidad del sensor NMHC (pendiente) - {name}')
        ax.set_xlabel('Fecha'); ax.set_ylabel('Pendiente (β1)')
        ax.grid(True); fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f'nmhc_pendiente_{name}.png'), dpi=220)
        plt.close(fig)

    rmse_m = []
    for g, sub in daily.groupby(pd.Grouper(freq='M')):
        if len(sub) < 10: continue
        b0, b1 = fit_linear(sub['PT08.S2(NMHC)'], sub['NMHC(GT)'])
        y_pred = b0 + b1 * sub['PT08.S2(NMHC)']
        rmse = np.sqrt(np.mean((sub['NMHC(GT)'] - y_pred) ** 2))
        rmse_m.append((g, rmse))
    rmse_m = pd.DataFrame(rmse_m, columns=['fecha', 'rmse'])

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(rmse_m['fecha'], rmse_m['rmse'], marker='o', color='tab:orange')
    ax.set_title('RMSE mensual del ajuste NMHC(GT) ~ PT08.S2(NMHC)')
    ax.set_xlabel('Fecha'); ax.set_ylabel('RMSE')
    ax.grid(True); fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'nmhc_rmse_mensual.png'), dpi=220)
    plt.close(fig)

    print("Modelamiento II generado en:", OUT_DIR)

if __name__ == '__main__':
    main()