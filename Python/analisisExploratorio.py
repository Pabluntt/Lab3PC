import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype

DEFAULT_OUTPUT_DIR = '../data/graficos'
PROCESSED_FILE = '../data/dataset/AirQuality_procesado.csv'

def _numeric_only(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if not is_numeric_dtype(out[c]) and out[c].dtype != 'bool':
            out[c] = pd.to_numeric(out[c], errors='coerce')
    return out

def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if not isinstance(out.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        out.index = pd.to_datetime(out.index)
    return out

def generate_correlation_heatmap(df: pd.DataFrame, output_dir: str = DEFAULT_OUTPUT_DIR, save: bool = True):
    os.makedirs(output_dir, exist_ok=True)
    df_num = (
        _numeric_only(df)
        .select_dtypes(include='number')
        .drop(["Hour", "DayOfWeek", "Month"], axis=1, errors='ignore')
    )
    corr = df_num.corr()

    if corr.empty or corr.shape[1] < 2:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, 'No hay suficientes columnas numéricas para correlación', ha='center', va='center')
        ax.axis('off')
        return fig, None

    fig, ax = plt.subplots(figsize=(18, 12))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5, ax=ax)
    ax.set_title('Mapa de Calor de Correlación de Variables Numéricas', fontsize=16)

    out_path = None
    if save:
        out_path = os.path.join(output_dir, 'correlation_heatmap.png')
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
    return fig, out_path

def generate_avg_timeseries(
    df: pd.DataFrame,
    gases=None,
    *,
    mode: str = 'General',   # 'General' | 'Laboral' | 'Fin de semana'
    freq: str = 'D',         # 'D' diaria, 'W' semanal, 'M' mensual
    window: int = 7,         # suavizado (media móvil)
    output_dir: str = DEFAULT_OUTPUT_DIR,
    filename: str = 'avg_timeseries.png',
    save: bool = True
):

    os.makedirs(output_dir, exist_ok=True)
    df2 = _ensure_datetime_index(_numeric_only(df))

    if gases is None:
        gases = [g for g in ['CO(GT)', 'NO2(GT)', 'NOx(GT)', 'C6H6(GT)', 'NMHC(GT)'] if g in df2.columns]
    gases = [g for g in gases if g in df2.columns]
    if not gases or df2.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, 'Sin datos/gases para graficar', ha='center', va='center')
        ax.axis('off')
        return fig, None

    mode_norm = mode.lower()
    if 'isweekend' in (c.lower() for c in df2.columns):
        col_exact = next(c for c in df2.columns if c.lower() == 'isweekend')
        if mode_norm == 'laboral':
            df2 = df2[df2[col_exact] == False]
        elif mode_norm == 'fin de semana':
            df2 = df2[df2[col_exact] == True]

    ts = (
        df2[gases]
        .resample(freq).mean()
        .rolling(window, min_periods=1, center=True).mean()
    )

    if ts.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, 'Sin datos tras el filtro seleccionado', ha='center', va='center')
        ax.axis('off')
        return fig, None

    fig, ax = plt.subplots(figsize=(14, 6))
    ts.plot(ax=ax)

    title_freq = {'D': 'diario', 'W': 'semanal', 'M': 'mensual'}.get(freq, freq)
    ax.set_title(f'Tendencia de gases ({mode} • promedio {title_freq}, suavizado {window})')
    ax.set_xlabel('Fecha'); ax.set_ylabel('Concentración')
    ax.grid(True); ax.legend()
    fig.tight_layout()

    out_path = None
    if save:
        out_path = os.path.join(output_dir, filename)
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
    return fig, out_path

if __name__ == '__main__':
    if not os.path.exists(PROCESSED_FILE):
        print(f"Error: El archivo '{PROCESSED_FILE}' no fue encontrado.")
        raise SystemExit(1)
    df = pd.read_csv(PROCESSED_FILE, index_col='DateTime', parse_dates=True)
    _, p1 = generate_correlation_heatmap(df, save=True)
    _, p2 = generate_avg_timeseries(
        df,
        gases=[g for g in ['CO(GT)', 'NO2(GT)', 'NOx(GT)', 'C6H6(GT)', 'NMHC(GT)'] if g in df.columns],
        compare_weekend=True,
        freq='D',
        window=7,
        filename='avg_timeseries_weekday_vs_weekend.png',
        save=True
    )
    for p in [p1, p2]:
        if p: print("Guardado:", p)