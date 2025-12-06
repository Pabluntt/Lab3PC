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

def _columns_for_gases(df: pd.DataFrame, gases_base: list[str], show: str) -> list[str]:
    cols = []
    show = show.lower()
    for base in gases_base:
        gt_col = f"{base}(GT)"
        pt08_candidates = [c for c in df.columns if c.startswith("PT08.")]
        sensor_map_keywords = {
            "CO": ["S1", "CO"],
            "NMHC": ["S2", "NMHC"],
            "NOx": ["S3", "NOx"],
            "NO2": ["S4", "NO2"],
            "O3": ["S5", "O3"],
            "C6H6": ["C6H6", "benzene"],  
        }
        keys = sensor_map_keywords.get(base, [base])
        pt_cols = [c for c in pt08_candidates if any(k in c for k in keys)]
        if show in ("gt", "ambos") and gt_col in df.columns:
            cols.append(gt_col)
        if show in ("sensores", "ambos"):
            cols.extend(pt_cols)
  
    seen = set(); out = []
    for c in cols:
        if c in df.columns and c not in seen:
            out.append(c); seen.add(c)
    return out

def generate_avg_timeseries(
    df: pd.DataFrame,
    gases=None, 
    *,
    mode: str = 'General',
    freq: str = 'D',
    window: int = 7,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    filename: str = 'avg_timeseries.png',
    save: bool = True,
    show: str = 'gt'  
):
    os.makedirs(output_dir, exist_ok=True)
    df2 = _ensure_datetime_index(_numeric_only(df))

    if mode == 'Laboral':
        df2 = df2[df2.index.dayofweek < 5]
    elif mode == 'Fin de semana':
        df2 = df2[df2.index.dayofweek >= 5]

    if gases is None or not gases:
        gases = ["CO", "NO2", "NOx", "C6H6", "NMHC", "O3"]
    cols = _columns_for_gases(df2, gases, show)
    if not cols:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'No hay columnas para la selección dada', ha='center', va='center')
        ax.axis('off')
        return fig, None

    dfr = df2[cols].resample(freq).mean()
    if window and window > 1:
        dfr = dfr.rolling(window, min_periods=1).mean()

    if dfr.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, 'Serie vacía tras resample/suavizado', ha='center', va='center')
        ax.axis('off')
        return fig, None

    fig, ax = plt.subplots(figsize=(12, 5))
    dfr.plot(ax=ax)
    ax.set_title(f"Tendencia de gases ({mode} • promedio {freq}, suavizado {window})")
    ax.set_xlabel('Fecha'); ax.set_ylabel('Concentración')
    ax.grid(True); fig.tight_layout()

    out_path = None
    if save:
        out_path = os.path.join(output_dir, filename)
        fig.savefig(out_path, dpi=220, bbox_inches='tight')
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