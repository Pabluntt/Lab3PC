import pandas as pd
import numpy as np

def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors='coerce')
    return df

def cargar_diario(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_datetime_index(df.copy())
    cols = [c for c in ['CO(GT)', 'NO2(GT)'] if c in df.columns]
    d = df[cols].apply(pd.to_numeric, errors='coerce').resample('D').mean()
    return d.dropna(how='all')

def cat_co(v: float) -> str | None:
    if np.isnan(v): return None
    return 'Buena' if v <= 4 else ('Moderada' if v <= 10 else 'Mala')

def cat_no2(v: float) -> str | None:
    if np.isnan(v): return None
    return 'Buena' if v <= 40 else ('Moderada' if v <= 100 else 'Mala')

def peor_cat(row: pd.Series) -> str | float:
    orden = {'Buena': 0, 'Moderada': 1, 'Mala': 2}
    cats = []
    if 'CO(GT)' in row:
        c = cat_co(row['CO(GT)'])
        if c: cats.append(c)
    if 'NO2(GT)' in row:
        c = cat_no2(row['NO2(GT)'])
        if c: cats.append(c)
    if not cats:
        return np.nan
    return max(cats, key=lambda x: orden[x])

def clasificar_diario(df: pd.DataFrame) -> pd.DataFrame:
    d = cargar_diario(df)
    out = d.copy()
    if 'CO(GT)' in d.columns: out['CO_cat'] = d['CO(GT)'].apply(cat_co)
    if 'NO2(GT)' in d.columns: out['NO2_cat'] = d['NO2(GT)'].apply(cat_no2)
    out['Calidad'] = d.apply(peor_cat, axis=1)
    return out