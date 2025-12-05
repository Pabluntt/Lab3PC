import os
import pandas as pd
import streamlit as st
from pandas.api.types import is_numeric_dtype

from cargaDataset import load_and_prepare_data
from analisisExploratorio import (
    generate_correlation_heatmap,
    generate_avg_timeseries,
)

st.set_page_config(page_title="Análisis de presencia de gases", page_icon="🌬️", layout="wide")
st.title("Dashboard de Análisis de presencia de gases")
st.markdown(
    """
    <style>
    .stApp [data-testid="stDecoration"] { display: none; }
    .stApp [data-testid="stHeader"] { height: 0px; }
    .stApp [data-testid="stToolbar"] { display: none; }
    .block-container { max-width: 1250px; margin: 0 auto; }
    </style>
    """,
    unsafe_allow_html=True
)

IMAGES_DIR = '../data/graficos'
PROCESSED_PATH = '../data/dataset/AirQuality_procesado.csv'

def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if not is_numeric_dtype(out[col]) and df[col].dtype != 'bool':
            out[col] = pd.to_numeric(out[col], errors='coerce')
    return out

@st.cache_data
def get_data():
    if os.path.exists(PROCESSED_PATH):
        df = pd.read_csv(PROCESSED_PATH, index_col='DateTime', parse_dates=True)
    else:
        df = load_and_prepare_data(output_dir=os.path.dirname(PROCESSED_PATH))
    if df is None:
        return None
    return _coerce_numeric_columns(df)

with st.spinner("Cargando y preparando datos..."):
    df = get_data()

if df is None or df.empty:
    st.error("No se pudieron cargar los datos. Verifica el archivo RAW en ../air+quality/.")
    st.stop()

with st.expander("Ver datos procesados (tabla y resumen)", expanded=False):
    st.dataframe(df.head(50), use_container_width=True)
    st.write(df.describe(include='all'))

st.header("Mapa de correlación")
fig_corr, _ = generate_correlation_heatmap(df, output_dir=IMAGES_DIR, save=True)
st.pyplot(fig_corr, use_container_width=True)

st.header("Tendencia de gases (promedios)")
gases_opts = [g for g in ['CO(GT)', 'NO2(GT)', 'NOx(GT)', 'C6H6(GT)', 'NMHC(GT)'] if g in df.columns]
gases_sel = st.multiselect("Gases", options=gases_opts, default=gases_opts)

col1, col2 = st.columns(2)
with col1:
    mode = st.selectbox("Modo", ["General", "Laboral", "Fin de semana"], index=0)
with col2:
    freq = st.selectbox("Frecuencia", ["Diaria", "Semanal", "Mensual"], index=0)
freq_map = {"Diaria": "D", "Semanal": "W", "Mensual": "M"}
window = st.slider("Suavizado (ventana de períodos)", 3, 30, 7)

fig_avg, _ = generate_avg_timeseries(
    df,
    gases=gases_sel,
    mode=mode,
    freq=freq_map[freq],
    window=window,
    output_dir=IMAGES_DIR,
    filename='avg_timeseries.png',
    save=True
)
st.pyplot(fig_avg, use_container_width=True)

import glob
from modelamiento1 import univariate_models, multivariable_co, OUT_DIR as OUT_DIR_M1
from modelamiento2 import main as run_modelamiento2
OUT_DIR_M2 = '../data/modelos/modelamiento2'

st.header("Modelamiento I (calibración de sensores)")
col_m1a, col_m1b = st.columns(2)
with col_m1a:
    st.text("HOLA CARAJO")
with col_m1b:
    calib_imgs = sorted(glob.glob(os.path.join(OUT_DIR_M1, "*.png")))
    if calib_imgs:
        st.image(calib_imgs, caption=[os.path.basename(p) for p in calib_imgs], use_container_width=True)
    else:
        st.info(f"No hay imágenes de calibración en {OUT_DIR_M1}. Pulsa el botón para generarlas.")

st.header("Modelamiento II (deriva del sensor NMHC)")
col_m2a, col_m2b = st.columns(2)
with col_m2a:
    st.text("Se generaron 2 graficos para este modelamiento, en uno puede apreciarse la sensibilidad del sensor NMHC a lo largo del tiempo, " \
    "se escogió este sensor ya que su valor real de concentración era bastante estable a lo largo del tiempo, por lo que en teoría la señal" \
    "de este no debería variar tanto. En el otro gráfico se puede observar la RMSE mensual del ajuste entre la señal del sensor y la concentración real, " \
    "lo que es un indicador para saber que tan acertado era el funcionamiento del sensor." \
    "Con esto mencionado puede apreciarse como las sensibilidad del sensor tiene tendencia a disminuir a medida que pasa el tiempo, y a su vez" \
    "el RMSE aumenta significativamente en los últimos meses de medicion, lo que indicaría un claro desgaste en el sensor.")
with col_m2b:
    imgs_m2 = sorted(glob.glob(os.path.join(OUT_DIR_M2, "*.png")))
    if imgs_m2:
        st.image(imgs_m2, caption=[os.path.basename(p) for p in imgs_m2], use_container_width=True)
    else:
        st.info(f"No hay gráficos de Modelamiento II en {OUT_DIR_M2}. Pulsa el botón para generarlos.")

with st.expander("Visualizar todos los gráficos (incluso los que se generaron erroneamente a lo largo del desarrollo del laboratorio)", expanded=False):
    import glob
    import os

    os.makedirs(IMAGES_DIR, exist_ok=True)
    patterns = ("*.png", "*.jpg", "*.jpeg", "*.svg")
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(IMAGES_DIR, pat)))

    if not files:
        st.info(f"No se encontraron imágenes en {IMAGES_DIR}.")
    else:
        files = sorted(files, key=lambda p: os.path.getmtime(p), reverse=True)
        st.write(f"Imágenes encontradas: {len(files)}")
        st.image(files, caption=[os.path.basename(p) for p in files], use_container_width=True)
