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
st.text("El mapa de correlación nos muestra que tan fuerte es la relación lineal entre cada variable, teniendo en cuenta concentrarciones" \
"reales y las señales de los sensores. De esta forma podemos evidenciar y visualizar si sus valores son acordes, además de identificar" \
"que sensores son más propensos a verse afectados por características ambientales." \
"Por ejemplo puede apreciarse que el sensor NOx tiene una correlación negativa con todos los gases, inclusive con su propia concentración" \
"real, lo que podría indicar un funcionamiento bastante deficiente. Por otra parte, no se evidencia el hecho de que factores ambientales puedan" \
"afectar de gran manera a las concentraciones de gases ni sensores, a excepción del sensor de NO2, que se ve afectado de forma clara" \
"por la temperatura y la humedad absoluta, sin llenar a ser una correlación muy fuerte de todas maneras.")

st.header("Tendencia de gases (promedios)")
st.text("A continuación se presenta un gráfico interactivo, con el que puede verse la concentranción y tendencia real de los gases a lo largo" \
"del tiempo, además de permitir seleccionar diferentes frecuencias y suavizados para poder interpretar la información de mejor manera," \
"y escoger si quieren verse las tendencias de todos los días, solo días laborales o fines de semana.")
st.text("Por ejemplo puede apreciarse que en general los gases tienen conductas que se repiten en  periodos de tiempo cercanos a la semana," \
"por otra parte, también puede notarse que el gas NOx tiene tendencia a aumentar a partir de la segunda mitad de la medición.")
gas_base_all = ["CO", "NO2", "NOx", "C6H6", "NMHC", "O3"]
def _has_any(df, base: str) -> bool:
    return (f"{base}(GT)" in df.columns) or any(c.startswith("PT08.") and (base in c or (base=="O3" and "O3" in c)) for c in df.columns)
gases_opts = [g for g in gas_base_all if _has_any(df, g)]
gases_sel = st.multiselect("Gases", options=gases_opts, default=gases_opts)

col1, col2 = st.columns(2)
with col1:
    mode = st.selectbox("Modo", ["General", "Laboral", "Fin de semana"], index=0)
    opt = st.radio(
        "Datos a mostrar",
        ["Concentraciones reales (GT)", "Sensores MOX (PT08)", "Ambos"],
        horizontal=True
    )
    show_map = {
        "Concentraciones reales (GT)": "gt",
        "Sensores MOX (PT08)": "sensores",
        "Ambos": "ambos",
    }
    show_choice = show_map[opt]
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
    save=True,
    show=show_choice     
)
st.pyplot(fig_avg, use_container_width=True)

import glob
from modelamiento1 import univariate_models, multivariable_co, OUT_DIR as OUT_DIR_M1
from modelamiento2 import main as run_modelamiento2
OUT_DIR_M2 = '../data/modelos/modelamiento2'

st.header("Modelamiento I (calibración de sensores)")
st.text("Se generaron modelos de calibración para los sensores de los gases CO, NOx, NO2 y NMHC a través de modelos univariables para" \
"los gases NOx, NO2 y NMHC, y mutivariable para el CO. Los dos modelos ocupan ajustes lineales y polinomiales, y se esoje el que tenga mejor" \
"desempeño, basandose en su valor de R² y RMSE." \
"")
calib_imgs = sorted(glob.glob(os.path.join(OUT_DIR_M1, "*.png")))
if calib_imgs:
    mid = len(calib_imgs) // 2 if len(calib_imgs) > 1 else 1
    col_left, col_right = st.columns(2)
    with col_left:
        st.image(calib_imgs[:mid], caption=[os.path.basename(p) for p in calib_imgs[:mid]], use_container_width=True)
    with col_right:
        st.image(calib_imgs[mid:], caption=[os.path.basename(p) for p in calib_imgs[mid:]], use_container_width=True)
else:
    st.info(f"No hay imágenes de calibración en {OUT_DIR_M1}. Ejecuta por terminal: python Python\\modelamiento1.py")
    
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

from clasificacion import clasificar_diario
import matplotlib.pyplot as plt

st.header("Reporte: Clasificación de calidad del aire (CO y NO2)")
clasif = clasificar_diario(df)

st.text("Se clasificaron los días en tres categorías (Buena, Moderada y Mala) según los niveles diarios promedio de CO(GT) y NO2(GT), " \
"gases que afectan la calidad del aire según su concentración establecida por la OMS. Los intervalos de los gases indican: " )
st.text("- CO(GT): Buena ≤ 5, moderado 5–10, malo > 10 (aprox. 8 h EPA ≈ 10 mg/m³).")
st.text("- NO2(GT): Buena (<40 µg/m³), Moderada (40-100 µg/m³), Mala (>100 µg/m³)")
st.text("Para estimar la categoría del aire, se toma en cuenta cualquiera de los 2 gases que este en la peor calidad")

colA, colB = st.columns(2)
with colA:
    counts = clasif['Calidad'].value_counts(dropna=True)
    st.subheader("Conteo de días por categoría")
    st.write(counts.to_dict())

with colB:
    fig, ax = plt.subplots(figsize=(8, 3))
    y = clasif['Calidad'].map({'Buena':0, 'Moderada':1, 'Mala':2})
    ax.plot(clasif.index, y, drawstyle='steps-post')
    ax.set_yticks([0,1,2]); ax.set_yticklabels(['Buena','Moderada','Mala'])
    ax.set_title('Calidad del aire diaria')
    ax.grid(True, axis='y', alpha=0.3)
    st.pyplot(fig, use_container_width=True)

st.text("Podemos ver como increíblemente solo hubieron 2 días en todo el periodo de medición donde la calidad del aire fue buena," \
"seguida de 161 días de calidad moderada y 228 días con mala calidad del aire, lo que indica que en general la calidad del aire es muy mala" \
"teniendo en cuenta que la mayor arte del año se estuvo en una mala categoría. Además cabe destacar que a esar de que el CO nunca alcanzó niveles" \
"potencialmente peligrosos, la exposición prolongada a este gas es altamente dañina para la salud, reduciendo la cantidad de oxígeno que el " \
"cuerpo transporta en sangre.")

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
