import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

try:
    df = pd.read_csv('../air+quality/AirQualityUCI.csv', sep=';', decimal=',')
    print("Archivo CSV cargado exitosamente.")
except FileNotFoundError:
    print("Error: No se encontró el archivo 'AirQualityUCI.csv'. Asegúrate de que esté en el mismo directorio.")
    exit()

df = df.iloc[:, :-2]
df.replace(to_replace=-200, value=pd.NA, inplace=True)
df.dropna(inplace=True)

df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'].str.replace('.', ':', regex=False), dayfirst=True)
df['Hour'] = df['DateTime'].dt.hour
df['DayOfWeek'] = df['DateTime'].dt.day_name()
df['Month'] = df['DateTime'].dt.month_name()
df['IsWeekend'] = df['DateTime'].dt.dayofweek >= 5
df.set_index('DateTime', inplace=True)
df.drop(['Date', 'Time'], axis=1, inplace=True)

print("\n--- Primeras 5 filas del dataset ---")
print(df.head())
print("\n--- Resumen estadístico de los principales gases ---")
columnas_interes = ['CO(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)']
print(df[columnas_interes].describe())

output_dir = '../data'
output_filename = 'AirQuality_procesado.csv'
output_path = os.path.join(output_dir, output_filename)
os.makedirs(output_dir, exist_ok=True)
df.to_csv(output_path, sep=',', decimal='.', index=True)
print(f"\nDataset procesado guardado en: {output_path}")

print("\nGenerando gráfico de concentraciones de CO, NO2 y C6H6")

plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(15, 7))

ax.plot(df.index, df['CO(GT)'], label='Monóxido de Carbono (CO)', color='red', alpha=0.8, marker='.', linestyle='-')
ax.plot(df.index, df['NO2(GT)'], label='Dióxido de Nitrógeno (NO2)', color='blue', alpha=0.8, marker='.', linestyle='-')
ax.plot(df.index, df['C6H6(GT)'], label='Benceno (C6H6)', color='green', alpha=0.7, marker='.', linestyle='-')
ax.set_title('Concentración de Gases en el Tiempo', fontsize=16)
ax.set_xlabel('Fecha', fontsize=12)
ax.set_ylabel('Concentración (mg/m^3)', fontsize=12)
ax.legend()

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()