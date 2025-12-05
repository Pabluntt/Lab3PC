import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

def load_and_prepare_data(raw_path='../air+quality/AirQualityUCI.csv', output_dir='../data/dataset'):
    try:
        df = pd.read_csv(raw_path, sep=';', decimal=',')
        print("Archivo CSV cargado exitosamente.")
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo '{raw_path}'.")
        return None

    df = df.iloc[:, :-2]
    df.replace(to_replace=-200, value=pd.NA, inplace=True)
    df.replace(to_replace=-200, value=pd.NA, inplace=True)
    df = df.sort_values(['Date', 'Time'])
    df = df.ffill().bfill()

    for col in df.columns:
        if col not in ['Date', 'Time']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.fillna(df.median(numeric_only=True))

    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'].str.replace('.', ':', regex=False), dayfirst=True)
    df['Hour'] = df['DateTime'].dt.hour
    num_cols = df.select_dtypes(include='number').columns
    df[num_cols] = df.groupby('Hour')[num_cols].transform(lambda s: s.fillna(s.mean()))

    df.dropna(inplace=True)

    df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'].str.replace('.', ':', regex=False), dayfirst=True)
    df['Hour'] = df['DateTime'].dt.hour
    df['DayOfWeek'] = df['DateTime'].dt.day_name()
    df['Month'] = df['DateTime'].dt.month_name()
    df['IsWeekend'] = df['DateTime'].dt.dayofweek >= 5
    df.set_index('DateTime', inplace=True)
    df.drop(['Date', 'Time'], axis=1, inplace=True)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'AirQuality_procesado.csv')
    df.to_csv(output_path, sep=',', decimal='.', index=True)
    print(f"Dataset procesado guardado en: {output_path}")

    return df

if __name__ == "__main__":
    df = load_and_prepare_data()
    if df is not None:
        print("\n--- Primeras 5 filas del dataset ---")
        print(df.head())
        print("\n--- Resumen estadístico ---")
        columnas_interes = ['CO(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)']
        print(df[columnas_interes].describe())
        