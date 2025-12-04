import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

file_path = '../data/AirQuality_procesado.csv'
output_dir = '../data'

if not os.path.exists(file_path):
    print(f"Error: El archivo '{file_path}' no fue encontrado.")
    exit()

df = pd.read_csv(file_path, index_col='DateTime', parse_dates=True)
print("Dataset procesado cargado exitosamente.\n")

print("Calculando la matriz de correlación")

numeric_cols = df.select_dtypes(include=['float64', 'int64'])
correlation_matrix = numeric_cols.corr()

plt.figure(figsize=(18, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Mapa de Calor de Correlación de Variables Numéricas', fontsize=16)
plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
plt.close()

print("\n--- Cómo interpretar el mapa de calor ---")
print("Valores cercanos a 1.0 (rojo oscuro): Fuerte correlación positiva")
print("Valores cercanos a -1.0 (azul oscuro): Fuerte correlación negativa")
print("Valores cercanos a 0.0 (colores claros): Poca o ninguna correlación lineal\n")

print("Generando histogramas de distribución para variables clave...")

key_vars = ['CO(GT)', 'C6H6(GT)', 'NO2(GT)', 'T', 'RH', 'AH']
df[key_vars].hist(bins=30, figsize=(15, 10), layout=(2, 3))
plt.suptitle('Distribución de Variables Clave (Histogramas)', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96]) # Ajustar para que el supertítulo no se solape
plt.savefig(os.path.join(output_dir, 'key_variables_distribution.png'))
plt.close()


# --- 4. Análisis de Relaciones Específicas (Gráficos de Dispersión) ---
print("Generando gráficos de dispersión para analizar relaciones específicas...")

# Relación entre Temperatura (T) y Monóxido de Carbono (CO(GT))
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='T', y='CO(GT)', alpha=0.5)
plt.title('Temperatura vs. Concentración de CO(GT)', fontsize=14)
plt.xlabel('Temperatura (°C)')
plt.ylabel('Concentración de CO(GT)')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'scatter_temp_vs_co.png'))
plt.close()

# Relación entre Humedad Relativa (RH) y Monóxido de Carbono (CO(GT))
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='RH', y='CO(GT)', alpha=0.5)
plt.title('Humedad Relativa vs. Concentración de CO(GT)', fontsize=14)
plt.xlabel('Humedad Relativa (%)')
plt.ylabel('Concentración de CO(GT)')
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'scatter_rh_vs_co.png'))
plt.close()