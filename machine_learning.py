import os
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Cargar el CSV si existe
csv_path = "melb_data.csv"
if not os.path.exists(csv_path):
    print(f"❌ Error: El archivo '{csv_path}' no se encontró.")
    exit()

df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()  # Eliminar espacios en nombres de columnas

# Definir columnas necesarias para el modelo
required_columns = ["Rooms", "Bathroom", "Landsize", "Lattitude", "Longtitude"]

# Verificar si las columnas existen y son numéricas
existing_columns = [col for col in required_columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]

if len(existing_columns) < len(required_columns):
    print("⚠️ Advertencia: No todas las columnas requeridas están disponibles o son numéricas.")
    print("Columnas encontradas:", existing_columns)
    print("Columnas faltantes o no numéricas:", set(required_columns) - set(existing_columns))

# Manejo de valores nulos
df = df.dropna(subset=existing_columns + ["Price"])

# Convertir a numérico si es necesario
for col in existing_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Seleccionar datos para entrenamiento
data_y = df["Price"]
data_x = df[existing_columns]

# Dividir los datos en entrenamiento y validación
train_X, val_X, train_Y, val_Y = train_test_split(data_x, data_y, random_state=1)

# Modelo: Árbol de Decisión
decision_tree_model = DecisionTreeRegressor(random_state=1)
decision_tree_model.fit(train_X, train_Y)
val_predictions = decision_tree_model.predict(val_X)
mae_dt = mean_absolute_error(val_Y, val_predictions)

print("\nDecision Tree Model:")
print(f"Mean Absolute Error (MAE): {mae_dt:.2f}")

# Evaluar diferentes tamaños de hojas
def get_mae(max_leaf_nodes, train_X, val_X, train_Y, val_Y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=1)
    model.fit(train_X, train_Y)
    preds_val = model.predict(val_X)
    return mean_absolute_error(val_Y, preds_val)

print("\nEvaluación con diferentes valores de max_leaf_nodes:")
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_Y, val_Y)
    print(f"Max leaf nodes: {max_leaf_nodes} → Mean Absolute Error: {my_mae:.2f}")

# Modelo: Random Forest
random_forest_model = RandomForestRegressor(n_estimators=100, random_state=1)
random_forest_model.fit(train_X, train_Y)
melb_preds = random_forest_model.predict(val_X)
mae_rf = mean_absolute_error(val_Y, melb_preds)

print("\nRandom Forest Model:")
print(f"Mean Absolute Error (MAE): {mae_rf:.2f}")
