import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

#  Cargar el CSV
csv_path = "melb_data.csv"
df = pd.read_csv(csv_path)

#  Eliminar espacios en los nombres de columnas
df.columns = df.columns.str.strip()

#  Eliminar filas con valores nulos en 'Price'
df = df.dropna(subset=["Price"])

#  Definir columnas necesarias para el modelo
required_columns = ["Rooms", "Bathroom", "Landsize", "Lattitude", "Longtitude"]

#  Verificar qué columnas existen en el dataset y son numericas
existing_columns = [col for col in required_columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]

if len(existing_columns) < len(required_columns):
    print("⚠️ Advertencia: No todas las columnas requeridas están disponibles o son numericas.")
    print("Columnas encontradas:", existing_columns)
    print("Columnas faltantes o no numericas:", set(required_columns) - set(existing_columns))

#  Seleccionar los datos para entrenamiento
data_y = df["Price"]
data_x = df[existing_columns]

#  Mostrar los primeros datos
print("\nPrimeras filas de X:")
print(data_x.head())

print("\nPrimeras filas de Y:")
print(data_y.head())

#  Dividir los datos en conjuntos de entrenamiento y validación
print("Ejecutando división de datos...")
train_X, val_X, train_Y, val_Y = train_test_split(data_x, data_y, random_state=1)
print("División de datos completada.")
print(f"train_X shape: {train_X.shape}")
print(f"val_X shape: {val_X.shape}")
print(f"train_Y shape: {train_Y.shape}")
print(f"val_Y shape: {val_Y.shape}")

#  **Modelo: Árbol de Decisión**
decision_tree_model = DecisionTreeRegressor(random_state=1)
decision_tree_model.fit(train_X, train_Y)
val_predictions = decision_tree_model.predict(val_X)
mae_dt = mean_absolute_error(val_Y, val_predictions)

print("\n Decision Tree Model:")
print(f"Mean Absolute Error (MAE): {mae_dt:.2f}")

#  Función para evaluar diferentes tamaños de hojas
def get_mae(max_leaf_nodes, train_X, val_X, train_Y, val_Y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=1)
    model.fit(train_X, train_Y)
    preds_val = model.predict(val_X)
    return mean_absolute_error(val_Y, preds_val)

#  Evaluar diferentes valores de `max_leaf_nodes`
print("\n Evaluación con diferentes valores de max_leaf_nodes:")
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_Y, val_Y)
    print(f"Max leaf nodes: {max_leaf_nodes} → Mean Absolute Error: {my_mae:.2f}")

#  **Modelo: Random Forest**
random_forest_model = RandomForestRegressor(random_state=1)
random_forest_model.fit(train_X, train_Y)
melb_preds = random_forest_model.predict(val_X)
mae_rf = mean_absolute_error(val_Y, melb_preds)

print("\n Random Forest Model:")
print(f"Mean Absolute Error (MAE): {mae_rf:.2f}")