import pandas as pd


read_csv=pd.read_csv("melb_data.csv")
clear_scv=read_csv.dropna(axis=0)

data_y=clear_scv["price"]
data_x=clear_scv[["Rooms", "Bathroom", "Landsize", "Latitude", "Longitude"]]