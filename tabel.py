import pandas as pd

df = pd.read_csv("train.csv")

tabel = df["Emotion"].value_counts().reset_index()
tabel.columns = ["Emotie", "Numar exemple"]

print(tabel)