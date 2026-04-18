import pandas as pd

df = pd.read_csv("output/AUDCHF/data_for_neuronal/data/data_Asia_UP.csv")

print(df.dtypes)
print("\nValores únicos input1 (primeros 10):")
print(df["input1"].unique()[:10])

print("\nMáximo valor input1:")
print(df["input1"].max())

def is_binary_like(n):
    return set(str(n)).issubset({"0", "1"})

df["is_binary"] = df["input1"].apply(is_binary_like)

print(df["is_binary"].value_counts())

df["as_string"] = df["input1"].astype(str)
df["as_bits"] = df["as_string"].apply(lambda x: [int(b) for b in x])

print(df[["input1", "as_bits"]].head())

print("input1 únicos:", df["input1"].nunique())
print("input2 únicos:", df["input2"].nunique())