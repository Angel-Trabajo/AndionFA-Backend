import pandas as pd

df = pd.read_csv("output/AUDCHF/data_for_neuronal/data/data_Asia_DOWN.csv")

print("Tipos de datos:")
print(df.dtypes)

print("\nPrimeras filas (RAW):")
print(df[["input1", "input2", "hour"]].head(10))

print("\nChequeo de longitudes:")

for col, expected_len in [("input1", 8), ("input2", 8), ("hour", 5)]:
    lengths = df[col].astype(str).apply(len)
    print(f"\n{col}:")
    print("Min len:", lengths.min())
    print("Max len:", lengths.max())
    print("Ejemplos únicos cortos:")
    print(df[col].astype(str).unique()[:10])
    
def has_leading_zero(x):
    s = str(x)
    return s.startswith("0")

for col in ["input1", "input2", "hour"]:
    count = df[col].astype(str).apply(has_leading_zero).sum()
    print(f"{col} con ceros a la izquierda:", count)
    
df_str = pd.read_csv("output/AUDCHF/data_for_neuronal/data/data_Asia_DOWN.csv", dtype=str)

print("\nLeído como string:")
print(df_str[["input1", "input2", "hour"]].head(10))

print("\nLongitudes reales:")
for col in ["input1", "input2", "hour"]:
    lengths = df_str[col].apply(len)
    print(f"{col}: min={lengths.min()}, max={lengths.max()}")