from src.db.query import get_nodes
import pandas as pd

df = pd.read_parquet('output/AUDCAD/extrac_os/Ext-011616_AUDCAD_20200101_20230101_timeframeH1.parquet')
df.head(1).to_csv('esta.csv', index=False)

print(df)