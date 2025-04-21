import pandas as pd

di_err_path = "Lookup_tables/diabetes_err.csv"
di_pen_path = "Lookup_tables/diabetes_pen.csv"
ca_err_path = "Lookup_tables/cancer_err.csv"
ca_pen_path = "Lookup_tables/cancer_pen.csv"
he_err_path = "Lookup_tables/heart_err.csv"
he_pen_path = "Lookup_tables/heart_pen.csv"
wi_err_path = "Lookup_tables/wine_err.csv"
wi_pen_path = "Lookup_tables/wine_pen.csv"

err_df = pd.read_csv(he_err_path, header=None, names=['key', 'err'], dtype={"key": str})
# err_df.columns = ['key', 'err']
err_df['key'][3]
pen_df = pd.read_csv(he_pen_path, header=None)
pen_df.columns = ['key', 'pen']

err_df['pen'] = pen_df['pen']
err_df['f32'] = (1/32) * pen_df['pen'] + err_df['err']
err_df['f16'] = (1/16) * pen_df['pen'] + err_df['err']
err_df['f8'] = (1/8) * pen_df['pen'] + err_df['err']
print(err_df.head())
err_df.to_csv('Lookup_tables/heart_fitness_lut.csv', index=False)
