import pandas as pd
df = pd.read_csv('./data/influenza_month.csv')
# df = df[["Country", "All Influenza A", "All Influenza B"]]
vietname_df = df[df["Country"] == "Vietnam"]
vietname_df = vietname_df[["Month", "Influenza B - All types of surveillance"]]
# vietname_df.rename(columns={"Influenza A - All types of surveillance": "Influ_A", "Influenza B - All types of surveillance": "Influ_B"})
vietname_df.to_csv('./data/vietnam_flu_B_month.csv', index=0)