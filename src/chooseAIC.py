from utils import printAIC
from pandas import read_csv
#Load data set
series_influ_A_df = read_csv('../data/vietnam_flu_A.csv')
series_influ_A_df = series_influ_A_df.dropna()
# Create Training and Test
train = series_influ_A_df["Influenza A - All types of surveillance"][:735]
test = series_influ_A_df["Influenza A - All types of surveillance"][735:]

printAIC(1,6, train)
