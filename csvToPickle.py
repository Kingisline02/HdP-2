import pandas as pd
import pickle

# Load data from CSV file
csv_filename = "data.csv"
df = pd.read_csv(csv_filename)

# Save the dataframe to a pickle file
pkl_filename = "data.pkl"
df.to_pickle(pkl_filename)

print(f"Data has been written to {pkl_filename}")
