import csv
import numpy as np

# Create a dataset
data = np.array([
    [45, 165, 70, 120, 1, 1],  # Age, Height, Weight, Blood Pressure, Smoking Status, Target (0: Low, 1: Moderate, 2: High)
    [50, 170, 80, 130, 0, 2],
    [30, 180, 75, 110, 1, 0],
    [60, 160, 90, 140, 1, 2],
    [35, 175, 65, 125, 0, 1],
    # Add more data as needed
])

# Save to CSV file
filename = "data.csv"
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Age", "Height", "Weight", "Blood Pressure", "Smoking Status", "Target"])
    for row in data:
        writer.writerow(row)

print(f"Data has been written to {filename}")
