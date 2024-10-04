import os
import csv
import shutil

# Step 1: Define folder name
folder_name = 'csv_folder'

# Step 2: Check if folder exists and delete if it does
if os.path.exists(folder_name):
    shutil.rmtree(folder_name)  # Delete the folder and its contents
    print(f"Folder '{folder_name}' deleted.")

# Step 3: Create a new folder
os.makedirs(folder_name)
print(f"Folder '{folder_name}' created.")

# Step 4: Define CSV files and headers
csv_files = [
    ('file1.csv', ['Name', 'Age', 'City']),
    ('file2.csv', ['Product', 'Price', 'Quantity']),
    ('file3.csv', ['Date', 'Temperature', 'Humidity']),
    ('file4.csv', ['ID', 'Score', 'Grade'])
]

# Step 5: Create CSV files with headers
for filename, header in csv_files:
    filepath = os.path.join(folder_name, filename)
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)  # Write the header row
        print(f"CSV file '{filename}' created with header: {header}")

print("All CSV files have been created successfully.")
