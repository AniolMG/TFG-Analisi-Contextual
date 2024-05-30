import csv
import os

# Directory containing the text files
web_links_dir = "WebLinks"

# CSV file to write to
csv_file = 'url_labels.csv'

# Read the existing rows from the CSV file into a list
existing_rows = []
if os.path.exists(csv_file):
    with open(csv_file, 'r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        existing_rows = list(reader)[1:]  # Exclude the header row

# List to store new rows
new_rows = []

# Iterate over the text files in the directory
for txt_file in os.listdir(web_links_dir):
    if txt_file.endswith(".txt"):
        # Use the filename (without extension) as the main label
        main_label = os.path.splitext(txt_file)[0]

        # Open the text file and read the URLs
        with open(os.path.join(web_links_dir, txt_file), 'r') as file:
            urls = file.read().splitlines()

            # Store each URL in the new_rows list with the main label and an empty secondary label
            for url in urls:
                if url not in [row[0] for row in existing_rows]:
                    new_row = [url, main_label, '']
                    new_rows.append(new_row)
                    existing_rows.append(new_row)

# Sort the all_rows list by the main_label, then secondary_label, and finally url
all_rows = sorted(existing_rows, key=lambda row: (row[1], row[2], row[0]))

# Open the CSV file in write mode
with open(csv_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    # Write the header row
    writer.writerow(['url', 'main_label', 'secondary_label'])
    # Write the all rows to the CSV file
    for row in all_rows:
        writer.writerow(row)
