import json

# Read the data from captions.json
with open('datasets/captions.json', 'r') as file:
    captions_data = json.load(file)

# Create a .txt file for each caption
for item in captions_data:
    nsd_id = int(item['nsdId']) + 1  # Increment nsdId by 1
    filename = f'datasets/coco_labels/{nsd_id}.txt'

    with open(filename, 'w') as txt_file:
        txt_file.write(item['caption'])

print("Text files created successfully.")
