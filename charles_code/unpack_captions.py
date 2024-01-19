import json

# Read the data from captions.json
with open('datasets/captions.json', 'r') as file:
    captions_data = json.load(file)

# Create a .txt file for each caption
for i, item in enumerate(captions_data):
    filename = f'datasets/coco_labels/{i+1}.txt'

    with open(filename, 'w') as txt_file:
        txt_file.write(item['caption'])

print("Text files created successfully.")
