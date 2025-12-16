import json
with open('data/raw/shipsnet.json', 'r') as f:
    data = json.load(f)

print("Sample locations:")
for i in range(5):
    print(f"Image {i}: {data['locations'][i]}")