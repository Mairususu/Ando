import requests
import json
import time

# -----------------------------
# Step 1: Fetch all characters
# -----------------------------
print("Fetching all character IDs...")
url = "https://api.api-onepiece.com/v2/characters/en"
try:
    all_characters = requests.get(url).json()
except Exception as e:
    print(f"Error fetching characters: {e}")
    all_characters = []

print(f"Total characters fetched: {len(all_characters)}")

# -----------------------------
# Step 2: Fetch full details for each character
# -----------------------------
output = []
print("Fetching character details...")

for i, char in enumerate(all_characters, start=1):
    char_id = char["id"]
    try:
        response = requests.get(f"https://api.api-onepiece.com/v2/characters/en/{char_id}")
        if response.status_code != 200 or not response.text:
            print(f"Skipping character {char_id}, status: {response.status_code}")
            continue
        details = response.json()
    except Exception as e:
        print(f"Error fetching details for {char_id}: {e}")
        continue

    bounty = details.get("bounty")
    if bounty and bounty != 0:  # keep only characters with bounty != 0
        output.append({
            "Name": details.get("name"),
            "Bounty": bounty,
            "Age": details.get("age"),
            "Crew": details.get("crew").get("name") if details.get("crew") else None,
            "Fruit": details.get("fruit").get("name") if details.get("fruit") else None,
        })

    if i % 50 == 0:
        print(f"Processed {i}/{len(all_characters)} characters...")
    time.sleep(0.05)  # slightly longer delay to avoid rate limiting

print(f"Characters with bounty != 0: {len(output)}")

# -----------------------------
# Step 3: Save to JSON
# -----------------------------
with open("onepiece_characters.json", "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=4)

print("JSON created successfully as 'onepiece_characters.json'!")
