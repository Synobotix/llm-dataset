import json

path = "data/tokenized/train.jsonl"
target = [750, 978, 261, 977]

found = False

with open(path, "r", encoding="utf-8") as f:
    for sequence_number, line in enumerate(f, 1):
        data = json.loads(line)

        input_ids = data["input_ids"]
        labels = data["labels"]

        for position in range(len(input_ids) - len(target) + 1):
            if input_ids[position:position + len(target)] == target:

                print("=" * 70)
                print("SEQUENCE TROUVEE")
                print("=" * 70)

                print("Numéro de séquence :", sequence_number)
                print("Position           :", position)

                print()
                print("Input IDs :")
                print(input_ids[position:position + len(target)])

                print()
                print("Labels :")
                print(labels[position:position + len(target)])

                found = True

print()
print("=" * 70)
print("RESULTAT FINAL")
print("=" * 70)
print("Séquence trouvée :", found)
