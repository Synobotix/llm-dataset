from datasets import load_dataset


def main():
    dataset = load_dataset(
        "allenai/c4",
        "fr",
        split="train",
        streaming=True,
    )

    print("Dataset C4 connecté.")
    print(dataset)

    for i, example in enumerate(dataset):
        print("\n" + "=" * 80)
        print(f"DOCUMENT {i + 1}")
        print("=" * 80)
        print(example["text"][:1000])

        if i >= 2:
            break


if __name__ == "__main__":
    main()