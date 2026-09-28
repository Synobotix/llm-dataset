from llm.distillation.teachers import TEACHERS


def main():
    print("Nombre de teachers :", len(TEACHERS))

    for index, teacher in enumerate(TEACHERS, start=1):
        print(f"{index:02d} → {teacher}")


if __name__ == "__main__":
    main()