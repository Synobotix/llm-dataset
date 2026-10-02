from llm.distillation.storage import save_distilled_example


OUTPUT_PATH = "data/distilled/test_distilled.jsonl"


def main():
    save_distilled_example(
        output_path=OUTPUT_PATH,
        source="Python est un langage de programmation.",
        prompt="Explique le contenu du texte.",
        response=(
            "Python est un langage de programmation interprété "
            "et polyvalent."
        ),
        teacher="MODELE_A",
    )

    save_distilled_example(
        output_path=OUTPUT_PATH,
        source="FastAPI est un framework Python.",
        prompt="Explique le contenu du texte.",
        response=(
            "FastAPI est un framework Python utilisé "
            "pour créer des API web."
        ),
        teacher="MODELE_B",
    )

    print(f"Données enregistrées dans : {OUTPUT_PATH}")


if __name__ == "__main__":
    main()