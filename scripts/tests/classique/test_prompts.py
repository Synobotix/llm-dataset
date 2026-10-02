from llm.distillation.prompts import build_distillation_prompt


def main():
    text = """
    Python est un langage de programmation interprété.
    Il est utilisé dans le développement web, la science des données,
    l'intelligence artificielle et l'automatisation.
    """

    prompt = build_distillation_prompt(text)

    print("\n===== PROMPT GÉNÉRÉ =====\n")
    print(prompt)


if __name__ == "__main__":
    main()