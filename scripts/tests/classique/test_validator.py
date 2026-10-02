from llm.distillation.validator import validate_teacher_response


def main():

    tests = [
        "",
        "   ",
        "Python est un langage.",
        (
            "Python est un langage de programmation interprété et "
            "polyvalent. Il est notamment utilisé dans le développement "
            "web, la science des données, l'intelligence artificielle "
            "et l'automatisation."
        ),
    ]

    for index, response in enumerate(tests, start=1):
        result = validate_teacher_response(response)

        print(
            f"Test {index} → "
            f"{'VALIDE' if result else 'REJETÉE'}"
        )


if __name__ == "__main__":
    main()