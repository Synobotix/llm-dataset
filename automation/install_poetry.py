
import subprocess


def install_poetry() -> None:
    """
    Installe Poetry dans l'environnement Colab
    et vérifie son installation.
    """

    print("=" * 60)
    print("INSTALLATION DE POETRY")
    print("=" * 60)

    subprocess.run(
        [
            "pip",
            "install",
            "poetry",
        ],
        check=True,
    )

    subprocess.run(
        [
            "poetry",
            "--version",
        ],
        check=True,
    )

    print("Poetry : OK")
