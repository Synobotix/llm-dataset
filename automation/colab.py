
from automation.clone_repository import clone_repository
from automation.install_poetry import install_poetry
from automation.configure_cuda import configure_cuda
from automation.regenerate_lock import regenerate_lock
from automation.install_dependencies import install_dependencies
from automation.verify_environment import verify_environment
from automation.verify_dataset import verify_dataset
from automation.train_student_v1 import train_student_v1
from automation.verify_checkpoint import verify_checkpoint
from automation.inference_student_v1 import inference_student_v1

def main() -> None:
    """
    Orchestre l'ensemble du workflow Colab
    depuis le clonage jusqu'à la vérification
    du checkpoint Student v1.
    """

    print()
    print("=" * 60)
    print("AUTOMATISATION COLAB - STUDENT V1")
    print("=" * 60)

    # 1. Clonage
    clone_repository()

    # 2. Installation Poetry
    install_poetry()

    # 3. Configuration PyTorch CUDA
    configure_cuda()

    # 4. Régénération du lock
    regenerate_lock()

    # 5. Installation des dépendances
    install_dependencies()

    # 6. Vérification environnement
    verify_environment()

    # 7. Vérification dataset
    verify_dataset()

    # 8. Entraînement Student v1
    train_student_v1()

    # 9. Vérification checkpoint
    verify_checkpoint()
    inference_student_v1()

    print()
    print("=" * 60)
    print("PIPELINE COLAB STUDENT V1 : TERMINE")
    print("=" * 60)


if __name__ == "__main__":
    main()
