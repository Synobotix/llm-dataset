
import os

import requests
from dotenv import load_dotenv


load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

OPENROUTER_URL = (
    "https://openrouter.ai/api/v1/chat/completions"
)


def generate_response(
    models: list[str],
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 512,
) -> tuple[str, str]:
    """
    Génère une réponse avec OpenRouter.

    Parameters
    ----------
    models:
        Liste des modèles utilisés pour le fallback.
        OpenRouter accepte au maximum 3 modèles
        dans cette liste.

    prompt:
        Prompt envoyé au teacher.

    temperature:
        Température de génération.

    max_tokens:
        Nombre maximum de tokens générés.

    Returns
    -------
    tuple[str, str]
        response:
            Réponse générée par le teacher.

        model_used:
            Modèle qui a effectivement généré la réponse.
    """

    # --------------------------------------------------
    # Vérification de la clé API
    # --------------------------------------------------

    if not OPENROUTER_API_KEY:
        raise ValueError(
            "La variable OPENROUTER_API_KEY "
            "n'est pas définie."
        )

    # --------------------------------------------------
    # Vérification de la liste des modèles
    # --------------------------------------------------

    if not models:
        raise ValueError(
            "La liste des modèles est vide."
        )

    if len(models) > 3:
        raise ValueError(
            "OpenRouter accepte au maximum 3 modèles "
            "dans une liste de fallback."
        )

    # --------------------------------------------------
    # Headers HTTP
    # --------------------------------------------------

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    # --------------------------------------------------
    # Payload
    # --------------------------------------------------

    payload = {
        "models": models,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    # --------------------------------------------------
    # Requête OpenRouter
    # --------------------------------------------------

    response = requests.post(
        OPENROUTER_URL,
        headers=headers,
        json=payload,
        timeout=120,
    )

    # --------------------------------------------------
    # Gestion des erreurs HTTP
    # --------------------------------------------------

    if not response.ok:

        try:
            error_data = response.json()

            raise RuntimeError(
                f"Erreur OpenRouter "
                f"{response.status_code}: "
                f"{error_data}"
            )

        except ValueError:

            raise RuntimeError(
                f"Erreur OpenRouter "
                f"{response.status_code}: "
                f"{response.text}"
            )

    # --------------------------------------------------
    # Conversion JSON
    # --------------------------------------------------

    data = response.json()

    # --------------------------------------------------
    # DEBUG TEMPORAIRE
    # --------------------------------------------------

    print("\n========== OPENROUTER DEBUG ==========")

    print(data)

    print("======================================\n")

    # --------------------------------------------------
    # Récupération de la réponse
    # --------------------------------------------------

    choices = data.get("choices", [])

    if not choices:
        raise RuntimeError(
            "OpenRouter n'a retourné aucun choix "
            "dans la réponse."
        )

    choice = choices[0]

    # Récupérer la raison de fin de génération
    finish_reason = choice.get("finish_reason")

    print(f"Finish reason : {finish_reason}")

    # Récupérer le message
    message = choice.get("message", {})

    content = message.get("content") or ""

    # Vérifier si la génération a été coupée
    if finish_reason == "length":
        raise RuntimeError(
            "La génération a été interrompue car "
            "la limite max_tokens a été atteinte."
        )

    # Vérifier qu'une réponse finale existe
    if not content.strip():
        raise RuntimeError(
            "Le teacher n'a retourné aucune réponse finale."
        )

    model_used = data.get(
        "model",
        "unknown",
    )

    return content, model_used

