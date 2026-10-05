import time


class TrainingTimer:
    """
    Chronomètre la durée d'un entraînement.
    """

    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self):
        """
        Démarre le chronomètre.
        """
        self.start_time = time.perf_counter()
        self.end_time = None

    def stop(self):
        """
        Arrête le chronomètre et retourne la durée en secondes.
        """
        if self.start_time is None:
            raise RuntimeError(
                "Le chronomètre n'a pas été démarré."
            )

        self.end_time = time.perf_counter()

        return self.end_time - self.start_time

    @staticmethod
    def format_duration(duration_seconds):
        """
        Convertit une durée en format lisible.
        """

        hours = int(duration_seconds // 3600)

        minutes = int(
            (duration_seconds % 3600) // 60
        )

        seconds = (
            duration_seconds % 60
        )

        return (
            f"{hours:02d}h "
            f"{minutes:02d}min "
            f"{seconds:05.2f}s"
        )