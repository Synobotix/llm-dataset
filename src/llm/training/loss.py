import torch
import torch.nn as nn


class CausalLanguageModelingLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.loss_function = nn.CrossEntropyLoss()

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:

        # --------------------------------------------------
        # Dimensions
        # --------------------------------------------------

        batch_size = logits.size(0)
        sequence_length = logits.size(1)
        vocab_size = logits.size(2)

        # --------------------------------------------------
        # [batch, sequence, vocab_size]
        #
        # ->
        #
        # [batch * sequence, vocab_size]
        # --------------------------------------------------

        logits = logits.reshape(
            batch_size * sequence_length,
            vocab_size,
        )

        # --------------------------------------------------
        # [batch, sequence]
        #
        # ->
        #
        # [batch * sequence]
        # --------------------------------------------------

        labels = labels.reshape(
            batch_size * sequence_length,
        )

        # --------------------------------------------------
        # Cross Entropy
        # --------------------------------------------------

        loss = self.loss_function(
            logits,
            labels,
        )

        return loss