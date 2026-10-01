import torch


class TextGenerator:
    def __init__(
        self,
        model,
        tokenizer,
        device="cpu",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
    ) -> str:

        # ==================================================
        # 1. TOKENISATION DU PROMPT
        # ==================================================

        encoded = self.tokenizer.encode(prompt)

        input_ids = torch.tensor(
            [encoded.ids],
            dtype=torch.long,
            device=self.device,
        )

        print("\n" + "=" * 60)
        print("DÉBUT DE LA GÉNÉRATION")
        print("=" * 60)

        print("Prompt :", repr(prompt))
        print("Token IDs :", encoded.ids)
        print("Nombre de tokens du prompt :", len(encoded.ids))

        # ==================================================
        # 2. GÉNÉRATION
        # ==================================================

        for step in range(max_new_tokens):

            # --------------------------------------------------
            # Limitation au contexte maximal du Transformer
            # --------------------------------------------------

            input_context = input_ids[:, -128:]

            print("\n" + "-" * 60)
            print(f"ÉTAPE {step + 1}")
            print("-" * 60)

            print(
                "Nombre de tokens dans input_ids :",
                input_ids.shape[1],
            )

            print(
                "Nombre de tokens dans input_context :",
                input_context.shape[1],
            )

            # --------------------------------------------------
            # Passage dans le Transformer
            # --------------------------------------------------

            logits = self.model(input_context)

            print(
                "Shape logits :",
                logits.shape,
            )

            # --------------------------------------------------
            # Logits correspondant au dernier token
            # --------------------------------------------------

            next_token_logits = logits[:, -1, :]

            print(
                "Shape derniers logits :",
                next_token_logits.shape,
            )

            print(
                "Nombre de tokens du vocabulaire :",
                next_token_logits.shape[-1],
            )

            # ==================================================
            # 3. DEBUG DES LOGITS
            # ==================================================

            if step == 0:

                print("\n" + "=" * 60)
                print("DEBUG DES LOGITS - PREMIÈRE ÉTAPE")
                print("=" * 60)

                print(
                    "Min   :",
                    next_token_logits.min().item(),
                )

                print(
                    "Max   :",
                    next_token_logits.max().item(),
                )

                print(
                    "Mean  :",
                    next_token_logits.mean().item(),
                )

                print(
                    "Std   :",
                    next_token_logits.std().item(),
                )

                print(
                    "NaN   :",
                    torch.isnan(
                        next_token_logits
                    ).any().item(),
                )

                print(
                    "Inf   :",
                    torch.isinf(
                        next_token_logits
                    ).any().item(),
                )

                # --------------------------------------------------
                # Top 10 logits
                # --------------------------------------------------

                top_values, top_ids = torch.topk(
                    next_token_logits,
                    k=10,
                    dim=-1,
                )

                print("\nTop 10 tokens :")

                for rank, (value, token_id) in enumerate(
                    zip(
                        top_values[0],
                        top_ids[0],
                    ),
                    start=1,
                ):

                    token_id = token_id.item()
                    value = value.item()

                    # Décodage du token
                    try:
                        token_text = self.tokenizer.decode(
                            [token_id]
                        )
                    except Exception:
                        token_text = "<erreur>"

                    print(
                        f"{rank:2d}. "
                        f"ID={token_id:3d} | "
                        f"logit={value:10.6f} | "
                        f"token={repr(token_text)}"
                    )

            # ==================================================
            # 4. GREEDY DECODING
            # ==================================================

            next_token_id = torch.argmax(
                next_token_logits,
                dim=-1,
                keepdim=True,
            )

            token_id = next_token_id.item()

            # --------------------------------------------------
            # Décodage du token choisi
            # --------------------------------------------------

            try:
                token_text = self.tokenizer.decode(
                    [token_id]
                )
            except Exception:
                token_text = "<erreur>"

            print("\nTOKEN CHOISI")
            print(
                "Token ID :",
                token_id,
            )

            print(
                "Token texte :",
                repr(token_text),
            )

            # --------------------------------------------------
            # Logit du token choisi
            # --------------------------------------------------

            selected_logit = next_token_logits[
                0,
                token_id,
            ].item()

            print(
                "Logit du token choisi :",
                selected_logit,
            )

            # ==================================================
            # 5. AJOUT DU TOKEN
            # ==================================================

            input_ids = torch.cat(
                [
                    input_ids,
                    next_token_id,
                ],
                dim=1,
            )

        # ==================================================
        # 6. DÉCODAGE FINAL
        # ==================================================

        generated_ids = input_ids[0].tolist()

        generated_text = self.tokenizer.decode(
            generated_ids
        )

        print("\n" + "=" * 60)
        print("FIN DE LA GÉNÉRATION")
        print("=" * 60)

        print(
            "Nombre total de tokens :",
            len(generated_ids),
        )

        print(
            "IDs générés :",
            generated_ids,
        )

        print(
            "Texte généré :",
            generated_text,
        )

        return generated_text