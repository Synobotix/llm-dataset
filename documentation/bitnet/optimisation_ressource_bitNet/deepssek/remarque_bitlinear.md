🔴 1. Problème majeur : la quantification ternaire n'est pas correcte
Dans ton quantize_weights, tu fais :

python
quantized_weight = torch.where(
    normalized_weight > 0.5, 1,
    torch.where(normalized_weight < -0.5, -1, 0)
)
Cette approche avec un seuil fixe à 0.5 n'est pas celle de BitNet. Le vrai BitNet b1.58 utilise un rounding vers l'entier le plus proche, pas un seuil arbitraire. La formule officielle est :

W
~
=
RoundClip
(
W
γ
+
ϵ
,
 
−
1
,
 
1
)
W
~
 =RoundClip( 
γ+ϵ
W
​
 , −1, 1)
avec :

γ
=
1
n
m
∑
i
j
∣
W
i
j
∣
γ= 
nm
1
​
 ∑ 
ij
​
 ∣W 
ij
​
 ∣ (moyenne des valeurs absolues)

RoundClip
(
x
,
−
1
,
1
)
=
max
⁡
(
−
1
,
min
⁡
(
1
,
round
(
x
)
)
)
RoundClip(x,−1,1)=max(−1,min(1,round(x)))

Traduction en PyTorch :

python
@staticmethod
def quantize_weights(weight: torch.Tensor):
    scale = weight.abs().mean().clamp(min=1e-5)
    # Équivalent de RoundClip
    w_scaled = weight / scale
    w_rounded = torch.round(w_scaled)
    w_clipped = torch.clamp(w_rounded, -1, 1)
    return w_clipped * scale
La différence : avec round, une valeur de 0.6 devient 1, mais 0.4 devient 0. Avec ton seuil à 0.5, 0.4 devient aussi 0 — donc à première vue similaire. Mais avec round, 1.4 devient 1 et 1.6 devient 2 → 1 après clamp. Là où ton code donne 1 aussi, mais pour une autre raison. Le vrai problème est que ton seuil 0.5 n'a aucune justification théorique et ne correspond pas à la distribution réelle des poids après normalisation.

🟠 2. La quantification des activations n'est pas conforme
BitNet utilise une quantification absmax sur 8 bits pour les activations, et non une quantification par ligne comme tu le fais :

python
scale = x.abs().amax(dim=-1, keepdim=True)   # ← par ligne
La formule officielle BitNet (pour les activations) est :

x
~
=
Clip
(
x
⋅
Q
b
β
,
 
−
Q
b
+
1
,
 
Q
b
−
1
)
⋅
β
Q
b
x
~
 =Clip(x⋅ 
β
Q 
b
​
 
​
 , −Q 
b
​
 +1, Q 
b
​
 −1)⋅ 
Q 
b
​
 
β
​
 
avec 
β
=
∥
x
∥
∞
β=∥x∥ 
∞
​
  (max absolu global, pas par ligne), et 
Q
b
=
2
b
−
1
=
128
Q 
b
​
 =2 
b−1
 =128 pour 8 bits.

En pratique, dans les implémentations récentes (BitNet b1.58 2B4T), la quantification des activations utilise LayerNorm avant quantification et un absmax global par tenseur, pas par ligne. Ta version par ligne peut créer des instabilités.

python
@staticmethod
def quantize_activations(x: torch.Tensor, bits: int = 8):
    Qb = 2 ** (bits - 1)          # 128 pour 8 bits
    scale = x.abs().max().clamp(min=1e-5)  # absmax global
    x_scaled = x * (Qb / scale)
    x_clipped = torch.clamp(x_scaled, -Qb + 1, Qb - 1)
    x_quant = torch.round(x_clipped)
    return (x_quant / Qb) * scale
🟠 3. Ton STE est appliqué deux fois, mais mal placé
Tu appliques le STE après la quantification, ce qui est correct en soi, mais :

python
x_quantized = self.straight_through_estimator(x, x_quantized)
Cette ligne fait x + (x_q - x).detach(). En forward, ça donne x_q ✅. En backward, le gradient de x_q par rapport à x est 1 (car x_q - x est détaché). Mais attention : cela signifie que le gradient remonte aussi à travers x sans être atténué, ce qui est l'effet recherché du STE — donc c'est OK.

En revanche, tu appliques le STE aux poids à chaque forward, ce qui est correct, mais tu devrais t'assurer que le gradient de scale ne disparaît pas. Dans ta version, scale = weight.abs().mean() est calculé à partir de weight, et comme tu fais quantized * scale puis STE, le gradient de la loss par rapport à scale est bien propagé. ✅

Optimisation : utilise directement .detach() sur la partie quantification pour éviter les calculs inutiles :

python
w_q = self.quantize_weights(self.weight)
w_q = self.weight + (w_q - self.weight).detach()
C'est exactement ce que tu fais, mais tu peux le faire en une ligne dans forward.

🟡 4. Optimisations de performance
a) Éviter les allocations inutiles
Ton quantize_weights crée torch.ones_like, torch.zeros_like, torch.full_like via torch.where imbriqués. Avec round + clamp, tu n'as plus besoin de tout ça.

b) Utiliser torch.compile (PyTorch 2.x)
Si tu es sur PyTorch ≥ 2.0, tu peux wrapper ton forward :

python
self.forward = torch.compile(self.forward, mode="reduce-overhead")
Cela fusionne les opérations et réduit le surcoût mémoire.

c) Fusionner quantification des poids dans un buffer
Si tu ne fais pas de fine-tuning des poids (inférence seule), tu peux pré-calculer les poids ternaires une fois et les stocker :

python
def prepare_for_inference(self):
    with torch.no_grad():
        self.weight_ternary = self.quantize_weights(self.weight).detach()
    # Puis en forward, utilise self.weight_ternary directement
d) Kernel BitLinear optimisé
Pour vraiment exploiter BitNet, il faut un kernel C++/CUDA qui fait l'addition au lieu de la multiplication. C'est ce que fait bitnet.cpp de Microsoft. En PyTorch pur, tu ne pourras pas atteindre ces performances — mais tu peux tester bitblas ou T-MAC qui fournissent des kernels optimisés pour les poids ternaires.

🟢 5. Structure : séparer entraînement et inférence
Ton code mélange les deux. Une bonne pratique est de définir un mode :

python
def forward(self, x):
    if self.training:
        # STE + quantification dynamique
        w_q = self._quantize_with_ste(self.weight)
    else:
        # Poids déjà ternaires, pas de recalcul
        w_q = self.weight_ternary
    x_q = self._quantize_activations(x)
    return F.linear(x_q, w_q, self.bias)
📋 Version optimisée condensée
Voici une version corrigée des parties clés :

python
@staticmethod
def quantize_weights(weight: torch.Tensor):
    scale = weight.abs().mean().clamp(min=1e-5)
    w_scaled = weight / scale
    w_rounded = torch.round(w_scaled).clamp(-1, 1)
    return w_rounded * scale

@staticmethod
def quantize_activations(x: torch.Tensor, bits: int = 8):
    Qb = 2 ** (bits - 1)
    scale = x.abs().max().clamp(min=1e-5)
    x_scaled = x * (Qb / scale)
    x_clipped = torch.clamp(x_scaled, -Qb + 1, Qb - 1)
    return (torch.round(x_clipped) / Qb) * scale

def forward(self, x):
    w_q = self.quantize_weights(self.weight)
    w_q = self.weight + (w_q - self.weight).detach()  # STE

    x_q = self.quantize_activations(x)
    x_q = x + (x_q - x).detach()  # STE

    return F.linear(x_q, w_q, self.bias)
⚠️ Point critique sur le vrai BitNet
Attention : dans le vrai BitNet b1.58, les activations sont quantifiées en 8 bits, mais après une LayerNorm, et surtout il n'y a pas de biais dans les couches linéaires (bias=False partout, car LayerNorm compense). Tu as mis bias optionnel, c'est bien, mais pour un vrai BitNet, mets bias=False par défaut.

De plus, BitNet remplace toutes les couches linéaires, y compris dans l'attention (Q, K, V, O) et le MLP, et utilise RMSNorm (pas LayerNorm classique).

Veux-tu que je te montre comment intégrer cette BitLinear dans un bloc Transformer complet (attention + MLP) pour avoir un vrai modèle BitNet fonctionnel ?