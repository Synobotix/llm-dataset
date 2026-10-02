======================================================================
ENTRAÎNEMENT BITNET
======================================================================

Device : cpu

Chargement du tokenizer...
Vocabulaire réel : 5885

Création des DataLoaders...
Train batches      : 122
Validation batches : 7
Batch size         : 2
Sequence length    : 128

Création du BitTransformer...
Création du LM Head...

Paramètres Transformer : 2,728,448
Paramètres LM Head    : 1,506,560
Paramètres totaux     : 4,235,008

Chargement du checkpoint précédent...
Checkpoint chargé : checkpoint/bitnet/bitnet_25docs.pt
Époque précédente : 3
Ancienne train loss : 5.263747
Ancienne validation loss : 7.286995

Configuration :
  Documents           = 26
  Vocabulaire         = 5885
  D model             = 256
  Num heads           = 4
  Hidden dim          = 680
  Num blocks          = 2
  Max sequence        = 128
  Batch size          = 2
  Nouvelles époques   = 20
  Learning rate       = 0.001
  Weight decay        = 0.01
  Gradient clip       = 1.0
  Ancien checkpoint   = checkpoint/bitnet/bitnet_25docs.pt
  Nouveau checkpoint  = checkpoint/bitnet/bitnet_25docs_continued.pt


  Train loss       : 0.561962
Validation loss  : 10.964425
Train PPL        : 1.7541
Validation PPL   : 57781.5745
Gradient norm    : 3.448278

Nouveau checkpoint sauvegardé : checkpoint/bitnet/bitnet_25docs_continued.pt

======================================================================
POURSUITE DE L'ENTRAÎNEMENT TERMINÉE
======================================================================

Dernière époque : 23
Ancien checkpoint : checkpoint/bitnet/bitnet_25docs.pt
Nouveau checkpoint : checkpoint/bitnet/bitnet_25docs_continued.pt
