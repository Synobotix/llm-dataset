
Vocabulaire tokenizer : 994
vocab_size: 994
d_model: 256
num_heads: 4
hidden_dim: 680
num_blocks: 2
max_sequence_length: 128
epochs: 3
learning_rate: 0.001
weight_decay: 0.01
gradient_clip: 1.0

Résultat:

Epoch 1/3
  Train Loss            : 6.895225
  Validation Loss       : 6.775599
  Train PPL             : 987.547811
  Validation PPL        : 876.203830
  Train PLL/token       : -6.895225
  Validation PLL/token  : -6.775599
  Gradient norm         : 1.233607
  Learning rate         : 0.00100000

Epoch 2/3
  Train Loss            : 6.136259
  Validation Loss       : 6.245333
  Train PPL             : 462.320574
  Validation PPL        : 515.600746
  Train PLL/token       : -6.136259
  Validation PLL/token  : -6.245333
  Gradient norm         : 1.285185
  Learning rate         : 0.00100000

Epoch 3/3
  Train Loss            : 5.576481
  Validation Loss       : 6.099547
  Train PPL             : 264.140513
  Validation PPL        : 445.655638
  Train PLL/token       : -5.576481
  Validation PLL/token  : -6.099547
  Gradient norm         : 1.282125
  Learning rate         : 0.00100000

POIDS BITLINEAR FINAUX
----------------------------------------------------------------------
-1                      : 545,716 (37.00%)
 0                      : 384,677 (26.08%)
+1                      : 544,679 (36.93%)
Total                   : 1,475,072
Moyenne |poids latent| : 0.029854

CHECKPOINT
----------------------------------------------------------------------
Fichier                 : checkpoint/bitnet/bitnet_5docs.pt

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


B-
======================================================================
ENTRAÎNEMENT BITNET TERMINÉ
======================================================================

CONFIGURATION
----------------------------------------------------------------------
Vocabulary              : 994
D_model                 : 256
Nombre de heads         : 4
Head dimension          : 64
Hidden dimension        : 680
Nombre de blocks        : 2
Sequence length         : 128
Epochs                  : 20
Learning rate           : 0.001
Weight decay            : 0.01
Gradient clipping       : 1.0
Optimizer               : AdamW
Loss                    : CrossEntropyLoss

PARAMÈTRES
----------------------------------------------------------------------
Paramètres totaux       : 1,730,816
Paramètres entraînables : 1,730,816

DATASET
----------------------------------------------------------------------
Train sequences         : 21
Validation sequences    : 13
Batch size              : 2

RÉSULTATS DES NOUVELLES EPOCHS
----------------------------------------------------------------------

Epoch 4/20
  Train Loss            : 5.096227
  Validation Loss       : 5.995258
  Train PPL             : 163.404228
  Validation PPL        : 401.520322
  Train log-likelihood/token : -5.096227
  Validation log-likelihood/token : -5.995258
  Gradient norm         : 1.242953
  Learning rate         : 0.00100000

Epoch 5/20
  Train Loss            : 4.664534
  Validation Loss       : 5.980238
  Train PPL             : 106.116121
  Validation PPL        : 395.534450
  Train log-likelihood/token : -4.664534
  Validation log-likelihood/token : -5.980238
  Gradient norm         : 1.254684
  Learning rate         : 0.00100000

Epoch 6/20
  Train Loss            : 4.230992
  Validation Loss       : 5.915922
  Train PPL             : 68.785467
  Validation PPL        : 370.896254
  Train log-likelihood/token : -4.230992
  Validation log-likelihood/token : -5.915922
  Gradient norm         : 1.221244
  Learning rate         : 0.00100000

Epoch 7/20
  Train Loss            : 3.797655
  Validation Loss       : 5.916434
  Train PPL             : 44.596497
  Validation PPL        : 371.086166
  Train log-likelihood/token : -3.797655
  Validation log-likelihood/token : -5.916434
  Gradient norm         : 1.217974
  Learning rate         : 0.00100000

Epoch 8/20
  Train Loss            : 3.392064
  Validation Loss       : 5.937403
  Train PPL             : 29.727245
  Validation PPL        : 378.949331
  Train log-likelihood/token : -3.392064
  Validation log-likelihood/token : -5.937403
  Gradient norm         : 1.202202
  Learning rate         : 0.00100000

Epoch 9/20
  Train Loss            : 3.033769
  Validation Loss       : 5.987988
  Train PPL             : 20.775387
  Validation PPL        : 398.611794
  Train log-likelihood/token : -3.033769
  Validation log-likelihood/token : -5.987988
  Gradient norm         : 1.221218
  Learning rate         : 0.00100000

Epoch 10/20
  Train Loss            : 2.711326
  Validation Loss       : 6.077364
  Train PPL             : 15.049215
  Validation PPL        : 435.878657
  Train log-likelihood/token : -2.711326
  Validation log-likelihood/token : -6.077364
  Gradient norm         : 1.246799
  Learning rate         : 0.00100000

Epoch 11/20
  Train Loss            : 2.387798
  Validation Loss       : 6.075209
  Train PPL             : 10.889491
  Validation PPL        : 434.940377
  Train log-likelihood/token : -2.387798
  Validation log-likelihood/token : -6.075209
  Gradient norm         : 1.242358
  Learning rate         : 0.00100000

Epoch 12/20
  Train Loss            : 2.118512
  Validation Loss       : 6.203563
  Train PPL             : 8.318748
  Validation PPL        : 494.507925
  Train log-likelihood/token : -2.118512
  Validation log-likelihood/token : -6.203563
  Gradient norm         : 1.324667
  Learning rate         : 0.00100000

Epoch 13/20
  Train Loss            : 1.885807
  Validation Loss       : 6.274491
  Train PPL             : 6.591669
  Validation PPL        : 530.855882
  Train log-likelihood/token : -1.885807
  Validation log-likelihood/token : -6.274491
  Gradient norm         : 1.288995
  Learning rate         : 0.00100000

Epoch 14/20
  Train Loss            : 1.658808
  Validation Loss       : 6.383308
  Train PPL             : 5.253047
  Validation PPL        : 591.882402
  Train log-likelihood/token : -1.658808
  Validation log-likelihood/token : -6.383308
  Gradient norm         : 1.258189
  Learning rate         : 0.00100000

Epoch 15/20
  Train Loss            : 1.433601
  Validation Loss       : 6.608606
  Train PPL             : 4.193773
  Validation PPL        : 741.448942
  Train log-likelihood/token : -1.433601
  Validation log-likelihood/token : -6.608606
  Gradient norm         : 1.272486
  Learning rate         : 0.00100000

Epoch 16/20
  Train Loss            : 1.262423
  Validation Loss       : 6.623899
  Train PPL             : 3.533975
  Validation PPL        : 752.874580
  Train log-likelihood/token : -1.262423
  Validation log-likelihood/token : -6.623899
  Gradient norm         : 1.224171
  Learning rate         : 0.00100000

Epoch 17/20
  Train Loss            : 1.068632
  Validation Loss       : 6.902170
  Train PPL             : 2.911394
  Validation PPL        : 994.430178
  Train log-likelihood/token : -1.068632
  Validation log-likelihood/token : -6.902170
  Gradient norm         : 1.178373
  Learning rate         : 0.00100000

Epoch 18/20
  Train Loss            : 0.917141
  Validation Loss       : 6.919050
  Train PPL             : 2.502126
  Validation PPL        : 1011.358373
  Train log-likelihood/token : -0.917141
  Validation log-likelihood/token : -6.919050
  Gradient norm         : 1.155993
  Learning rate         : 0.00100000

Epoch 19/20
  Train Loss            : 0.803486
  Validation Loss       : 6.969338
  Train PPL             : 2.233312
  Validation PPL        : 1063.518093
  Train log-likelihood/token : -0.803486
  Validation log-likelihood/token : -6.969338
  Gradient norm         : 1.139997
  Learning rate         : 0.00100000

Epoch 20/20
  Train Loss            : 0.710548
  Validation Loss       : 7.209262
  Train PPL             : 2.035105
  Validation PPL        : 1351.893562
  Train log-likelihood/token : -0.710548
  Validation log-likelihood/token : -7.209262
  Gradient norm         : 1.078056
  Learning rate         : 0.00100000

POIDS BITLINEAR FINAUX
----------------------------------------------------------------------
-1                      : 527,571 (35.77%)
 0                      : 420,803 (28.53%)
+1                      : 526,698 (35.71%)
Total                   : 1,475,072
Moyenne |poids latent| : 0.044034

CHECKPOINT
----------------------------------------------------------------------
Ancien checkpoint       : checkpoint/bitnet/bitnet_5docs.pt
Nouveau checkpoint      : checkpoint/bitnet/bitnet_5docs_20epochs.pt
Epoch finale            : 20

