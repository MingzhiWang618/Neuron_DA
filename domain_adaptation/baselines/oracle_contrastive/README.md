# CREMI-A supervised contrastive oracle

This is the fully supervised target-domain upper bound for AC4 to CREMI-A
adaptation. It trains from scratch on the labeled CREMI-A train split
(`100 x 1250 x 1250`) and selects checkpoints only on the disjoint labeled
CREMI-A validation split (`25 x 1250 x 1250`).

The model and objective match the archived source-only contrastive baseline:

```text
MedNeXt-S, 3 nearest-neighbor DeepEM affinities (X, Y, Z)
L_total = L_affinity + 0.1 * L_instance_contrastive
```

All configurations train for 100 epochs (1000 iterations per epoch) on physical
GPUs 5 and 6. Run one dataset at a time:

```bash
bash domain_adaptation/baselines/oracle_contrastive/train_ac4.sh
bash domain_adaptation/baselines/oracle_contrastive/train_cremi_a.sh
bash domain_adaptation/baselines/oracle_contrastive/train_cremi_b.sh
bash domain_adaptation/baselines/oracle_contrastive/train_cremi_c.sh
```

Every completed epoch is retained (`save_top_k: -1`), and checkpoint filenames
contain that epoch's `val_loss_total`. `last.ckpt` is also updated for resuming.

Smoke test on GPU 5:

```bash
CUDA_VISIBLE_DEVICES=5 bash \
  domain_adaptation/baselines/oracle_contrastive/train.sh \
  --fast-dev-run
```

Resume both-GPU training:

```bash
bash domain_adaptation/baselines/oracle_contrastive/train.sh \
  --checkpoint /absolute/path/to/last.ckpt
```
