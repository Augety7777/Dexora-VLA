# `tools/data/` — Internal data-operations scripts

These scripts are **not part of the published Dexora training pipeline** —
they were used internally to sync, ingest, and inspect raw BSON / LeRobot
captures while building the dataset. They are kept under version control
because they are useful for anyone reproducing our data pipeline, but they
are intentionally separated from the public entry points (`train_ours.sh`,
`train_scoring.sh`, `post_train.sh`, `run_all_stages.sh`).

| Script | Purpose |
|---|---|
| `batch_sync_bson.py`             | Bulk sync raw BSON episodes from a remote staging server. |
| `batch_sync_bson_break_exist.py` | Variant that skips already-synced episodes. |
| `batch_sync_bson_kefei.py`       | Variant for the Kefei capture site. |
| `restore_backup_bson.py`         | Restore episodes from a tar/zip backup. |
| `ref_fast_stat.py`               | Quick smoke statistics over a BSON dump. |
| `plot_buggy_episodes.py`         | Visualise episodes flagged as broken during ingestion. |
| `vqvae_training.py`              | (Legacy) experiment with a VQ-VAE action prior; not used in the ICRA'26 paper. |
| `hf_upload_script.py`            | Upload checkpoints / datasets to the Hugging Face Hub. |

If you are following the public README, you do not need any of these
scripts. They run only against the internal data staging that the Dexora
team uses, and may need hard-coded paths to be patched before use.
