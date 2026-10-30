# Training

The live model is the `core64-nongame` ensemble: the game core plus every non-game training clip, same validation and test lists (no Kaggle token):

```bash
PYTHONPATH=training python training/scan_popsign.py
PYTHONPATH=training python training/build_cache.py --train-cap 200 --nongame-cap 100000 --core-only --with-nongame --workers 3 --progress training/metrics/extract-progress.jsonl
PYTHONPATH=training python training/train_sequence_model.py --cache training/cache --index index.jsonl --epochs 32 --patience 8 --batch-size 128 --seeds 0,1,2 --augment base --mixup 0 --label-smoothing 0.05 --experiment core64-nongame --skip-student --resume
```

`--core-only` still draws the 64-per-gloss training subset from the cap-200 game train selection, then downloads validation, test, and that subset. `--with-nongame` keeps that same game validation and test list and adds every selected non-game training clip. A cap of `100000` keeps all 27,547 candidates; their signers are already outside game validation and test. Game validation and test signers are never added to training. The 53.7% model omitted those clips:

```bash
PYTHONPATH=training python training/build_cache.py --train-cap 200 --nongame-cap 0 --core-only --workers 3
PYTHONPATH=training python training/train_sequence_model.py --cache training/cache --index index-core.jsonl --epochs 32 --patience 8 --batch-size 128 --seeds 0,1,2 --augment base --mixup 0 --label-smoothing 0.05 --experiment core64-base
```

`download_popsign.py` and `extract_landmarks.py` still build a cache from whole game shards when those files are already on disk.

See the repository README for metrics, licenses, and why signer ids cannot cross the train, validation, and test splits.

`train_sl2t_dataset.py` and `train_fingerspelling.py` no longer write a fake accuracy. Do not train on generated circles.
