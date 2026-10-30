# Training

The live model is trained from the full PopSign frame catalog (no Kaggle token):

```bash
PYTHONPATH=training python training/scan_popsign.py
PYTHONPATH=training python training/build_cache.py --train-cap 200 --nongame-cap 0 --core-only --workers 3
PYTHONPATH=training python training/train_sequence_model.py --cache training/cache --index index-core.jsonl --epochs 32 --patience 8 --batch-size 128 --seeds 0,1,2 --augment base --mixup 0 --label-smoothing 0.05 --experiment core64-base
```

`--core-only` still draws the 64-per-gloss training subset from the cap-200 game train selection, then downloads validation, test, and that subset. Non-game clips are optional extra training (`--nongame-cap 80` without `--core-only`). Game validation and test signers are never added to training.

`download_popsign.py` and `extract_landmarks.py` still build a cache from whole game shards when those files are already on disk.

See the repository README for metrics, licenses, and why signer ids cannot cross the train, validation, and test splits.

`train_sl2t_dataset.py` and `train_fingerspelling.py` no longer write a fake accuracy. Do not train on generated circles.
