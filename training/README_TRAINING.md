# Training

The live model is trained by:

```bash
PYTHONPATH=training python training/download_popsign.py
PYTHONPATH=training python training/extract_landmarks.py
PYTHONPATH=training python training/train_sequence_model.py
```

See the repository README for metrics, licenses, and why signer ids cannot cross the train, validation, and test splits.

`train_sl2t_dataset.py` and `train_fingerspelling.py` no longer write a fake accuracy. Do not train on generated circles.
