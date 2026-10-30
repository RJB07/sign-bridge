# Kaggle datasets to add later

These two sets are the ones that would change the model if a Kaggle API token is available. Do not download them until `KAGGLE_USERNAME` and `KAGGLE_KEY` are set. Both are the same underlying MediaPipe landmark family this app already consumes, so they plug into `training/features.py` rather than a new recognizer.

## Google ASL Signs (full landmark release)

- Competition: `asl-signs` (Google Isolated Sign Language Recognition)
- Command: `kaggle competitions download -c asl-signs`
- License: CC BY 4.0. Do not use the landmarks to identify a person.
- What is in it: `train.csv` (`path`, `participant_id`, `sequence_id`, `sign`), `sign_to_prediction_index_map.json` (the same 250 glosses), and `train_landmark_files/{participant_id}/{sequence_id}.parquet`.
- Each parquet is long-form MediaPipe Holistic: `frame`, `type` (`face`, `pose`, `left_hand`, `right_hand`), `landmark_index`, `x`, `y`, `z`. About 94,000 sequences from 21 signers. The archive is on the order of 50 GB.
- What it adds over the frames we train on now: the same words, including hello, thank you, please, yes, and no, but at the original video frame rate instead of the ~5 fps stills in `sign/popsign-images`, and hundreds of examples per gloss instead of the 64 game clips per gloss in the shipped core subset. That is the most direct way to raise signer-independent top-1 without changing the label set. It does not add “help”; help is not one of the 250.
- How to run it end to end, once a token is in the environment:

```bash
export KAGGLE_API_TOKEN=...
# or: export KAGGLE_USERNAME=... KAGGLE_KEY=...
pip install kaggle
PYTHONPATH=training python training/train_asl_signs.py
```

That downloads `asl-signs` into `training/data/asl-signs` (gitignored), writes a signer-disjoint cache at `training/cache-asl-signs`, and calls `training/train_sequence_model.py`. Participants are sorted and split about 65% / 15% / 20% into train / validation / test, with no participant in two splits. The trainer replaces `public/models/asl-en-sequence.json` only when test top-1 is higher than the file already there. Do not mix these sequences into the 5 fps PopSign image cache.

## Google ASL Fingerspelling

- Competition: `asl-fingerspelling`
- Command: `kaggle competitions download -c asl-fingerspelling`
- License: treat it as CC BY 4.0, the same terms Google used for the ASL Signs competition, and confirm the data page before the first download.
- What is in it: `train.csv` with `participant_id`, `sequence_id`, and a `phrase`; landmark files in wide format (`x_left_hand_0`, …) with many sequences per file. The archive is on the order of 200 GB. Some supplemental phrases were generated from addresses and URLs.
- What it adds: signer ids, so a real signer-independent fingerspelling split, and the motion of J and Z instead of the single still frame in `sid220/asl-now-fingerspelling`. The current letter model’s 92.3% holdout is not signer-independent because those files have no participant id.
- How to plug it in: slice the dominant-hand columns over time, train a short sequence model (the same transformer, or a small CTC model if the target is a whole phrase), and replace `public/models/asl-fingerspelling.json`. Keep the static MIT model only as a fallback.

## Not a Kaggle drop-in

ASL Citizen and WLASL are not on this list. ASL Citizen’s Microsoft research license forbids distributing the data or modifications. WLASL does not grant redistribution of the videos.
