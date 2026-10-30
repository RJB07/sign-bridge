# Data sources

The signer-independent number is top-1 on the official PopSign **game** test partition (seed 0, at most 20 clips per gloss). A signer who appears there is never used for training. The shipped weights score **53.7% top-1** and **77.9% top-5** on the 4,540 clips that remain after the two-frame filter.

## Used

| Source | License | How it is reached | What it adds |
| --- | --- | --- | --- |
| [sign/popsign-images](https://huggingface.co/datasets/sign/popsign-images) game subset | CC BY 4.0 | HTTPS, no login | The 250 PopSign / Google ASL Signs glosses. Frames at about 5 fps. This is the test split. |
| Same dataset, non-game subset | CC BY 4.0 | HTTPS, no login | Extra training clips. A clip is kept only when its signer is outside the game validation and test partitions. |

Cite Starner et al., PopSign ASL v1.0 (2023). Landmark files are derived and are not a way to identify the signers. Weights in this repo are the derived artifact; the videos are not.

`training/scan_popsign.py` reads the gloss and signer columns with HTTP range requests. `training/build_cache.py` downloads image bytes only for the selected row groups.

## Checked, not used

| Source | Why it is not in the training set |
| --- | --- |
| Google ASL Signs (`asl-signs`) and ASL Fingerspelling on Kaggle | CC BY 4.0, same 250 glosses, native-rate landmarks. The competition download returns HTTP 404 without a Kaggle token. No public Hugging Face or `storage.googleapis.com` mirror of `train_landmark_files` was found. `training/train_asl_signs.py` still runs once `KAGGLE_API_TOKEN` (or `KAGGLE_USERNAME` and `KAGGLE_KEY`) is set. |
| [martinctl/how2sign-asl-landmarks](https://huggingface.co/datasets/martinctl/how2sign-asl-landmarks) | CC BY-NC 4.0. Sentence-level How2Sign, not the 250 isolated glosses. An MIT-licensed weight file trained on it would let downstream commercial use contradict the non-commercial source. |
| [Kibalama/wlasl-upper-arm-pose-landmarks](https://huggingface.co/datasets/Kibalama/wlasl-upper-arm-pose-landmarks) | Pose-only, about 1,000 clips, no signer id, no license that allows shipping weights. WLASL video rights are research-only. |
| [Siruyy/asl-static-landmarks-v1](https://huggingface.co/datasets/Siruyy/asl-static-landmarks-v1) | CC BY 4.0, but static fingerspelling letters, not the 250 glosses. |
| `cornelismus/asl_signs` and `Keithsel/asl_signs` | Alphabet photos, not the isolated-sign landmark release. No usable license card. |
| ASL Citizen | Microsoft research license forbids distributing the data or modifications. |
| WLASL / MS-ASL videos | Research-only or C-UDA. Not redistributable as weights. |
