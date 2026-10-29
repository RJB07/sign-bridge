# Contributing to SignBridge

SignBridge is a non-profit open-source ASL-to-English tool. The camera and the recognition stay on the device. There is no account system and no database. English text is the only language we output today. The language pack in `src/languages/aslEnglish.ts` is where another sign language could plug in later. Please do not add one in a drive-by change.

## Run it

```bash
npm install
npm test
npm run dev
```

`npm run build` checks `SIGNATURE.json` and writes a static site. Do not remove that file or the lead attribution it records. `npm run test:e2e` needs Chrome (`CHROME_PATH` if it is not on the default path). It replays a held-out clip, checks the English sentence, saves a sign locally, undoes a gloss, types a reply, and reloads offline.

Python checks that do not need the dataset:

```bash
PYTHONPATH=training python -m unittest training.test_augment training.test_splits
```

## What a change should keep true

- The signer-independent number is the PopSign test split. Do not report a training-set score, and do not invent a higher accuracy. The current test top-1 is in `training/metrics/asl_en_metrics.json` and the README.
- More training on the PopSign image cache we already have is not the next accuracy step. The Google ASL Signs path is `training/train_asl_signs.py`. It downloads the `asl-signs` competition only when `KAGGLE_API_TOKEN` is set (or both `KAGGLE_USERNAME` and `KAGGLE_KEY`), writes a signer-disjoint cache, and retrains. `docs/KAGGLE_DATASETS.md` has the details. ASL Citizen and WLASL are not training sources: their licenses do not allow shipping the data or a model built from it.
- PopSign ASL v1.0 is CC BY 4.0. Keep that credit.
- Fingerspelling holdout from `sid220/asl-now-fingerspelling` is not signer-independent. Say so if you quote it.
- Do not upload camera frames. Personal signs in `src/personalize/personalSigns.ts` are landmark summaries in `localStorage`.
- Gloss-to-English lives in `src/textAssembly/glossEnglish.ts`. It is a small set of word-order rules. The raw glosses have to stay visible. Do not describe it as full ASL grammar or as a translation model.
- Help is not one of the 250 PopSign glosses. Do not add a fake “help” class.

## Project layout

- `src/tracking/MediaPipeTracker.ts` loads Holistic from this origin (`public/vendor/mediapipe` and `public/models/holistic_landmarker.task`) so the installed app can run offline.
- `src/recognition/` is features, the landmark transformer, the letter model, and the live segmenter.
- `src/textAssembly/` buffers glosses and builds the English line.
- `training/` downloads, extracts, and trains. `training/cache/` and `training/data/` are gitignored.
- `docs/AUDIT.md` records why there is no database.

Paths to files in `public/` go through `publicUrl()` so a GitHub Pages build under `/sign-bridge/` still finds them.

## Tests and review

GitHub Actions (`.github/workflows/ci.yml`) runs unit tests, the Python checks, a production build, the browser test, and a second build with `GITHUB_PAGES=true`. Add a unit test for a rule or a storage change. Update the end-to-end script if you change the replay, undo, sentence, or “save last sign” controls.

Please keep the interface usable without sound: large text, a visible commit, undo, and captions that do not depend on audio.

## Publish the demo

`.github/workflows/pages.yml` deploys `main` to GitHub Pages. One setting has to be flipped in the repository before the first deploy succeeds:

1. Settings → Pages.
2. Build and deployment → Source: GitHub Actions.

The site is then [https://rjb07.github.io/sign-bridge/](https://rjb07.github.io/sign-bridge/). The workflow does not run on pull requests, so a red Pages check does not appear before that setting exists.

## Data you may train on

| Source | Use |
| --- | --- |
| PopSign ASL v1.0 / the image cache already extracted | Yes. CC BY 4.0. Cite Starner et al. |
| Google ASL Signs on Kaggle (`asl-signs`) | Yes, when you have access. Same 250 glosses, native frame rate. See `docs/KAGGLE_DATASETS.md`. |
| Google ASL Fingerspelling on Kaggle | Yes for a signer-aware letter model, after you confirm the data-page license. |
| ASL Citizen, WLASL, MS-ASL video | No. Research licenses do not allow this project to ship them. |

`training/train_sequence_model.py` replaces `public/models/asl-en-sequence.json` only when the signer-disjoint test top-1 is higher than the file already there.
