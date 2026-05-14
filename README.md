# CISC 442 Final — Dog Gait CCL Classifier

This pipeline takes a dog gait video, runs pose estimation on it, and
classifies the dog as **CCL (knee injury)** or **Normal** using a trained
random-forest model.

The 12 GB of clinical dog videos we trained on aren't in this repo (private +
huge), but the trained model is committed at `results/model.joblib`, and we've
included a sample video at `videos/trial_sample.MOV` so you can try the
pipeline on something out of the box. You can also point it at any dog video
you have.

## How to use it

Install dependancies:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install dlclibrary transformers timm opencv-python pandas scikit-learn xgboost joblib Pillow numpy
```

Then run `predict.py` on any video:

```bash
python src/predict.py videos/trial_sample.MOV
python src/predict.py path/to/your_dog.mp4
```

It'll print the CCL probability and verdict to the terminal, and drop a
pose-overlaid version of the video at `results/<video>_overlaid.mp4` so you
can see what the model was looking at.

A couple of useful flags:

- `--force` — skip the lateral-view check (use this if the dog isn't walking
  perfectly sideways; results get less reliable though)
- `--no-overlay` — skip writing the overlay video if you just want the verdict

> **First run note:** the pose model (~300 MB) auto-downloads from HuggingFace
> the first time you run anything. Needs internet that once, cached after.
> GPU is used automatically if available — startup line will say
> `using device: cuda` if it picked up your GPU.

## What's here

- `src/predict.py` — the script described above. This is the one to use.
- `src/main.py` — drives the full training pipeline (dataset scan → feature
  extraction -> training -> evaluation). The whole backend is configurable from
  the globals at the top.
- `src/animal_openpose.py` — pose estimator: RT-DETR for animal detection,
  then SuperAnimal-Quadruped HRNet-W32 (39 keypoints) from DeepLabCut's model
  zoo. We load the checkpoint directly instead of pulling in the full DLC
  package, which has Python 3.13 / linux-related install issues.
- `src/dataset.py`, `src/features.py`, `src/train.py`, `src/evaluate.py` — the
  rest of the training-side pipeline.
- `results/` — committed outputs: the trained model, feature CSV, CV metrics,
  holdout metrics, sample-video pose outputs.

## How the model did

Trained on 12 dogs, tested on 2 held-out dogs the model never saw:

- **Both held-out dogs correctly diagnosed** at the dog level
- Clip-level: 95% accuracy, 0.98 AUC on the holdout
- Cross-validation across the 12 training dogs: 83% dog-level accuracy, 0.97
  dog-level AUC (the more honest number, since the holdout is just 2 dogs)

Random forest won model selection over logistic regression and XGBoost. Full
numbers in `results/cv_metrics.json` and `results/test_metrics.json`.
