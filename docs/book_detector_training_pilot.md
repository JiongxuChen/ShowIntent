# Book detector training pilot (24 Nov 2026)

This is an A-module experiment for the new book videos. It is a local pilot, not
a production-model selection.

## Data and split

- 17 manually annotated frames from three videos, with 99 book boxes.
- Training: 11 frames / 66 boxes from clips `20261118_183631` and
  `20261118_184102`.
- Validation: 6 frames / 33 boxes from the held-out clip `20261118_183847`.
- Frames from one clip never cross the train/validation boundary.
- Source videos, annotated images, YOLO labels, weights, predictions, and the
  virtual environment are local and ignored by Git.

## Pilot result

Fine-tuned the existing `yolo11m.pt` from its original weights on RTX 4060
Laptop GPU, at 960 px input size. Training was configured for 30 epochs with
patience 10 and stopped at epoch 11; the best validation mAP50 occurred at
epoch 1. The output checkpoint is local at
`outputs/book_training_pilot/yolo11m/finetune/weights/best.pt`.

The report below uses the same one-to-one box matching rule for both models:
each ground-truth book can match at most one predicted book, and a match needs
IoU >= 0.5. Results are pooled over the 33 manually annotated validation boxes.

| Confidence | Model | Matched / GT | Predictions | Precision | Recall |
|---:|---|---:|---:|---:|---:|
| 0.10 | Pretrained YOLO11m | 17 / 33 | 39 | 43.6% | 51.5% |
| 0.10 | Fine-tuned YOLO11m | 17 / 33 | 38 | 44.7% | 51.5% |
| 0.20 | Pretrained YOLO11m | 16 / 33 | 33 | 48.5% | 48.5% |
| 0.20 | Fine-tuned YOLO11m | 15 / 33 | 32 | 46.9% | 45.5% |
| 0.35 | Pretrained YOLO11m | 14 / 33 | 26 | 53.8% | 42.4% |
| 0.35 | Fine-tuned YOLO11m | 14 / 33 | 25 | 56.0% | 42.4% |

The change is small: recall did not improve, and the fine-tuned checkpoint
should not replace the original model based on this evidence. The validation
set is only six frames from one video, so these numbers are an initial signal,
not a generalization claim.

## Next data action

Eight additional full-resolution candidate frames were extracted between the
existing annotation timestamps. They are stored locally in
`outputs/book_annotation_batch3/images/`; the source timestamps and dimensions
are in `outputs/book_annotation_batch3/frames.csv`. Once labeled, the five frames
from clips `183631` and `184102` can expand training, while the three from
`183847` expand the held-out validation set.

For every candidate frame, draw one separate box around each visible book,
including visible spines. Do not merge adjacent books into one stack box. For a
partly occluded book, box the visible extent if the individual book can still
be distinguished. Preserve the source image and return a marked-up copy.

Import a marked screenshot into the local annotation seed and refresh the
video-disjoint YOLO dataset with:

```powershell
python -m scripts.import_book_annotation --annotation C:\path\to\marked.png --image book_batch3_01_20261118_183631_t25s.jpg
python scripts\prepare_book_training_dataset.py
```

## Reproduce locally

```powershell
python scripts\prepare_book_training_dataset.py
python -m scripts.run_book_training_pilot
```

The pilot runner expects the repository-local `.venv` with CUDA-enabled
PyTorch, Ultralytics, OpenCV, and the ignored local `yolo11m.pt` file. The
resulting dataset and report stay under ignored `outputs/`.
