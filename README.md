# Single-microphone STFT heart-sound denoising

This project answers two practical questions:

1. Does adding CirCor clean-heart windows improve denoising on held-out device
   data?
2. How does the resulting STFT system compare with the previous waveform
   denoising system?

The comparison is intentionally small. Every arm receives the same single
noisy waveform and predicts the same clean waveform.

| Arm | Model | Clean training source |
|---|---|---|
| `device_only` | simple 2-D STFT U-Net | device |
| `combined` | the identical 2-D STFT U-Net | 50% device, 50% CirCor |
| `waveform` | waveform CleanUNet | device |

The waveform result is a denoising-system comparison, not a claim that input
representation is the only difference between the architectures.

## Minimal signal path

Training data is generated with:

```text
noise_scaled = noise * rms(clean) / (rms(noise) * 10^(SNR_dB / 20))
noisy_chest  = clean + noise_scaled
```

There is no second/reference microphone, random transfer path, leakage,
dropout, transient injection, hidden bandpass, or chest-only noise.

The STFT arm is:

```text
noisy waveform
  -> fixed STFT
  -> [real, imaginary]                    # exactly two real Conv2d channels
  -> ordinary three-level 2-D U-Net
  -> predicted complex residual
  -> add to noisy STFT
  -> fixed ISTFT
```

It contains no recurrent block, analytical spectral-subtraction prior,
hand-crafted input features, or constrained mask. Its only training objective
is target-normalized complex Re/Im L1. The same loss is computed from the final
waveform for all arms.

## Single-stage leakage-safe protocol

Training uses every available non-test device recording:

- clean:
  `heart_aw1`, `heart_bw1`, `heart_aw2`, `heart_bw2`, `heart_aw4`,
  `heart_bw4`, and `heart_bw5`
- noise-only: `noise1`, `noise2`, `noise3`, and `noise5`

Clean and noise are sampled independently. The four noise recordings are
scheduled exactly 1:1:1:1 per epoch and then a window is selected within the
scheduled recording, so a long recording cannot dominate and the numeric
suffix never forces `clean1 + noise1`, for example. In the combined arm, every
device/CirCor-clean × noise-recording combination is equally represented. The
measured noise-only recordings are nuisance sources, not clean-heart targets.

Subject 6 is completely held out. The final synthetic test uses
`heart_aw6`/`heart_bw6` with `noise6`. The real walking recording `heart_w6`
has no clean reference and is used only for a fixed qualitative spectrogram and
listening example. Freeze the model, loss, epoch budget, and SNR evaluation
recipe before the first subject-6 result; changing them in response to that
result would turn the final test into a tuning set.

There is no validation-based model selection, early stopping, or
validation-driven plateau scheduler. Each arm receives the same fixed
epoch/optimizer-step budget, and the checkpoint after the complete budget is
written as `final.pt`. `last.pt` is only a resumable training state.

Audit the device arm now; run the second command after building CirCor below:

```bash
PYTHONPATH=. python scripts/audit_pools.py --no_circor
PYTHONPATH=. python scripts/audit_pools.py
```

## Prepare CirCor

```bash
wget -c -O circor.zip \
  https://physionet.org/content/circor-heart-sound/get-zip/1.0.3/
unzip -q circor.zip
PYTHONPATH=. python scripts/build_circor_pool.py \
  --root circor-heart-sound-1.0.3
```

CirCor is used only for combined training, and only its stored `train` split is
eligible. Its stored `val` split is not used. The dataset schedules device and
CirCor clean targets exactly 1:1, so its larger window pool cannot dominate.
Official `Additional ID` aliases are merged before subject limiting and
train/val assignment, so repeat visits by one person cannot be double-weighted
or split as if they were different people.
All three runs use the same `samples_per_epoch`, batch size, epoch count,
optimizer-step count, SNR range, seed, and optimizer settings.

## Verify and train

```bash
pip install -r requirements.txt
PYTHONPATH=. python scripts/check_frequency_pipeline.py
PYTHONPATH=. python train_frequency.py --smoke --device cpu --no_circor
```

```bash
C="--epochs 60 --samples_per_epoch 20000 --batch_size 16 --seed 2026"

PYTHONPATH=. python train_frequency.py $C --no_circor \
  --output_dir checkpoints/device_only

PYTHONPATH=. python train_frequency.py $C \
  --output_dir checkpoints/combined

PYTHONPATH=. python train_frequency.py $C --no_circor --arch cleanunet \
  --output_dir checkpoints/waveform
```

Each `final.pt` records its fixed training budget, explicit role
(`device_only`, `combined`, or `waveform`), single-microphone input mode, and
model, loss, mixing, and pool metadata. Incomplete/resume checkpoints and old
two-input checkpoints are deliberately rejected by the comparison scripts.

## Compare denoising effects

```bash
PYTHONPATH=. python scripts/compare_datasets.py \
  --device_only checkpoints/device_only/final.pt \
  --combined checkpoints/combined/final.pt \
  --waveform checkpoints/waveform/final.pt

PYTHONPATH=. python scripts/fetch_device_data.py --include-test-real
PYTHONPATH=. python scripts/make_demo_figures.py --all \
  --device_only checkpoints/device_only/final.pt \
  --combined checkpoints/combined/final.pt \
  --waveform checkpoints/waveform/final.pt
```

All models receive byte-identical synthetic test mixtures. The output focuses
on per-SNR denoising metrics and shared-scale spectrograms. It intentionally
does not report window-level p-values or confidence intervals: overlapping
windows from one held-out subject are not independent population samples.

Spectrogram panels use the same example, frequency range, dynamic range, and
color scale across arms. The real walking figure uses a fixed interval and is
labelled qualitative because no clean target exists.

For a final claim, repeat the three arms with at least three matched seeds.
One seed is sufficient only for a pipeline smoke test or preliminary figure.

## Main files

```text
train_frequency.py
src/
  frequency_data.py       clean + scaled noise
  frequency_model.py      two-plane complex STFT U-Net
  frequency_loss.py       normalized complex Re/Im L1
  waveform_model.py       single-input waveform baseline
  pools.py                normalization, provenance, subject splits
  arms.py                 checkpoint construction and protocol checks
scripts/
  audit_pools.py
  compare_datasets.py
  make_demo_figures.py
  benchmark.py
  check_frequency_pipeline.py
notebooks/
  colab_demo.ipynb
```
