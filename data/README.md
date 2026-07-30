# Data

The experiment uses clean heart-sound windows and measured noise-only windows.
Every sample is 2 seconds at 4 kHz (8,000 samples). Training mixtures are made
online with one transparent equation:

```text
noisy_chest = clean_heart_sound + noise_scaled_to_snr
```

There is no reference microphone input and no hidden transfer function,
leakage, transient injection, or bandpass target.

## Device protocol

The experiment has one fixed-budget training stage followed by one held-out
test. The exact device recordings are:

| Role | Clean files | Noise-only files |
|---|---|---|
| train | `heart_aw1`, `heart_bw1`, `heart_aw2`, `heart_bw2`, `heart_aw4`, `heart_bw4`, `heart_bw5` | `noise1`, `noise2`, `noise3`, `noise5` |
| final synthetic test | `heart_aw6`, `heart_bw6` | `noise6` |
| qualitative only | `heart_w6` | none |

Subject 6 is absent from every optimization input. `heart_w6` is a real walking
recording with no clean reference, so it cannot produce a denoising score.
Training and evaluation settings must be frozen before inspecting subject 6;
subject-6 results must not be used to tune or select another checkpoint.

Noise-only recordings are sampled independently of clean subjects. The four
training noise recordings are scheduled exactly 1:1:1:1 per epoch, followed by
a window from the scheduled recording; numeric suffixes do not constrain a
synthetic pair. In the combined arm, every device/CirCor-clean ×
noise-recording combination is equally represented. Recording-uniform sampling
also prevents the longest noise file from dominating.

There is no validation split, validation-based checkpoint selection, early
stopping, or plateau scheduler. All arms train for the same fixed
epoch/optimizer-step budget and are compared using `final.pt`; `last.pt` is
only for resuming an interrupted run.

The upstream archives still live under their legacy `train/` and `val/`
directories. The loader uses the manifest above, so `heart_aw4` and
`heart_bw4` participate in training without moving binary files.

Device files come from
<https://github.com/jiayimaggieshao/denoise_stft>. Stored `step_1s` windows have
a 1-second hop and therefore overlap 50%. `scripts/fetch_device_data.py` pins
upstream commit `c38092d286e03fea81f72fa66c7731100d9266ec` so re-fetching cannot
silently change the experiment.

## CirCor

CirCor supplies additional clean targets only. Only its stored `train` split is
used; its stored `val` split and all CirCor noise/test roles are excluded.
Patient IDs connected by the official `Additional ID` field are canonicalized
as one real subject before window limiting and split assignment.

```bash
wget -c -O circor.zip https://physionet.org/content/circor-heart-sound/get-zip/1.0.3/
unzip -q circor.zip
PYTHONPATH=. python scripts/build_circor_pool.py \
  --root circor-heart-sound-1.0.3
```

This writes `data/circor/circor_pool_4khz_2s_v2.npz`. The `v2` schema merges
linked Patient IDs and rejects older cached pools. The combined training
dataset schedules exactly half of its clean targets from device and half from
the CirCor `train` split, regardless of the physical pool sizes. Device-only
and combined runs use the same total `samples_per_epoch` and optimizer-step
budget.

## NPZ format

Device archives contain an `x` array with shape `(N, 8000)`. Optional timestamp
and segment arrays are provenance only and are not used to synthesize training
mixtures.

All windows pass through the same conversion, DC removal, purity check, and RMS
normalization in `src/pools.py`. Provenance fields retain source, recording,
and original file name for leakage auditing.
