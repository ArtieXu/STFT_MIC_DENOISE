# Data

Two clean-heart-sound sources and one noise source. Everything is 4 kHz, 2 s
windows (8000 samples). `src/pools.py` is the only place that reads them.

```
data/
  clean/step_1s/{train,val}/*.npz      device recordings, committed
  noise/step_1s/{train,val}/*.npz      device motion/ambient noise, committed
  circor/circor_pool_4khz_2s.npz       CirCor pool, you build this once
```

The experiment is `clean/` device-only versus `clean/` device + CirCor, with
`noise/` and the validation split held fixed. So `clean/` is the only thing that
changes between the two arms.

## device — committed

From <https://github.com/jiayimaggieshao/denoise_stft> (`step_1s` = 2 s windows
at a 1 s hop, so stored windows overlap 50%). `scripts/fetch_device_data.py`
re-downloads them, or pulls another hop.

| Split | clean | noise |
|-------|-------|-------|
| `train` | `heart_aw1,aw2,aw6,bw1,bw2,bw5,bw6` — 3936 windows | `noise1,2,3,5` — 1367 stored, 1345 after dropping 22 clipped |
| `val` | `heart_aw4`, `heart_bw4` — 1024 windows | `noise6` — 377 windows |

`bw` = before walking, `aw` = after walking. Subject 4 is held out, so device
subjects never cross the train/val line.

### NPZ format

| Key | Shape | Notes |
|-----|-------|-------|
| `x` | `(N, 8000)` int16 | full scale 32768 |
| `start_idx` | `(N,)` int64 | index of the first sample |
| `segment_id` | `(N,)` int32 | contiguous-index segment |
| `start_wall_epoch_us` | `(N,)` int64 | wall clock, used to align the reference mic |

Only `x` is used here; the mixing is synthetic, so the timestamps and segment
ids are not needed for this experiment.

## CirCor — build once, not committed

449 MB of source audio, so only the sampled pool lives here.

```bash
wget -O circor.zip https://physionet.org/content/circor-heart-sound/get-zip/1.0.3/
unzip -q circor.zip
PYTHONPATH=. python scripts/build_circor_pool.py --root circor-heart-sound-1.0.3
```

Writes `circor/circor_pool_4khz_2s.npz` (float32 in ±1) with a `split` column
that is subject-disjoint, plus per-window `subject`, `location`, `record`,
`bpm`, `murmur`, `age`. `bpm` comes from consecutive S1 onsets in the TSV, which
is how `--circor_heart_rate_max` can filter an already built pool.

Two filters are applied while sampling, and both matter:

- **only inside contiguous nonzero-state TSV runs.** State 0 is CirCor's own
  signal-quality label; the documented noise in those regions is stethoscope
  rubbing, speech, crying and laughing. Handing that to a denoiser as a clean
  target teaches it to output noise.
- **subjects with `Murmur == Unknown` are dropped** — the 119 subjects whose
  recordings did not meet the signal-quality standard.

CirCor is pediatric (0–21 y, ~107 bpm median) recorded with a digital
stethoscope; the device data is adult at 61–79 bpm. Rhythm is the main cue the
model has when heart sound and motion artifact overlap in frequency, so keep
`--circor_heart_rate_max` in mind. `scripts/audit_pools.py` prints the bpm gap.

## What is deliberately not here

The upstream repo also ships `test_real/walking/` — real walking recordings with
no clean reference. They cannot produce a number, so they cannot take part in
this comparison, and they are left out. `scripts/fetch_device_data.py` can pull
them if the question ever changes to a listening test.
