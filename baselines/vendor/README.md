# Vendored baseline source snapshots

These directories are source-only snapshots copied from the local research
workspace for review and reproducibility. No network checkout was performed.
Model checkpoints, embedding archives, datasets, logs, generated outputs, and
Python caches were deliberately excluded.

The snapshots are not automatically covered by FiSeR's MIT license. Keep each
upstream license and attribution with its source, and obtain a redistribution
decision before publishing a release containing these files. `NOASSERTION`
means that the local source did not include a machine-readable license; it is
not permission to relicense the code.

| Snapshot | Registry license | Local status |
| --- | --- | --- |
| `AIDE-main` | MIT | source copied |
| `C2P-CLIP-DeepfakeDetection` | NOASSERTION | source copied |
| `CLIPDetection-main` | local FiSeR baseline terms | source copied |
| `CNNDetection-master` | CC-BY-NC-SA-4.0 | source copied; non-commercial restriction |
| `DIRE` | NOASSERTION | source copied |
| `Effort-AIGI-Detection` | NOASSERTION | source copied |
| `FreqNet-DeepfakeDetection-main` | NOASSERTION | source copied |
| `Gram-Net-main` | NOASSERTION | source copied |
| `LASTED` | MIT | source copied |
| `LGrad-master` | NOASSERTION | source copied |
| `LOTA` | MIT | source copied |
| `NPR-DeepfakeDetection-main` | NOASSERTION | source copied |
| `Resnet50-main` | local FiSeR baseline terms | source copied |
| `SAFE-main` | Apache-2.0 | source copied |
| `spai` | Apache-2.0 | source copied |
| `UniversalFakeDetect` | MIT | source copied |

## Compatibility notes

The copied code is kept structurally close to the local workspace. The local
snapshots contain these compatibility adjustments:

- native `test_logger.py` copies honor `BASELINE_LOG_DIR`, so evaluation logs
  can stay outside the source tree;
- Effort includes the WildFake LMDB implementation used for Table 1 and casts
  bf16 predictions to float32 before NumPy conversion;
- LOTA includes the cosine schedule and warmup used by its released checkpoint;
- SPAI and C2P-CLIP include the local LMDB and distributed-GPU evaluation
  adapters used for Table 1;
- C2P-CLIP resolves standard Hugging Face cache locations and honors
  `BASELINE_LOG_DIR` instead of writing evaluation logs into its source tree;
- LASTED includes the already-local OpenAI CLIP files and restores their
  standard Transformer return interface;
- LASTED explicitly uses constant zero padding so Albumentations 1.x and 2.x
  reproduce the same native preprocessing;
- the local shared `robustness_utils.py` is included with AIDE,
  CLIPDetection, and DIRE because their dataloaders import it.
