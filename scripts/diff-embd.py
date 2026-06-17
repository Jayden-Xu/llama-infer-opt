#!/usr/bin/env python3
"""Compare ViT embedding dumps from MTMD_DUMP_EMBD.

Each dump file format:
    int32 ne0, int32 ne1, int32 ne2, int32 ne3
    float32 [ne0 * ne1 * ne2 * ne3]   (row-major: ne0 fastest, ne3 slowest)

Typical workflow:
  baseline (single image, OPT_VIT_BATCH=0, N_IMAGES=1):
      MTMD_DUMP_EMBD=/tmp/embd_base ./llama-server ...
      -> writes /tmp/embd_base.000

  vit-batch (4 same images, OPT_VIT_BATCH=1, N_IMAGES=4):
      MTMD_DUMP_EMBD=/tmp/embd_batch OPT_VIT_BATCH=1 ./llama-server ...
      -> writes /tmp/embd_batch.000 with shape [n_embd, n_tokens, 4]

  python scripts/diff-embd.py /tmp/embd_base.000 /tmp/embd_batch.000

It prints a per-image-slice max abs diff against the baseline. If the
batched path is correct, slice 0/1/2/3 should all match the baseline up
to fp16/quant rounding (~1e-3).
"""
import sys
import struct

import numpy as np


def load(path):
    with open(path, "rb") as f:
        hdr = struct.unpack("4i", f.read(16))
        ne0, ne1, ne2, ne3 = hdr
        n = ne0 * ne1 * ne2 * ne3
        data = np.frombuffer(f.read(n * 4), dtype=np.float32)
    arr = data.reshape(ne3, ne2, ne1, ne0)  # row-major: ne0 fastest -> last
    return arr, hdr


def main():
    if len(sys.argv) < 3:
        print("usage: diff-embd.py <baseline.bin> <batched.bin>", file=sys.stderr)
        sys.exit(2)
    base_path, batch_path = sys.argv[1], sys.argv[2]
    base, base_hdr = load(base_path)
    batch, batch_hdr = load(batch_path)

    print(f"baseline {base_path}: ne={base_hdr}  shape={base.shape}")
    print(f"batched  {batch_path}: ne={batch_hdr}  shape={batch.shape}")

    # baseline shape: [ne3=1, ne2=1or B, ne1=n_tok, ne0=n_embd]
    # batched  shape: [ne3=1, ne2=B,     ne1=n_tok, ne0=n_embd]
    # collapse leading singleton dims and align on the slice axis.
    while base.ndim > 3 and base.shape[0] == 1:
        base = base[0]
    while batch.ndim > 3 and batch.shape[0] == 1:
        batch = batch[0]
    if base.ndim == 2:
        base = base[None, ...]   # [1, n_tok, n_embd]
    if batch.ndim == 2:
        batch = batch[None, ...]

    base_slice = base[0]  # baseline always single image
    print(f"\nbaseline slice 0: shape={base_slice.shape}, "
          f"mean={base_slice.mean():.6f}, std={base_slice.std():.6f}, "
          f"min={base_slice.min():.6f}, max={base_slice.max():.6f}")

    print()
    n_slices = batch.shape[0]
    for i in range(n_slices):
        cur = batch[i]
        if cur.shape != base_slice.shape:
            print(f"slice {i}: shape mismatch base={base_slice.shape} batch={cur.shape}")
            continue
        diff = cur - base_slice
        max_abs = np.max(np.abs(diff))
        rel = max_abs / (np.max(np.abs(base_slice)) + 1e-9)
        cos = float(np.dot(cur.ravel(), base_slice.ravel()) /
                    (np.linalg.norm(cur.ravel()) * np.linalg.norm(base_slice.ravel()) + 1e-9))
        print(f"slice {i}: max|diff|={max_abs:.6f}  rel={rel:.4e}  "
              f"cos_sim={cos:.6f}  mean(batch)={cur.mean():.6f}  std(batch)={cur.std():.6f}")


if __name__ == "__main__":
    main()
