"""Wall-clock benchmark for the Metal spconv on MPS at trellis2-relevant shapes.

Reports per-call latency for the new MPS-native flex_gemm path, and contrasts
with the old "force-to-CPU and wait" behavior the previous mtlgemm shipped
(emulated by setting FLEX_GEMM_AUTOTUNE=1, which re-enables the per-call sync
on the CPU dispatch path; we additionally force CPU tensors into the call to
mimic what the old code was doing on every input).
"""

import os
import sys
import time
import torch

assert torch.backends.mps.is_available(), "Need MPS"

DEVICE = "mps"


def sphere_coords(res: int, ch: int, dtype: torch.dtype, device: str):
    coords = torch.stack(torch.meshgrid(
        torch.arange(res), torch.arange(res), torch.arange(res), indexing="ij",
    ), dim=-1).int().contiguous()
    dist = ((coords.float() - res / 2 + 0.5) ** 2).sum(dim=-1).sqrt()
    active = (dist <= res / 2) & (dist >= res / 2 - 1.25)
    coords = torch.nonzero(active).int()
    coords = torch.cat([torch.zeros(coords.shape[0], 1, dtype=torch.int32), coords], dim=-1)
    coords = coords.contiguous().to(device)
    feats = torch.randn(coords.shape[0], ch, dtype=dtype).to(device).contiguous()
    return feats, coords, torch.Size([1, ch, res, res, res])


def bench(label, fn, num_warmup=5, num_iter=50, sync_fn=None):
    for _ in range(num_warmup):
        fn()
    if sync_fn:
        sync_fn()
    t0 = time.perf_counter()
    for _ in range(num_iter):
        fn()
    if sync_fn:
        sync_fn()
    elapsed = time.perf_counter() - t0
    per_call_ms = (elapsed / num_iter) * 1000.0
    print(f"  {label:48s} {per_call_ms:8.3f} ms/call")
    return per_call_ms


def run_shape(res, ch, dtype):
    # Re-import so module-level FLEX_GEMM_AUTOTUNE is honored if it was just set.
    from importlib import reload
    import flex_gemm
    from flex_gemm.ops.spconv import sparse_submanifold_conv3d, Algorithm, set_algorithm

    Co = ch
    Ks = 3
    V = Ks ** 3
    set_algorithm(Algorithm.IMPLICIT_GEMM)

    # MPS path (the new code)
    feats_m, coords_m, shape = sphere_coords(res, ch, dtype, "mps")
    weight_m = torch.randn(Co, Ks, Ks, Ks, ch, dtype=dtype).to("mps")
    bias_m = torch.randn(Co, dtype=dtype).to("mps")
    N = feats_m.shape[0]

    # Build cache once so the timing only covers the GEMM kernel itself.
    _, cache_m = sparse_submanifold_conv3d(feats_m, coords_m, shape, weight_m, bias_m)

    # CPU path (mimics the OLD broken code's behavior — every input went to CPU
    # and outputs were CPU tensors. Comparing these two tells us the perf gain
    # from staying on MPS.)
    feats_c = feats_m.cpu().contiguous()
    coords_c = coords_m.cpu().contiguous()
    weight_c = weight_m.cpu().contiguous()
    bias_c = bias_m.cpu().contiguous()
    _, cache_c = sparse_submanifold_conv3d(feats_c, coords_c, shape, weight_c, bias_c)

    print(f"\nshape: res={res}, ch={ch}, dtype={dtype}, N={N}, V={V}")

    def _mps():
        sparse_submanifold_conv3d(feats_m, coords_m, shape, weight_m, bias_m, neighbor_cache=cache_m)

    def _cpu_old_path():
        # Simulate the old broken behavior: take MPS inputs, drop to CPU, run
        # kernel on CPU memory, return CPU tensor — what every spconv call did.
        feats_round = feats_m.cpu().contiguous()
        weight_round = weight_m.cpu().contiguous()
        bias_round = bias_m.cpu().contiguous()
        out, _ = sparse_submanifold_conv3d(feats_round, coords_c, shape,
                                            weight_round, bias_round, neighbor_cache=cache_c)
        # And then the next op would push it back to MPS.
        out.to("mps")

    def _cpu_native():
        # Best case for the CPU path: inputs already on CPU, no round-trip.
        sparse_submanifold_conv3d(feats_c, coords_c, shape, weight_c, bias_c, neighbor_cache=cache_c)

    mps_ms = bench("flex_gemm MPS-native (new)", _mps, sync_fn=torch.mps.synchronize)
    old_ms = bench("flex_gemm OLD CPU-bounce path (per-call MPS roundtrip)", _cpu_old_path, sync_fn=torch.mps.synchronize)
    cpu_ms = bench("flex_gemm CPU-only (best case for CPU path)", _cpu_native)

    speedup_vs_old = old_ms / mps_ms
    print(f"  speedup vs old MPS-bouncing behavior: {speedup_vs_old:.2f}x")
    return mps_ms, old_ms, cpu_ms


print("=" * 75)
print("mtlgemm Metal spconv benchmark on MPS")
print(f"PyTorch {torch.__version__}, MPS device: {torch.mps.current_allocated_memory() // 1024**2}MB")
print("=" * 75)
results = []
# Trellis2-decoder-relevant shapes
for res, ch, dtype in [
    (16, 64, torch.float16),
    (32, 64, torch.float16),
    (32, 128, torch.float16),
    (64, 128, torch.float16),
    (64, 256, torch.float16),
]:
    mps, old, cpu = run_shape(res, ch, dtype)
    results.append((res, ch, dtype, mps, old, cpu))

print("\n" + "=" * 75)
print("Summary (lower is better)")
print("=" * 75)
print(f"{'shape':28s}  {'mps-new':>10s}  {'mps-old':>10s}  {'cpu-only':>10s}  {'gain':>6s}")
for res, ch, dt, m, o, c in results:
    print(f"res={res:3d} ch={ch:3d} dt={str(dt).split('.')[-1]:8s}  "
          f"{m:7.3f}ms  {o:7.3f}ms  {c:7.3f}ms  {o/m:5.2f}x")
