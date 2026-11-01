import torch
from torchtitan.models.glimmer_dense import build_model_config

V, T, W = 100352, 2048, 512
FULL_CTX = (T + 1) / 2
SWA_CTX = (W * (W + 1) / 2 + (T - W) * W) / T


def stats(m, dim):
    n = sum(p.numel() for p in m.parameters())
    nonemb = n - V * dim  # tied embedding counted once
    att = sum(
        12 * l.attention.n_heads * l.attention.head_dim
        * (SWA_CTX if l.attention.window_size else FULL_CTX)
        for l in m.layers.values()
    )
    return n, 6 * (nonemb + V * dim) + att, 6 * nonemb + att


def build(fl, iq=None):
    with torch.device("meta"):
        if iq is None:
            return build_model_config(fl, head_dim=128).build()
        return build_model_config(
            fl, arch="qmlpdeep", head_dim=128, qmlp_swa_per_full=7,
            qmlp_intermediate=iq, qmlp_residual=True,
        ).build()


for fl in ["d200m", "d400m", "d700m", "d1b"]:
    d = build(fl)
    dim = d.config.dim
    a = d.layers["0"].attention
    nd, fd, md = stats(d, dim)
    D = -(-20 * nd // 2**28) * 2**28
    C = md * D
    lr, B = 0.3118 * C**-0.125, 0.2920 * C**0.3271
    out = {iq: stats(build(fl, iq), dim)[:2] for iq in range(64, 3 * dim, 64)}
    by_f = max(iq for iq, (nq, fq) in out.items() if fq <= fd)
    by_p = max(iq for iq, (nq, fq) in out.items() if nq <= nd)
    iq_r = int(round(1.375 * dim / 64)) * 64
    nq, fq = out[iq_r]
    print(f"{fl}: hidden {dim} heads {a.n_heads}/{a.n_kv_heads}  dense {len(d.layers)}L {nd/1e6:.1f}M "
          f"FLOPs/tok {fd/1e9:.3f}G  tokens {D/1e9:.2f}B  DeepSeek lr {lr:.2e} batch {B/1e3:.0f}K")
    print(f"    qmlp 7:1 iQ=1.375x ({iq_r}): {nq/1e6:.1f}M ({(nq/nd-1)*100:+.1f}%) FLOPs {(fq/fd-1)*100:+.1f}%"
          f"   | largest iQ with FLOPs<=dense: {by_f} ({by_f/dim:.2f}x, params {(out[by_f][0]/nd-1)*100:+.1f}%)"
          f"   | largest iQ with params<=dense: {by_p} ({by_p/dim:.2f}x, FLOPs {(out[by_p][1]/fd-1)*100:+.1f}%)")
