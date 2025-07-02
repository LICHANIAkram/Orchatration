#!/usr/bin/env python3
"""
impl.py — Adaptive cipher selector pour le jeu vidéo DCSASS
Théorie : a* = argmax_a S(a, x)
Usage  : python impl.py --dataset PATH [--alpha A] [--beta B]
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import mimetypes
import os
import time
import hashlib
from functools import cache
from typing import Callable, Final, Iterable, Sequence

# ───────────────────────── 1. Crypto back-ends ──────────────────────────────
MISSING: list[str] = []

# The original version relied on external cryptographic libraries that may not
# be available in all environments.  To keep this script self contained, we use
# a tiny pure Python fallback located in ``simplecrypto``.  The implementations
# are **not** secure and merely provide deterministic transformations so that
# timing-based comparisons remain possible.
from simplecrypto import get_cipher

HAVE_TWOFISH = HAVE_CAMELLIA = True
HAVE_AESGCM = HAVE_XCHACHA = True
HAVE_ASCON = True

# ───────────────────────── 2. Paramètres globaux ────────────────────────────
CURRENT_YEAR: Final[int] = _dt.datetime.now(_dt.timezone.utc).year
CHUNK = 8 * 1024 * 1024          # 8 MiB
MAX_LABEL_LEN: Final[int] = 20   # pour le calcul de sensibilité

# Pondération de Sec(a,x)
OMEGA_KS, OMEGA_AR, OMEGA_AP, OMEGA_CA = 0.40, 0.35, 0.05, 0.20
# Pondération Score S(a,x)
ALPHA_DEF, BETA_DEF = 2.0, 0.5

# ───────────────────────── 3. Tables statiques ──────────────────────────────
ALGORITHMS: Final[Sequence[str]] = (
    "AES-128", "AES-256", "ChaCha20",
    "Blowfish", "Twofish", "Camellia-256", "ASCON-128",
)

KEY_BITS = {
    "AES-128": 128, "AES-256": 256, "ChaCha20": 256,
    "Blowfish": 128, "Twofish": 128,
    "Camellia-256": 256, "ASCON-128": 128,
}
FIRST_YEAR = {
    "AES-128": 2001, "AES-256": 2001, "ChaCha20": 2008,
    "Blowfish": 1993, "Twofish": 1998,
    "Camellia-256": 2003, "ASCON-128": 2014,
}
CONTENT_ADAPT = {  # vidéo uniquement
    "video": {
        "AES-128": 0.75, "AES-256": 0.80, "ChaCha20": 0.78,
        "Blowfish": 0.60, "Twofish": 0.70,
        "Camellia-256": 0.77, "ASCON-128": 0.83,
    }
}
KNOWN_VULN = {
    "AES-128": 0.02, "AES-256": 0.02,
    "ChaCha20": 0.04,
    "Blowfish": 0.15, "Twofish": 0.05, "Camellia-256": 0.02, "ASCON-128": 0.00,
}
ATTACK_EFF = {
    "AES-128": 0.01, "AES-256": 0.01,
    "ChaCha20": 0.03,
    "Blowfish": 0.10, "Twofish": 0.04, "Camellia-256": 0.01, "ASCON-128": 0.00,
}
SUSPICIOUS = {
    "abuse", "arrest", "arson", "assault", "burglary", "explosion",
    "fighting", "robbery", "shooting", "shoplifting", "vandalism",
    "stealing", "accident",
}
WEIGHTS = dict(conf=0.30, privacy=0.25, financial=0.20,
               regulatory=0.15, strategic=0.10)
EXTS = {".mp4", ".avi", ".mkv", ".mov"}

# ───────────────────────── 4. Sécurité & sensibilité ────────────────────────
@cache
def key_strength(algo: str) -> float:
    bits = KEY_BITS[algo]
    return 1.0 if bits >= 256 else 0.85 if bits >= 192 else 0.70 if bits >= 128 else 0.40


def attack_resilience(algo: str) -> float:
    v, a = KNOWN_VULN[algo], ATTACK_EFF[algo]
    p = min(1.0, (CURRENT_YEAR - FIRST_YEAR[algo]) / 50.0)
    return 1.0 - (0.4 * v + 0.4 * a + 0.2 * p)


def age_penalty(algo: str) -> float:
    return min(1.0, (CURRENT_YEAR - FIRST_YEAR[algo]) / 50.0)


def content_adaptability(algo: str, mime: str) -> float:
    return CONTENT_ADAPT.get(mime.split("/")[0], {}).get(algo, 0.6)


def sec_score(algo: str, mime: str) -> float:
    return (OMEGA_KS * key_strength(algo)
            + OMEGA_AR * attack_resilience(algo)
            - OMEGA_AP * age_penalty(algo)
            + OMEGA_CA * content_adaptability(algo, mime))


def sens_factors(label: str) -> dict[str, int]:
    if label.lower() in SUSPICIOUS:
        return dict(conf=3, privacy=2, financial=1, regulatory=2, strategic=2)
    return dict(conf=1, privacy=0, financial=0, regulatory=0, strategic=0)


def sensitivity(label: str, name: str = "") -> float:
    """Return a per-file sensitivity score in [0,1]."""
    label = label.strip()
    length_ratio = min(len(label), MAX_LABEL_LEN) / MAX_LABEL_LEN
    base = 0.2 + 0.8 * length_ratio
    if name:
        h = int.from_bytes(hashlib.sha256(name.encode()).digest(), "big")
        base += (h % 1000) / 10000
    if label.lower() in SUSPICIOUS:
        base = min(1.0, base + 0.2)
    return round(min(1.0, base), 3)

# ───────────────────────── 5. Chiffrement & timing ──────────────────────────
CipherFn = Callable[[bytes], bytes]


def cipher_fn(algo: str) -> CipherFn:
    """Return a cipher function for ``algo`` using the pure Python fallback."""
    return get_cipher(algo)


def enc_time(path: str, fn: CipherFn) -> float:
    t0 = time.perf_counter()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            fn(chunk)
    return time.perf_counter() - t0

# ───────────────────────── 6. Score composite ───────────────────────────────
def score(algo: str, mime: str, label: str, path: str,
          α: float, β: float, dbg: bool = False) -> float:
    t = enc_time(path, cipher_fn(algo))
    s = α * (sec_score(algo, mime) * sensitivity(label)) - β * t
    if dbg:
        print(f"{algo:<22} Sec={sec_score(algo, mime):.3f} "
              f"Sens={sensitivity(label):.3f} Time={t:.4f}s → Score={s:.3f}")
    return s

# ───────────────────────── 7. Self-test ─────────────────────────────────────
def _selftest() -> None:
    present = [a for a in ALGORITHMS if a not in MISSING]
    assert {"AES-128", "AES-256", "ChaCha20"}.issubset(present), "AES/ChaCha20 absents"
    for a in present:
        sec = sec_score(a, "video/mp4")
        assert 0.0 <= sec <= 1.2, (a, sec)
    print("✅ Self-test : algorithmes disponibles =", ", ".join(present))

# ───────────────────────── 8. Dataset scan ──────────────────────────────────
def iter_videos(dataset: str) -> Iterable[str]:
    for root, _dirs, files in os.walk(dataset):
        for f in files:
            if os.path.splitext(f)[1].lower() in EXTS:
                yield os.path.join(root, f)




def compute(dataset: str, out: str, α: float, β: float) -> None:
    rows: list[list] = []
    counts = {a: 0 for a in ALGORITHMS if a not in MISSING}
    first = True

    for path in iter_videos(dataset):
        name = os.path.basename(path)
        label = os.path.basename(os.path.dirname(path)) or "unknown"
        mime = mimetypes.guess_type(path)[0] or "video/unknown"
        size = os.path.getsize(path)
        ext = os.path.splitext(name)[1].lower()
        sens = sensitivity(label, name)

        if first:
            print(f"\n🔎 Debug pour {name} — label {label}, MIME {mime}")

        best, best_val = None, -1e9
        for a in counts:
            sc = score(a, mime, label, path, α, β, dbg=first)
            if sc > best_val:
                best, best_val = a, sc
        first = False
        counts[best] += 1
        rows.append([name, size, "video", ext, f"{sens:.3f}", best])

    with open(out, "w", newline='', encoding="utf-8") as fw:
        csv.writer(fw).writerows([
            ["name", "size", "type", "extension", "sensitivity", "algorithm"],
            *rows,
        ])

    tot = sum(counts.values()) or 1
    print("\n📊 Distribution :")
    for a, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {a:<22}: {n:6} vidéos ({n/tot:5.1%})")
    print(f"✅ Résultats écrits → {out}")

# ───────────────────────── 9. CLI ───────────────────────────────────────────
def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Adaptive cipher selector")
    p.add_argument("--dataset", required=True, help="Racine DCSASS")
    p.add_argument("--out", default="dcsass_best.csv", help="CSV de sortie")
    p.add_argument("--alpha", type=float, default=ALPHA_DEF, help="poids α (sécurité)")
    p.add_argument("--beta",  type=float, default=BETA_DEF,  help="poids β (temps)")
    return p.parse_args()


def main() -> None:
    args = get_args()
    _selftest()
    compute(args.dataset, args.out, args.alpha, args.beta)


if __name__ == "__main__":
    main()
