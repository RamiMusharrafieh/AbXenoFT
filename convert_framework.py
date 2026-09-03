#!/usr/bin/env python3
"""
Convert an antibody heavy-chain framework toward mouse or macaque.

Given the four framework regions of a VH domain (from any species, human for
instance), this asks the target species' model what residue belongs at each
framework position, and substitutes where that model is confident the input
residue is wrong.

The species switch selects a *model*, not a token. A single joint model with a
species marker was built and rejected: with 15% masking the visible residues
already identify the species, so the marker carried no information, received no
gradient, and was ignored: converting to mouse and to macaque produced identical
output. A model trained on one species has no such conflict. Never having seen the
other species, it can only answer in the one it knows, which is exactly the
behaviour conversion needs.

Each position is scored by masking it while leaving the rest of the sequence
intact, so the model conditions on the actual antibody rather than generating a
generic germline. Positions the target species is happy with are left alone; only
positions where it strongly prefers something else are changed. CDRs are not part
of the model's input and are never touched.

    python convert_framework.py --to mouse \\
        --fr1 QVQLVQSGAEVKKPGASVKVSCKAS --fr2 MHWVRQAPGQGLEWMGR \\
        --fr3 KYNEKFKSRVTLTVDKSTSTAYMELSSLRSEDTAVYYC --fr4 WGQGTLVTVSS

    python convert_framework.py --to macaque --seq <91-residue FR1+FR2+FR3+FR4>

The four regions are expected at their canonical lengths (25/17/38/11 = 91), which
is what ANARCI/IMGT numbering yields for the OAS data this model was trained on.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

PROJECT = Path(__file__).resolve().parent
CKPT = PROJECT / "checkpoints"
MODELS = {
    "mouse": CKPT / "esm2_t6_8M_mouse_fwr_vgene" / "final",
    "macaque": CKPT / "esm2_t6_8M_macaque_fwr_vgene" / "final",
}
REGIONS = [("FR1", 0, 25), ("FR2", 25, 42), ("FR3", 42, 80), ("FR4", 80, 91)]
CANON = 91
AAS = set("ACDEFGHIKLMNPQRSTVWY")


def region_of(pos: int) -> str:
    for name, a, b in REGIONS:
        if a <= pos < b:
            return name
    return "?"


class FrameworkConverter:
    def __init__(self, models: dict[str, Path] | None = None, device: str | None = None):
        self.paths = {k: Path(v) for k, v in (models or MODELS).items()}
        missing = {k: p for k, p in self.paths.items() if not p.exists()}
        if len(missing) == len(self.paths):
            raise SystemExit(f"no species models found: {list(self.paths.values())}")
        self.available = [k for k in self.paths if k not in missing]
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self._models: dict[str, object] = {}
        self.tok = AutoTokenizer.from_pretrained(str(self.paths[self.available[0]]))
        self.aa_ids = {a: self.tok.convert_tokens_to_ids(a) for a in sorted(AAS)}

    def _model(self, species: str):
        if species not in self.available:
            raise SystemExit(f"model for {species!r} not trained yet ({self.paths[species]})")
        if species not in self._models:
            m = AutoModelForMaskedLM.from_pretrained(str(self.paths[species])).eval()
            self._models[species] = m.to(self.device)
        return self._models[species]

    @torch.no_grad()
    def position_probs(self, seq: str, species: str) -> torch.Tensor:
        """P(residue | rest of sequence) under the species' own model. -> (L, 20)"""
        model = self._model(species)
        enc = self.tok(seq, return_tensors="pt")
        ids = enc["input_ids"].repeat(len(seq), 1)
        rows = torch.arange(len(seq))
        ids[rows, rows + 1] = self.tok.mask_token_id   # +1 skips <cls>
        attn = enc["attention_mask"].repeat(len(seq), 1)

        out = []
        for s in range(0, len(seq), 128):
            logits = model(input_ids=ids[s:s + 128].to(self.device),
                           attention_mask=attn[s:s + 128].to(self.device)).logits
            r = torch.arange(logits.shape[0])
            out.append(logits[r, rows[s:s + 128] + 1].float().cpu())
        logits = torch.cat(out)
        cols = torch.tensor([self.aa_ids[a] for a in sorted(AAS)])
        return torch.softmax(logits, dim=-1)[:, cols]

    def score(self, seq: str, species: str) -> float:
        """Mean log-probability of the actual residues under a species condition."""
        probs = self.position_probs(seq, species)
        order = sorted(AAS)
        idx = torch.tensor([order.index(c) for c in seq])
        return float(torch.log(probs[torch.arange(len(seq)), idx] + 1e-9).mean())

    def convert(self, seq: str, target: str, min_confidence: float = 0.5,
                rounds: int = 2) -> dict:
        order = sorted(AAS)
        current = list(seq)
        subs = []
        for _ in range(rounds):
            probs = self.position_probs("".join(current), target)
            changed = False
            for p in range(len(current)):
                top_i = int(probs[p].argmax())
                top_aa, top_p = order[top_i], float(probs[p, top_i])
                cur_aa = current[p]
                cur_p = float(probs[p, order.index(cur_aa)])
                if top_aa != cur_aa and top_p >= min_confidence:
                    subs.append({"pos": p + 1, "region": region_of(p), "frm": cur_aa,
                                 "to": top_aa, "p_from": cur_p, "p_to": top_p})
                    current[p] = top_aa
                    changed = True
            if not changed:
                break
        # collapse repeated edits at the same position, keeping the final one
        final = {}
        for s in subs:
            final[s["pos"]] = s
        return {"converted": "".join(current),
                "substitutions": sorted(final.values(), key=lambda s: s["pos"])}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--to", dest="target", choices=sorted(MODELS), required=True)
    p.add_argument("--seq", help="Concatenated FR1+FR2+FR3+FR4 (91 residues).")
    for r, _, _ in REGIONS:
        p.add_argument(f"--{r.lower()}")
    p.add_argument("--mouse-model", type=Path, default=MODELS["mouse"])
    p.add_argument("--macaque-model", type=Path, default=MODELS["macaque"])
    p.add_argument("--min-confidence", type=float, default=0.5,
                   help="Only substitute when the target species prefers a residue "
                        "with at least this probability.")
    p.add_argument("--rounds", type=int, default=2,
                   help="Re-score after applying edits so substitutions can see each other.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.seq:
        seq = args.seq.strip().upper()
    else:
        parts = [getattr(args, r.lower()) for r, _, _ in REGIONS]
        if any(x is None for x in parts):
            raise SystemExit("provide --seq, or all four of --fr1 --fr2 --fr3 --fr4")
        for (name, a, b), val in zip(REGIONS, parts):
            if len(val.strip()) != b - a:
                raise SystemExit(f"{name} must be {b - a} residues, got {len(val.strip())}")
        seq = "".join(x.strip().upper() for x in parts)

    if len(seq) != CANON:
        raise SystemExit(f"framework must be {CANON} residues (25/17/38/11), got {len(seq)}")
    bad = set(seq) - AAS
    if bad:
        raise SystemExit(f"unsupported residues: {''.join(sorted(bad))}")

    conv = FrameworkConverter({"mouse": args.mouse_model, "macaque": args.macaque_model})
    others = [s for s in conv.available if s != args.target]

    before = {sp: conv.score(seq, sp) for sp in conv.available}
    res = conv.convert(seq, args.target, args.min_confidence, args.rounds)
    after = {sp: conv.score(res["converted"], sp) for sp in conv.available}

    print(f"\ntarget species : {args.target}")
    print(f"substitutions  : {len(res['substitutions'])} of {CANON} positions "
          f"({100 * len(res['substitutions']) / CANON:.1f}%)\n")

    print("  region  pos  from -> to     p(from)   p(to)")
    print("  " + "-" * 46)
    for s in res["substitutions"]:
        print(f"  {s['region']:5s} {s['pos']:4d}   {s['frm']}  -> {s['to']}      "
              f"{s['p_from']:6.3f}  {s['p_to']:6.3f}")
    if not res["substitutions"]:
        print("  (none, the input already reads as this species)")

    cols = [args.target] + others
    print("\nmean log-likelihood      " + "".join(f"{c:>10s}" for c in cols))
    print("  before                 " + "".join(f"{before[c]:10.4f}" for c in cols))
    print("  after                  " + "".join(f"{after[c]:10.4f}" for c in cols))
    if others:
        moved = (after[args.target] - before[args.target]) - (after[others[0]] - before[others[0]])
        print(f"  net shift toward {args.target}: {moved:+.4f}")

    print("\noriginal :", seq)
    print("converted:", res["converted"])
    print("changed  :", "".join("^" if a != b else " "
                                for a, b in zip(seq, res["converted"])))
    for name, a, b in REGIONS:
        print(f"  {name}: {res['converted'][a:b]}")


if __name__ == "__main__":
    main()
