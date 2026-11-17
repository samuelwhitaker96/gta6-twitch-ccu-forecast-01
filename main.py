"""Forecast GTA VI's launch-day Twitch viewers hour by hour (make.fun
gta6-twitch-ccu-forecast-01).

Model
-----
The program reads one case file (schema "makefun/twitch-ccu-case/v1"): hour 1's
print of the release day's Twitch category viewers, plus the hourly counts of
past big game launches ("comparables"). It forecasts hours 2-24 as five
quantiles (0.05, 0.25, 0.5, 0.75, 0.95) of each hour's viewer count.

The forecast is anchored on hour 1's print p1 and built from the comparables'
release-day shapes, expressed as log-ratios to their own hour-1 print,
log(v[h] / v[1]). Two pools are blended in log space:

  (A) every comparable (except the target's own pre-release history, tier
      "target-history"): weighted by tier (comparable 1.0, early-access
      contaminated 0.25, weak 0.15, unknown 0.5), by unlock type (a launch
      that did not unlock at midnight local gets 0.35), and by scale -- a
      Gaussian kernel on log hour-1 viewers (sigma 0.8), so launches of a
      similar size to this one count most. Big launches ramp less steeply
      from hour 1 than small ones, and the kernel captures that.
  (B) the structural analogues: rolling midnight-local, console-only
      launches (the shape GTA VI itself has: PS5/Xbox Series, midnight local
      rolling, no PC at launch), with a wider scale kernel (sigma 1.5).
      This pool keeps the forecast ramping into the evening even when hour 1
      is enormous, where pool (A) alone would be dominated by PC
      front-loaded launches whose hour 1 already holds most of the world.

Per hour, each pool gives a weighted median and a spread (its interquartile
range / 1.349). The blended median is the mean of the two pools' medians; the
blended spread is the larger of the two pools' spreads and half the distance
between their medians (the two views disagreeing widens the interval), with a
floor that grows like sqrt(h - 1). Quantiles are median + z_tau * spread in
log space, times p1.

Why this works: launch-day Twitch viewership follows the unlock geography.
Hour 1 (00:00-01:00 UTC) holds the UK/Europe/Oceania midnight audience; the
US unlocks at 05:00/08:00 UTC (hours 6 and 9) and the evening peak lands in
hours 17-22, when the US afternoon overlaps the European evening. The
comparables' median log-ratio profile reproduces that shape, and the spread
across launches is the honest uncertainty. Backtested on the kit's four
practice launches (Wolverine, Red Dead Redemption 2, God of War Ragnarok,
Spider-Man 2) this scores a mean pinball loss of about 0.20 on the log scale,
against about 0.39 for the house baseline (flat at hour 1's print).

The program is deterministic (no randomness, no time, no network), standard
library only, and runs in well under a second on one CPU.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level (exact constants).
Z_STD = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817,
         1.6448536269514722]

# --- model constants (tuned on the four practice launches) -----------------
TIER_WEIGHT = {"comparable": 1.0, "early-access-contaminated": 0.25,
               "weak": 0.15, "target-history": 0.0}
TIER_UNKNOWN_WEIGHT = 0.5
NON_ROLLING_WEIGHT = 0.35      # pool A: launches that did not unlock at midnight local
SCALE_SIGMA_ALL = 0.8          # pool A: Gaussian kernel on log hour-1 viewers
SCALE_SIGMA_ROLLING = 1.5      # pool B: wider kernel (small pool)
BLEND_ROLLING = 0.5            # weight of pool B in the blended median
SPREAD_K = 1.0                 # multiplier on the pooled spread
SPREAD_FLOOR = 0.05            # per sqrt(hour - 1), in log units
SIGMA_MAX = 1.0                # cap on the log-spread: guards the degenerate
                               # regime where a tiny hour-1 print matches only
                               # simultaneous launches whose ratios span 1000x;
                               # never binds at realistic hour-1 levels
                               # (there sigma is about 0.4-0.6).
MAX_VIEWERS = 1_000_000_000.0


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def z_scores(levels) -> list:
    """Standard-normal quantiles for the case's levels."""
    out = []
    for lv in levels:
        if lv in LEVELS:
            out.append(Z_STD[LEVELS.index(lv)])
        else:
            out.append(norm_ppf(lv))
    return out


def round2(x: float):
    """x rounded to 2 decimals, half up (as the starter rounds it)."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value == int(value) else value


def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def tier_of(comp: dict) -> str:
    m = re.search(r"Tier:\s*([a-z\-]+)", comp.get("unlock", ""))
    return m.group(1) if m else "unknown"


def is_rolling(comp: dict) -> bool:
    """Did this launch unlock at midnight local time, rolling by region?"""
    head = comp.get("unlock", "").lower().split("tier:")[0]
    if "single global" in head or "single simultaneous global" in head:
        return False
    if "midnight local" in head or "rolling" in head:
        return True
    if "unverified" in head:
        # Console-only Sony/Nintendo launches unlock at midnight local by norm.
        return "PC" not in comp.get("platforms", "")
    return False


def is_console_only(comp: dict) -> bool:
    return "PC" not in comp.get("platforms", "")


def day_profile(comp: dict):
    """The comparable's release-day hourly viewers, hours 1..24 (log-interpolated
    over gaps). None when hour 1 is missing or not positive."""
    ds = parse_ts(comp["day_start"])
    prof = {}
    for row in comp.get("hourly", []):
        try:
            t = parse_ts(row["from"])
            v = float(row["viewers"])
        except (KeyError, TypeError, ValueError):
            continue
        h = int((t - ds).total_seconds() // 3600) + 1
        if 1 <= h <= 24 and v > 0:
            prof[h] = v
    if 1 not in prof:
        return None
    have = sorted(prof)
    out = {}
    for h in range(1, 25):
        if h in prof:
            out[h] = prof[h]
        else:
            lo = max([k for k in have if k < h], default=None)
            hi = min([k for k in have if k > h], default=None)
            if lo is not None and hi is not None:
                lv, hv = math.log(prof[lo]), math.log(prof[hi])
                out[h] = math.exp(lv + (hv - lv) * (h - lo) / (hi - lo))
            elif lo is not None:
                out[h] = prof[lo]
            elif hi is not None:
                out[h] = prof[hi]
            else:
                return None
    return out


def weighted_quantile(pairs, tau: float) -> float:
    """Weighted quantile of [(value, weight)] with linear interpolation."""
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    if total <= 0:
        return pairs[len(pairs) // 2][0]
    pts = []
    cum = 0.0
    for v, w in pairs:
        cum += w
        pts.append((cum - w / 2.0, v))
    target = tau * total
    if target <= pts[0][0]:
        return pts[0][1]
    if target >= pts[-1][0]:
        return pts[-1][1]
    for i in range(1, len(pts)):
        if pts[i][0] >= target:
            x0, y0 = pts[i - 1]
            x1, y1 = pts[i]
            f = (target - x0) / (x1 - x0) if x1 > x0 else 0.0
            return y0 + f * (y1 - y0)
    return pts[-1][1]


def pool_stats(p1: float, comparables, rolling_console_only: bool):
    """Per hour, (weighted median, IQR-based sigma) of the log-ratios
    log(v[h] / v[1]) over one pool of comparables. Returns (stats, pool size)."""
    sigma_scale = SCALE_SIGMA_ROLLING if rolling_console_only else SCALE_SIGMA_ALL
    profs = []
    for comp in comparables:
        tier = tier_of(comp)
        if tier == "target-history":
            continue
        w = TIER_WEIGHT.get(tier, TIER_UNKNOWN_WEIGHT)
        if w <= 0:
            continue
        if rolling_console_only:
            if not (is_rolling(comp) and is_console_only(comp)):
                continue
        elif not is_rolling(comp):
            w *= NON_ROLLING_WEIGHT
        prof = day_profile(comp)
        if prof is None:
            continue
        w *= math.exp(-0.5 * ((math.log(prof[1]) - math.log(max(p1, 1.0)))
                               / sigma_scale) ** 2)
        if w <= 0:
            continue
        profs.append((prof, w))
    stats = {}
    for h in range(2, 25):
        pairs = [(math.log(prof[h] / prof[1]), w) for prof, w in profs]
        if not pairs:
            stats[h] = (0.0, 0.30)
            continue
        med = weighted_quantile(pairs, 0.5)
        q25 = weighted_quantile(pairs, 0.25)
        q75 = weighted_quantile(pairs, 0.75)
        stats[h] = (med, max((q75 - q25) / 1.349, 0.0))
    return stats, len(profs)


def baseline_quantiles(p1: float, h: int, z):
    """The house baseline: flat at hour 1's print, spread widening with horizon."""
    return [round2(p1 * math.exp(zi * 0.15 * math.sqrt(h - 1))) for zi in z]


def forecast_case(case: dict, name: str) -> dict:
    levels = case.get("levels") or LEVELS
    z = z_scores(levels)
    first = next((k for k in case.get("known", []) if k.get("hour") == 1), None)
    if first is not None and first.get("viewers"):
        p1 = max(1.0, float(first["viewers"]))
    else:
        hist = case.get("history") or []
        p1 = max(1.0, float(hist[-1]["viewers"])) if hist else 1.0
    comparables = case.get("comparables") or []
    stats_all, n_all = pool_stats(p1, comparables, rolling_console_only=False)
    stats_rc, n_rc = pool_stats(p1, comparables, rolling_console_only=True)
    hours = []
    for row in case["forecast"]:
        h = int(row["hour"])
        if n_all == 0 and n_rc == 0:
            qs = baseline_quantiles(p1, h, z)
        else:
            med_a, sig_a = stats_all[h]
            med_b, sig_b = stats_rc[h]
            if n_all == 0:
                med, sig = med_b, sig_b
            elif n_rc == 0:
                med, sig = med_a, sig_a
            else:
                med = (1 - BLEND_ROLLING) * med_a + BLEND_ROLLING * med_b
                sig = max(sig_a, sig_b, 0.5 * abs(med_a - med_b))
            sig = max(sig * SPREAD_K, SPREAD_FLOOR * math.sqrt(h - 1))
            sig = min(sig, SIGMA_MAX)
            qs = []
            prev = 0.0
            for zi in z:
                q = p1 * math.exp(med + zi * sig)
                q = min(max(q, 0.01), MAX_VIEWERS)
                q = round2(q)
                q = max(q, prev)  # keep the row non-decreasing after rounding
                prev = q
                qs.append(q)
        hours.append({"hour": h, "q": qs})
    return {"hours": hours}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
