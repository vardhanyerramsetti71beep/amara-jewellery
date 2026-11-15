"""Print a new brand name on the ring-box frame sequences.

usage: python rebrand_box.py SRC_DIR DST_DIR NAME FONT.ttf [open,drop,close]

SRC_DIR holds the ORIGINAL frames (open-000..060, drop-000..060, close-000..060 .webp),
where the old five-letter name is foil-stamped on the outside of the lid ("ext") and on
the satin lining inside it ("lin"). For every frame the script

  1. finds the old letters (five evenly spaced gold blobs near the middle of the box),
  2. paints them out with texture borrowed from the gaps between the letters,
  3. prints NAME in the same place, at the same size and tilt, in the foil's own colours.

Needs: opencv-python, numpy, Pillow.
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SRC, DST, NAME, FONT = sys.argv[1:5]
ONLY = set(sys.argv[5].split(",")) if len(sys.argv) > 5 else {"open", "drop", "close"}  # optional 5th argument
os.makedirs(DST, exist_ok=True)
N = 61               # frames per sequence
MARGIN = 150         # strip added left and right of the logo for medians and texture shifts
OLD_LETTERS = 5      # letters in the name being replaced
CENTRE_X = 797       # the logo is centred on the box, which is centred in the 1600 px frame
K3 = np.ones((3, 3), np.uint8)
LUMA = np.float32([0.114, 0.587, 0.299])  # for BGR pixels


def load(seq, i):
    return cv2.imread(f"{SRC}/{seq}-{i:03d}.webp")


def save(seq, i, im):
    rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(f"{DST}/{seq}-{i:03d}.webp", "WEBP", quality=92, method=5)


# ---------------------------------------------------------------- finding the old logo
def gold_mask(im, rim=False):
    h = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    H, S, V = h[..., 0], h[..., 1], h[..., 2]
    if rim:  # the polished rim is far more saturated than satin in shadow
        return ((H >= 8) & (H <= 32) & (S > 110) & (V > 120)).astype(np.uint8)
    return ((H >= 8) & (H <= 32) & (S > 70) & (V > 70)).astype(np.uint8)


def merge_blobs(comps):
    """Join components whose x-ranges touch: pieces of one letter."""
    out = []
    for x, y, w, h in sorted(comps):
        if out and x <= out[-1][2] + 2:
            b = out[-1]
            out[-1] = [b[0], min(b[1], y), max(b[2], x + w), max(b[3], y + h)]
        else:
            out.append([x, y, x + w, y + h])
    return out


def detect(im):
    """Box [x0, x1, y0, y1] of a row of ~five evenly spaced gold letters, or None."""
    n, _, st, _ = cv2.connectedComponentsWithStats(gold_mask(im), 8)
    cands = [tuple(int(v) for v in st[i][:4]) for i in range(1, n)
             if 2 <= st[i][3] <= 45 and 2 <= st[i][2] <= 45 and st[i][4] >= 4 and 600 < st[i][0] < 1000]
    best = None
    for c in cands:
        if c[3] < 3:
            continue
        cy = c[1] + c[3] / 2
        row = [d for d in cands if abs((d[1] + d[3] / 2) - cy) <= max(3, c[3] * 0.55)]
        bl = merge_blobs(row)
        if len(bl) < 4:
            continue
        hmax = np.percentile([b[3] - b[1] for b in bl], 80)
        bl = [b for b in bl if (b[3] - b[1]) >= 0.5 * hmax]
        if not 4 <= len(bl) <= 6:
            continue
        cx = np.array([(b[0] + b[2]) / 2 for b in bl])
        gaps = np.diff(cx)
        even = gaps.std() / gaps.mean()
        span, mid = cx[-1] - cx[0], (cx[0] + cx[-1]) / 2
        if even > 0.25 or not 140 < span < 330 or abs(mid - CENTRE_X) > 40:
            continue
        score = -even - abs(len(bl) - OLD_LETTERS) * 0.3 + hmax / 100
        if best is None or score > best[0]:
            best = (score, bl)
    if not best:
        return None
    bl = best[1]
    return [min(b[0] for b in bl), max(b[2] for b in bl), min(b[1] for b in bl), max(b[3] for b in bl)]


# ---------------------------------------------------------------- small numeric helpers
def hmedian(strip, k, mask=None):
    """Median along each row over a k-wide window; pixels under `mask` are ignored."""
    pad = k // 2
    if mask is None or not mask.any():
        p = np.pad(strip, ((0, 0), (pad, pad), (0, 0)), mode="reflect")
        return np.median(np.lib.stride_tricks.sliding_window_view(p, k, axis=1), axis=-1).astype(np.float32)
    a = strip.copy()
    a[mask] = np.nan
    p = np.pad(a, ((0, 0), (pad, pad), (0, 0)), mode="reflect")
    win = np.lib.stride_tricks.sliding_window_view(p, k, axis=1)
    # sort puts NaN last, so the median of the valid part sits at (count - 1) / 2
    srt = np.sort(win, axis=-1)
    cnt = (~np.isnan(win)).sum(axis=-1)
    lo = np.clip((cnt - 1) // 2, 0, k - 1)[..., None]
    hi = np.clip(cnt // 2, 0, k - 1)[..., None]
    med = (np.take_along_axis(srt, lo, -1) + np.take_along_axis(srt, hi, -1))[..., 0] / 2
    empty = cnt == 0
    if empty.any():
        med[empty] = strip[empty]
    return med.astype(np.float32)


def smooth(a, w=5):
    a = np.asarray(a, np.float64)
    if len(a) < w:
        return a
    pad = w // 2
    left = 2 * a[0] - a[pad:0:-1]
    right = 2 * a[-1] - a[-2:-pad - 2:-1]
    return np.convolve(np.concatenate([left, a, right]), np.ones(w) / w, mode="valid")


def fill_gaps(a):
    """Linear interpolation over NaN, with linear (not flat) extrapolation at the ends."""
    a = np.asarray(a, np.float64).copy()
    idx = np.arange(len(a))
    ok = ~np.isnan(a)
    good = idx[ok]
    if len(good) == 0:
        return a
    if len(good) == 1:
        a[:] = a[good[0]]
        return a
    a[~ok] = np.interp(idx[~ok], good, a[ok])
    j = good[:3]
    slope = (a[j[-1]] - a[j[0]]) / (j[-1] - j[0])
    for i in range(good[0]):
        a[i] = a[good[0]] + slope * (i - good[0])
    j = good[-3:]
    slope = (a[j[-1]] - a[j[0]]) / (j[-1] - j[0])
    for i in range(good[-1] + 1, len(a)):
        a[i] = a[good[-1]] + slope * (i - good[-1])
    return a


def drop_outliers(arr):
    """NaN-out detections whose height is off the local trend (the lid squashes the logo
    smoothly as it tilts, so a robust line through the neighbours predicts each frame)."""
    h = arr[:, 3] - arr[:, 2]
    bad = np.isnan(h)
    for k in range(len(h)):
        if bad[k]:
            continue
        js = [j for j in range(max(0, k - 4), min(len(h), k + 5)) if j != k and not np.isnan(h[j])]
        if len(js) < 3:
            continue
        slopes = [(h[b] - h[a]) / (b - a) for n, a in enumerate(js) for b in js[n + 1:]]
        s = np.median(slopes)
        pred = np.median([h[j] - s * j for j in js]) + s * k
        if abs(h[k] - pred) > max(5, 0.3 * abs(pred)):
            bad[k] = True
    arr[bad] = np.nan


def goldness(a):
    """How much warmer than neutral a BGR pixel is (R - B)."""
    return a[..., 2] - a[..., 0]


# ---------------------------------------------------------------- masks
def strip_of(im, box):
    x0, x1, y0, y1 = [int(round(v)) for v in box]
    sy0, sy1 = max(0, y0 - 12), min(im.shape[0], y1 + 12)
    sx0, sx1 = max(0, x0 - MARGIN), min(im.shape[1], x1 + MARGIN)
    return (sx0, sy0), (slice(sy0, sy1), slice(sx0, sx1))


def letter_mask(im, box, kind, exclude=None):
    """Strip origin, strip, surface estimate, raw letter mask and difference map.

    `exclude` marks pixels that are neither surface nor logo (the ring passing in front).
    """
    x0, x1, y0, y1 = [int(round(v)) for v in box]
    org, sl = strip_of(im, box)
    strip = im[sl].astype(np.float32)
    lim = np.zeros(strip.shape[:2], bool)
    lim[max(0, y0 - 7 - org[1]):max(0, y1 + 7 - org[1]), max(0, x0 - 12 - org[0]):x1 + 12 - org[0]] = True
    if exclude is not None:
        lim &= ~exclude
    thr = 34 if kind == "ext" else 18
    # a window two letter-pitches wide always holds the same share of letter pixels
    # (well under half), so the row median is the leather / satin underneath
    k = int(round((x1 - x0) / (OLD_LETTERS - 1) * 2)) | 1
    bg1 = hmedian(strip, k, exclude)
    m1 = (np.abs(strip - bg1).max(axis=2) > thr) & lim
    m1 = cv2.dilate(m1.astype(np.uint8), K3).astype(bool)
    bg = hmedian(strip, k, m1 if exclude is None else (m1 | exclude))
    diff = np.abs(strip - bg).max(axis=2)
    m = (diff > thr) & lim
    return org, strip, bg, m, diff


def core_pixels(strip, bg, diff, m, kind):
    """Pixels that are certainly foil: far from the surface colour and warmer than it."""
    warm = goldness(strip) - goldness(bg)
    return m & (diff > (60 if kind == "ext" else 40)) & (warm > (30 if kind == "ext" else 22))


def tight_box(strip, bg, diff, m, org, kind):
    core = core_pixels(strip, bg, diff, m, kind)
    ys = np.where(core.sum(axis=1) >= 4)[0]
    xs = np.where(core.sum(axis=0) >= 2)[0]
    if len(ys) < 3 or len(xs) < 20:
        return None
    return [org[0] + xs[0], org[0] + xs[-1] + 1, org[1] + ys[0], org[1] + ys[-1] + 1]


def rim_clip(im, box):
    """First row below a wide gold rim lying just above (or across) the box, else None."""
    x0, x1, y0, y1 = [int(round(v)) for v in box]
    ya, yb = max(0, y0 - 40), min(im.shape[0], y1 + 6)
    rows = np.where(gold_mask(im[ya:yb, x0:x1], rim=True).mean(axis=1) > 0.6)[0]
    return None if not len(rows) else ya + rows.max() + 1


def occluder(F, plate):
    """Pixels of strip `F` covered by something that is not in the clean `plate` (the ring).

    The shot is nominally static but the stamped letters wobble by a pixel or two, so the
    comparison takes the best match within +-2 px; what is left of the wobble is thin
    slivers, removed by an opening and a minimum size, while the ring is one large shape.
    """
    best = np.full(F.shape[:2], 1e9, np.float32)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            P = np.roll(np.roll(plate, dy, 0), dx, 1)
            # diamonds are about as bright as satin but neutral where satin is warm,
            # so compare colour temperature as well as the channels themselves
            d = np.maximum(np.abs(F - P).max(axis=2), np.abs(goldness(F) - goldness(P)) * 1.2)
            best = np.minimum(best, d)
    occ = cv2.morphologyEx((best > 24).astype(np.uint8), cv2.MORPH_OPEN, K3)
    n, lab, st, _ = cv2.connectedComponentsWithStats(occ, 8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= 70
    occ = keep[lab].astype(np.uint8)
    occ = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    # close small pockets inside the ring's head; the finger hole is large and open to the
    # strip's edge, so it is left alone and the logo shows through it
    n, lab, st, _ = cv2.connectedComponentsWithStats(1 - occ, 4)
    H, W = occ.shape
    for j in range(1, n):
        x, y, w, h, area = st[j]
        if area < 400 and x > 0 and y > 0 and x + w < W and y + h < H:
            occ[lab == j] = 1
    return cv2.dilate(occ, np.ones((5, 5), np.uint8)).astype(bool)


# ---------------------------------------------------------------- lettering
SS = 6  # supersampling
_probe = ImageFont.truetype(FONT, 400)
_capbox = _probe.getbbox("H")
_cap = _capbox[3] - _capbox[1]


def text_hires(W, hn, stroke):
    """Hi-res alpha of NAME whose ink spans W x hn target pixels, letters spread to fill W."""
    font = ImageFont.truetype(FONT, int(round(hn * SS * 400 / _cap)))
    top = font.getbbox("H")[1]
    boxes = [font.getbbox(ch) for ch in NAME]
    widths = [b[2] - b[0] for b in boxes]
    gap = (W * SS - sum(widths)) / (len(NAME) - 1)
    pad = 6 * SS
    cv = Image.new("L", (int(W * SS) + 2 * pad, int(hn * SS) + 2 * pad), 0)
    d = ImageDraw.Draw(cv)
    x = float(pad)
    for ch, b, w in zip(NAME, boxes, widths):
        d.text((x - b[0], pad - top), ch, font=font, fill=255, stroke_width=stroke, stroke_fill=255)
        x += w + gap
    return np.asarray(cv, np.float32) / 255.0, pad


def text_alpha(shape, org, box, r_nat, stroke):
    """Alpha (strip coords) of NAME filling `box`; squashed vertically when the lid is tilted."""
    x0, x1, y0, y1 = box
    W, h = x1 - x0, y1 - y0
    hn = r_nat * W
    hi, pad = text_hires(W, hn, stroke)
    sy = max(h / hn, 0.04)
    hi = cv2.GaussianBlur(hi, (0, 0), sigmaX=SS * 0.45, sigmaY=SS * 0.45 / min(sy, 1.0))
    M = np.float32([[1 / SS, 0, (x0 - org[0]) - pad / SS], [0, sy / SS, (y0 - org[1]) - pad * sy / SS]])
    return np.clip(cv2.warpAffine(hi, M, (shape[1], shape[0]), flags=cv2.INTER_LINEAR), 0, 1)


def letter_colours(strip, bg, diff, m, org, box, kind):
    """Face and edge colour (BGR) of the old foil: its bright half and its deep quarter."""
    y0, y1 = box[2], box[3]
    ys, xs = np.where(core_pixels(strip, bg, diff, m, kind))
    ok = (ys + org[1] >= y0 - 1) & (ys + org[1] < y1 + 1)
    px = strip[ys[ok], xs[ok]]
    if len(px) < 40:
        return np.full(3, np.nan), np.full(3, np.nan)
    order = np.argsort(px @ LUMA)
    n = len(px)
    return np.median(px[order[n // 2:]], axis=0), np.median(px[order[int(n * 0.08):int(n * 0.33)]], axis=0)


def paint_out(strip, bg, m, pitch, exclude=None):
    """Replace masked pixels with the surface estimate plus grain copied from half a
    letter-pitch away on the same row (the gap between two letters: same light, same tilt)."""
    md = cv2.dilate(m.astype(np.uint8), K3, iterations=2).astype(bool)
    unusable = md
    if exclude is not None:
        md &= ~exclude
        # never copy grain from on or right beside the thing in front
        unusable = md | cv2.dilate(exclude.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    out = strip.copy()
    todo = md.copy()
    grain = strip - bg
    half = pitch / 2
    for f in (1.0, 0.85, 1.15, 0.7, 1.3, 2.0):
        for o in (int(round(half * f)), -int(round(half * f))):
            if not todo.any():
                break
            bad_src = np.roll(unusable, o, axis=1)
            src = np.roll(grain, o, axis=1)
            if o > 0:
                bad_src[:, :o] = True
            else:
                bad_src[:, o:] = True
            take = todo & ~bad_src
            out[take] = bg[take] + src[take]
            todo &= ~take
    out[todo] = bg[todo]
    soft = np.clip(cv2.GaussianBlur(md.astype(np.float32), (0, 0), 0.9) * 1.6, 0, 1)[..., None]
    return strip * (1 - soft) + out * soft


def process(im, box, kind, r_nat, cols, clip_rim=False, exclude=None):
    """`im` with the logo inside `box` replaced by NAME."""
    face, edge = cols
    x0, x1, y0, y1 = box
    org, strip, bg, m, diff = letter_mask(im, box, kind, exclude)
    rows = np.arange(strip.shape[0]) + org[1]
    warm = goldness(strip) - goldness(bg)

    # keep only what hangs off certain-foil pixels, so grain, rim edges and folds stay untouched
    core = core_pixels(strip, bg, diff, m, kind)
    core[(rows < y0 - 2) | (rows > y1 + 2), :] = False
    m = m & cv2.dilate(core.astype(np.uint8), K3, iterations=3).astype(bool)
    if kind == "ext":
        # leather highlights are bright but neutral, and repainting them leaves a dark band:
        # keep foil (warm) and the letters' own shadows and undersides (darker than the leather)
        m &= (warm > 18) | ((strip @ LUMA) < (bg @ LUMA) - 10)

    show = np.ones(strip.shape[0], np.float32)
    if kind == "lin":
        # the lining logo slides under the lid's rim: it only exists on rows that are satin
        c0, c1 = int(x0 - org[0]), int(x1 - org[0])
        show = (np.median(bg[:, c0:c1] @ LUMA, axis=1) > 135).astype(np.float32)
        yc = rim_clip(im, box) if clip_rim else None
        if yc is not None:
            show *= np.clip((rows - yc) / 1.5, 0, 1)
        m[show < 0.5, :] = False
        if show.max() < 0.5:
            return im

    clean = paint_out(strip, bg, m, (x1 - x0) / (OLD_LETTERS - 1), exclude)

    outer = text_alpha(strip.shape, org, box, r_nat, 4 if kind == "ext" else 5)
    inner = np.minimum(text_alpha(strip.shape, org, box, r_nat, 0 if kind == "ext" else 1), outer)
    gate = show[:, None]
    if exclude is not None:
        gate = gate * (1 - np.clip(cv2.GaussianBlur(exclude.astype(np.float32), (0, 0), 0.8) * 1.5, 0, 1))
    outer, inner = outer * gate, inner * gate
    # edge-on, a letter is a few pixels tall: the dark bevel would swallow it, so fade to face colour
    squash = (y1 - y0) / (r_nat * (x1 - x0))
    w = float(np.clip((squash - 0.35) / 0.4, 0, 1))
    edge = face * (1 - w) + edge * w

    shadow = cv2.GaussianBlur(np.roll(np.roll(outer, 1, axis=0), 1, axis=1), (0, 0), 0.9)
    new = clean * (1 - shadow[..., None] * (0.42 if kind == "ext" else 0.25) * w)
    new = new * (1 - outer[..., None]) + edge[None, None, :] * outer[..., None]
    new = new * (1 - inner[..., None]) + face[None, None, :] * inner[..., None]
    out = im.copy()
    out[org[1]:org[1] + strip.shape[0], org[0]:org[0] + strip.shape[1]] = np.clip(new, 0, 255).astype(np.uint8)
    return out


# ---------------------------------------------------------------- tracking through a sequence
def build_track(seq, frames, kind, before=0, after=0):
    """Smoothed logo box and foil colours for each frame index in `frames`.

    `before` / `after` add that many frames by straight-line extrapolation: the frames where
    the lining logo is sliding under the rim and can no longer be detected as five letters.
    """
    ims = {i: load(seq, i) for i in frames}
    arr = np.full((len(frames), 4), np.nan)
    for k, i in enumerate(frames):
        r = detect(ims[i])
        if r:
            arr[k] = r
    drop_outliers(arr)
    arr = np.stack([fill_gaps(arr[:, c]) for c in range(4)], axis=1)
    for k, i in enumerate(frames):  # re-measure on the difference from the surface
        org, strip, bg, m, diff = letter_mask(ims[i], arr[k], kind)
        tb = tight_box(strip, bg, diff, m, org, kind)
        if tb:
            c_old, h_old = (arr[k, 2] + arr[k, 3]) / 2, arr[k, 3] - arr[k, 2]
            c_new, h_new = (tb[2] + tb[3]) / 2, tb[3] - tb[2]
            if abs(c_new - c_old) <= 6 and 0.6 * h_old - 2 <= h_new <= 1.5 * h_old + 3:
                arr[k] = tb
    c = smooth((arr[:, 2] + arr[:, 3]) / 2)
    hh = np.maximum(smooth(arr[:, 3] - arr[:, 2]), 2.0)
    xa, xb = smooth(arr[:, 0]), smooth(arr[:, 1])
    boxes = np.stack([xa, xb, c - hh / 2, c + hh / 2], axis=1)
    r_nat = float(np.percentile(hh / (xb - xa), 85))  # height / width when seen face-on
    cols = np.full((len(frames), 2, 3), np.nan)
    for k, i in enumerate(frames):
        org, strip, bg, m, diff = letter_mask(ims[i], boxes[k], kind)
        cols[k] = letter_colours(strip, bg, diff, m, org, boxes[k], kind)
    for j in range(2):
        for ch in range(3):
            cols[:, j, ch] = smooth(fill_gaps(cols[:, j, ch]), 5)
    out = {i: (boxes[k], cols[k]) for k, i in enumerate(frames)}
    if before:
        v = (boxes[min(3, len(frames) - 1)] - boxes[0]) / min(3, len(frames) - 1)
        for n in range(1, before + 1):
            out[frames[0] - n] = (boxes[0] - v * n, cols[0])
    if after:
        a = max(0, len(frames) - 4)
        v = (boxes[-1] - boxes[a]) / (len(frames) - 1 - a)
        for n in range(1, after + 1):
            out[frames[-1] + n] = (boxes[-1] + v * n, cols[-1])
    hs = boxes[:, 3] - boxes[:, 2]
    print(f"  {seq} {kind}: frames {frames[0]}-{frames[-1]}, letter height {hs.min():.1f}-{hs.max():.1f} px, "
          f"width {(boxes[:, 1] - boxes[:, 0]).mean():.0f} px")
    return out, r_nat


def run_sequence(seq, lid, lining):
    """lid / lining: (track, r_nat) or None."""
    for i in range(N):
        im = load(seq, i)
        if lid and i in lid[0]:
            im = process(im, lid[0][i][0], "ext", lid[1], lid[0][i][1])
        if lining and i in lining[0]:
            im = process(im, lining[0][i][0], "lin", lining[1], lining[0][i][1], clip_rim=True)
        save(seq, i, im)


print(f"printing {NAME} on the box frames")

if "open" in ONLY:
    # lid seen from outside while it tilts back (0-32), then the lining comes out from under the rim
    run_sequence("open", build_track("open", list(range(0, 33)), "ext"),
                 build_track("open", list(range(36, 61)), "lin", before=3))

if "drop" in ONLY:
    # the box stands still and the ring falls past the lining logo (about frames 16-40)
    clean_frames = list(range(0, 15)) + list(range(44, 61))
    trk, r_d = build_track("drop", clean_frames, "lin")
    box = np.median(np.array([trk[i][0] for i in clean_frames]), axis=0)
    cols = np.median(np.array([trk[i][1] for i in clean_frames]), axis=0)
    _, sl = strip_of(load("drop", 0), box)
    strips = np.stack([load("drop", i)[sl].astype(np.float32) for i in range(N)])
    for i in range(N):
        occ = None
        if 10 < i < 48:
            # the ring covers any one pixel for only a few frames, so the median over the
            # neighbouring frames is the empty box as it looks around this moment
            plate = np.median(strips[max(0, i - 15):i + 16], axis=0)
            occ = occluder(strips[i], plate)
            if not occ.any():
                occ = None
        save("drop", i, process(load("drop", i), box, "lin", r_d, cols, exclude=occ))

if "close" in ONLY:
    # the lining goes back under the rim (0-20), then the outside of the lid tilts into view
    run_sequence("close", build_track("close", list(range(22, 61)), "ext"),
                 build_track("close", list(range(0, 19)), "lin", after=2))

print(f"done: {len([f for f in os.listdir(DST) if f.endswith('.webp')])} frames in {DST}")
