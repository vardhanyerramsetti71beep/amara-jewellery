"""Make the hero film: the ring turning in the dark.

usage: python make_hero_film.py TURN_DIR OUT.mp4 [--preview SHEET.jpg | --frame N | --poster POSTER.jpg]

The source repo never shipped its hero film. What it does ship is real footage of the ring
turning: site/assets/turn/h2v-000..060.webp, the 61 frames the page plays when the ring
stands up, shot on a white studio background. This script

  1. cuts the ring out of every frame (the white studio and its soft floor shadow go),
  2. sets it on black with a faint warm glow behind it,
  3. plays the turn forwards and back as one seamless loop, the camera pulling back as the
     ring stands up so it always fills the frame,
  4. adds what a jewellery film has and a white-studio turn lacks: a band of light crossing
     the stones, facets that shimmer, and glints that flash on the brightest points.

--poster writes the film's first frame as a JPEG, for the <video poster>.
Needs: opencv-python, numpy, imageio-ffmpeg.
"""
import math
import sys

import cv2
import numpy as np

SRC, OUT = sys.argv[1], sys.argv[2]
MODE = sys.argv[3] if len(sys.argv) > 3 else None
ARG = sys.argv[4] if len(sys.argv) > 4 else None

OW, OH, FPS, SEC = 1920, 1080, 30, 14
NF = FPS * SEC
NSRC = 61
MAX_SCALE = 1.3        # how far the camera may close in on the ring lying flat
FILL_W, FILL_H = 0.74, 0.84
LUMA = np.float32([0.114, 0.587, 0.299])
K3 = np.ones((3, 3), np.uint8)
rng = np.random.default_rng(11)


# ---------------------------------------------------------------- cutting the ring out
def cut_out(im):
    """Ring on black (float32 BGR 0..1) and its mask, from a frame shot on white.

    Background is whatever can be reached from the frame's edge through pixels that are
    colourless and smooth: the white studio and the soft shadow on its floor. The ring's
    outline is a hard edge, so the walk stops there even where a stone is as bright as the
    backdrop. Studio seen through the ring (the finger hole, the gaps between head and band)
    is cut off from the edge; it is recognised as flat white that is ringed by gold.
    """
    f = im.astype(np.float32)
    mn, mx = f.min(axis=2), f.max(axis=2)
    sat = mx - mn
    gray = f @ LUMA
    grad = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)) / 8
    mean = cv2.blur(gray, (9, 9))
    std = np.sqrt(np.maximum(cv2.blur(gray * gray, (9, 9)) - mean * mean, 0))
    light = (mn >= 205) & (sat <= 14) & (grad <= 2.2)
    shadow = (mn >= 110) & (sat <= 16) & (grad <= 5.0) & (std <= 7.0)   # stones are colourless too, but never smooth
    n, lab, st, _ = cv2.connectedComponentsWithStats((light | shadow).astype(np.uint8), 4)
    bg = np.zeros(n, bool)
    bg[np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))] = True
    whites = np.bincount(lab[mn >= 253], minlength=n)
    gold = sat > 30
    for j in np.where((whites >= 150) & ~bg)[0]:
        if j == 0:
            continue
        x, y, w, h = st[j, :4]
        x0, y0, x1, y1 = max(0, x - 4), max(0, y - 4), min(lab.shape[1], x + w + 4), min(lab.shape[0], y + h + 4)
        m = (lab[y0:y1, x0:x1] == j).astype(np.uint8)
        rim = (cv2.dilate(m, np.ones((7, 7), np.uint8)) > 0) & (m == 0)
        # stones' own white facets are ringed by more stone (colourless); studio seen through the ring is ringed by gold
        if whites[j] >= 6000 or (rim.any() and gold[y0:y1, x0:x1][rim].mean() > 0.30):
            bg[j] = True
    bg[0] = False
    ring = cv2.morphologyEx((~bg[lab]).astype(np.uint8), cv2.MORPH_OPEN, K3)
    # the ring is one object: keep the largest piece, drop islands (the warm glow it casts on the floor)
    n2, lab2, st2, _ = cv2.connectedComponentsWithStats(ring, 8)
    ring = (lab2 == 1 + int(np.argmax(st2[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    # the floor shadow can touch the band: below the lowest gold in each column, anything
    # colourless is shadow, not ring (the band's own underside is gold)
    goldm = (sat > 30) & (ring > 0)
    has = goldm.any(axis=0)
    low = np.where(has, goldm.shape[0] - 1 - np.argmax(goldm[::-1], axis=0), -1)
    rows = np.arange(ring.shape[0])[:, None]
    ring[(rows > low[None, :] + 2) & (sat <= 22)] = 0
    n3, lab3, st3, _ = cv2.connectedComponentsWithStats(ring, 8)
    if n3 > 1:
        ring = (lab3 == 1 + int(np.argmax(st3[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    # pull the matte in a pixel and soften it, then take the white backdrop back out of the
    # edge pixels; erring inward gives a slightly dark rim, never a white halo on black
    alpha = cv2.GaussianBlur(cv2.erode(ring, K3).astype(np.float32), (0, 0), 0.8)
    out = np.clip(f - (1 - alpha[..., None]) * 255.0, 0, 255) / 255.0
    return out * (alpha[..., None] > 0.02), ring.astype(bool)


print("cutting the ring out of", NSRC, "frames")
RING, BOX, POINTS = [], [], []
for i in range(NSRC):
    img, mask = cut_out(cv2.imread(f"{SRC}/h2v-{i:03d}.webp"))
    RING.append((img * 255).astype(np.uint8))
    ys, xs = np.where(mask)
    BOX.append([xs.min(), xs.max(), ys.min(), ys.max()])
    # glint anchors: the brightest small highlights of this frame
    lum = cv2.GaussianBlur(img @ LUMA, (0, 0), 1.6)
    peak = (lum == cv2.dilate(lum, np.ones((41, 41), np.uint8))) & (lum > 0.86) & mask
    py, px = np.where(peak)
    order = np.argsort(-lum[py, px])[:50]
    POINTS.append(np.stack([px[order], py[order]], axis=1).astype(np.float32))
BOX = np.array(BOX, np.float64)


def smooth(a, w=7):
    pad = w // 2
    p = np.concatenate([np.repeat(a[:1], pad), a, np.repeat(a[-1:], pad)])
    return np.convolve(p, np.ones(w) / w, mode="valid")


CX, CY = smooth((BOX[:, 0] + BOX[:, 1]) / 2), smooth((BOX[:, 2] + BOX[:, 3]) / 2)
SCALE = smooth(np.minimum(MAX_SCALE, np.minimum(FILL_W * OW / (BOX[:, 1] - BOX[:, 0]), FILL_H * OH / (BOX[:, 3] - BOX[:, 2]))))


def at(arr, f):
    f0 = int(math.floor(f))
    f1 = min(NSRC - 1, f0 + 1)
    return arr[f0] * (1 - (f - f0)) + arr[f1] * (f - f0)


def turn(p):
    """Source frame (float) for loop phase p: 0 -> 60 -> 0, easing at both ends."""
    return (1 - math.cos(2 * math.pi * p)) / 2 * (NSRC - 1)


def camera(f):
    """(scale, tx, ty): source px -> output px."""
    s = at(SCALE, f)
    return s, OW / 2 - at(CX, f) * s, OH * 0.5 - at(CY, f) * s


# ---------------------------------------------------------------- light
X, Y = np.meshgrid(np.arange(OW, dtype=np.float32), np.arange(OH, dtype=np.float32))
GLOW = np.exp(-(((X - OW / 2) / (0.42 * OW)) ** 2 + ((Y - OH / 2) / (0.55 * OH)) ** 2))[..., None] * np.float32([0.035, 0.055, 0.075])


def tile_noise(size=256, sigma=7.0):
    # blur a 3x3 tiling and keep the middle tile, so the result wraps without a seam
    n = cv2.GaussianBlur(np.tile(rng.random((size, size)).astype(np.float32), (3, 3)), (0, 0), sigma)
    n = n[size:2 * size, size:2 * size]
    return np.clip((n - n.mean()) / (n.std() * 3.0) + 0.5, 0, 1)


N1, N2 = tile_noise(), tile_noise()
NS = 4.0  # output px per noise px
TH = math.radians(18)
U = (X * math.cos(TH) + Y * math.sin(TH)) / OW
BAND_W = 0.10


def band_centre(p, start):
    """A light band that crosses the frame during 30% of the loop, beginning at phase `start`."""
    q = min(max(((p - start) % 1.0) / 0.30, 0.0), 1.0)
    q = q * q * (3 - 2 * q)
    return -0.4 + q * 1.9


def star(size, length):
    """A glint: hot core, small glow, long cross and short diagonal streaks."""
    c = size // 2
    y, x = np.mgrid[-c:c + 1, -c:c + 1].astype(np.float32)
    r = np.hypot(x, y)

    def streak(a, b, ln, wd):
        along, across = x * a + y * b, -x * b + y * a
        return np.exp(-(across / wd) ** 2) * np.exp(-np.abs(along) / ln)

    d = 0.7071
    s = streak(1, 0, length, 1.05) + streak(0, 1, length, 1.05)
    s += 0.3 * (streak(d, d, length * 0.4, 0.9) + streak(d, -d, length * 0.4, 0.9))
    return np.exp(-(r / 2.6) ** 2) + 0.3 * np.exp(-(r / (length * 0.35)) ** 2) + 0.8 * s


SPRITES = [star(2 * int(L * 5) + 1, L) for L in (10, 16, 24, 34)]

# glints: each flashes once, anchored to a highlight of the frame showing at its peak and
# then re-anchored frame by frame to the nearest highlight, so it rides the stone as it turns
GLINTS = []
for _ in range(int(SEC * 4.2)):
    pk = int(rng.integers(0, NF))
    half = int(rng.integers(6, 13))
    pts = POINTS[int(round(turn(pk / NF)))]
    if not len(pts):
        continue
    pos = {pk: pts[int(rng.integers(0, len(pts)))].copy()}
    for step in (1, -1):
        cur = pos[pk]
        for d in range(1, half + 1):
            i = (pk + step * d) % NF
            cand = POINTS[int(round(turn(i / NF)))]
            if len(cand):
                dist = np.hypot(cand[:, 0] - cur[0], cand[:, 1] - cur[1])
                j = int(dist.argmin())
                if dist[j] < 26:
                    cur = cand[j].copy()
            pos[i] = cur
    GLINTS.append(dict(peak=pk, half=half, pos=pos, gain=float(rng.uniform(0.45, 1.0)),
                       sprite=int(rng.choice(4, p=[0.3, 0.35, 0.25, 0.1])),
                       tint=np.float32([1, 1, 1]) + rng.uniform(-0.15, 0.15, 3).astype(np.float32)))


def frame(i):
    p = i / NF
    f = turn(p)
    f0 = int(math.floor(f))
    f1 = min(NSRC - 1, f0 + 1)
    w = f - f0
    src = cv2.addWeighted(RING[f0], 1 - w, RING[f1], w, 0).astype(np.float32) / 255.0
    s, tx, ty = camera(f)
    if s < 0.9:
        src = cv2.GaussianBlur(src, (0, 0), 0.45 / s)
    M = np.float32([[s, 0, tx], [0, s, ty]])
    img = cv2.warpAffine(src, M, (OW, OH), flags=cv2.INTER_CUBIC if s > 1 else cv2.INTER_LINEAR)
    img = np.clip(img, 0, 1)

    lum = img @ LUMA
    hl = np.clip((lum - 0.30) / 0.5, 0, 1)
    hl = hl * hl * (3 - 2 * hl)
    a = cv2.remap(N1, (X / NS + p * 512) % 256, (Y / NS) % 256, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    b = cv2.remap(N2, (X / NS - p * 512) % 256, (Y / NS + p * 256) % 256, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    shimmer = (a * b * 4 - 1) * 0.30
    band = np.exp(-((U - band_centre(p, 0.10)) / BAND_W) ** 2) + np.exp(-((U - band_centre(p, 0.60)) / BAND_W) ** 2)
    # studio white is a little flat on black: deepen the mid-tones, then add the moving light
    img = img ** 1.18 * (1 + hl * (shimmer + 0.38 * band))[..., None]
    img = np.where(img > 0.78, 0.78 + 0.22 * (1 - np.exp(-(img - 0.78) / 0.22)), img)
    img += GLOW * (1 - np.clip(lum * 4, 0, 1))[..., None]

    for g in GLINTS:
        d = min((i - g["peak"]) % NF, (g["peak"] - i) % NF)
        if d > g["half"] or i not in g["pos"]:
            continue
        e = g["gain"] * math.cos(math.pi / 2 * d / (g["half"] + 1)) ** 2
        ox, oy = g["pos"][i][0] * s + tx, g["pos"][i][1] * s + ty
        sp = SPRITES[g["sprite"]]
        size = max(9, int(sp.shape[0] * (0.5 + 0.5 * e) * min(1.0, 0.55 + 0.45 * s)) | 1)
        spr = cv2.resize(sp, (size, size), interpolation=cv2.INTER_AREA) * (1.3 * e)
        x0, y0 = int(round(ox)) - size // 2, int(round(oy)) - size // 2
        xa, ya, xb, yb = max(0, x0), max(0, y0), min(OW, x0 + size), min(OH, y0 + size)
        if xa < xb and ya < yb:
            img[ya:yb, xa:xb] += spr[ya - y0:yb - y0, xa - x0:xb - x0, None] * g["tint"][None, None, :]

    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


if MODE == "--mattes":  # every cut-out frame in one sheet, to check the matte
    tiles = [cv2.resize(r, (320, 320), interpolation=cv2.INTER_AREA) for r in RING] + [np.zeros((320, 320, 3), np.uint8)] * 3
    cv2.imwrite(ARG, np.vstack([np.hstack(tiles[k:k + 8]) for k in range(0, 64, 8)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("mattes ->", ARG)
elif MODE == "--frame":
    cv2.imwrite(OUT, frame(int(ARG)), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("frame", ARG, "->", OUT)
elif MODE == "--poster":
    cv2.imwrite(ARG, frame(0), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("poster ->", ARG)
elif MODE == "--preview":
    picks = [int(NF * q) for q in (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65, 0.85)]
    tiles = [cv2.resize(frame(i), (640, 360), interpolation=cv2.INTER_AREA) for i in picks]
    for t, i in zip(tiles, picks):
        cv2.putText(t, f"{i / FPS:.1f}s", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(ARG, np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("preview ->", ARG, "| glints:", len(GLINTS))
else:
    import imageio_ffmpeg
    wr = imageio_ffmpeg.write_frames(OUT, (OW, OH), fps=FPS, codec="libx264", pix_fmt_in="bgr24", pix_fmt_out="yuv420p",
                                     macro_block_size=8,
                                     output_params=["-crf", "21", "-preset", "slow", "-movflags", "+faststart", "-an"])
    wr.send(None)
    for i in range(NF):
        wr.send(np.ascontiguousarray(frame(i)))
    wr.close()
    print("film ->", OUT, f"| {NF} frames, {SEC}s at {FPS} fps, {OW}x{OH} | glints: {len(GLINTS)}")
