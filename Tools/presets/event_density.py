"""event_density.py: how many new things happen a minute -- onsets counted the old way, and only what is new.

Plan B4 (20.11.2026). The figures went into half of the bank because, measured as played, it had "one and a half
onsets a minute where 322 recordings have twenty-five" (Family/tools/texture.py, `events`). Rene suspected that some
of what was added only raises that number: an onset is counted whether it says anything or not. This tool measures
both on the same audio, so that the recordings and renders of the bank stand in comparable columns:

  onsets      the old measure, texture.py's `events` copied verbatim (old_onsets() below): peaks of the spectral
              flux over third-octave bands, 21 ms hop, more than five median deviations over a four-second running
              median, peaks closer than 0.15 s merged. A minute of it.
  new         new events a minute. A candidate is a pitch whose salience rises by at least RISE_DB within RISE_S,
              measured from the lowest level that pitch (+/- one third of a semitone) had since its own last
              rise's peak, and whose peak is a tonal peak of the spectrum. It counts when it is new:
                - a new pitch: at its peak it stands RISE_DB over everything that pitch had in the MEMORY_S before
                  its rise began -- however slowly it came (a sung note), as long as it rose RISE_DB within RISE_S;
                - a clearly separated repetition: the pitch was heard in those seconds, but it had decayed by
                  RISE_DB, and rises again quickly (within FAST_S) with an onset of the broadband flux (old_onsets)
                  in its rise;
                - not a re-strike of the same pitch while its predecessor still rings (less than RISE_DB down);
                - not a partial that beats or breathes back up to where it was a moment ago, a vibrato within
                  +/- 33 cents, a swell slower than RISE_DB in RISE_S.
              Rises in several pitches within MERGE_S are one event (a chord, a note and its harmonics' ghosts); a
              weaker rise a harmonic away from an event of the last RISE_S is that event's late ghost.
              What it cannot tell apart: a partial of a held chord that surfaces as the sound moves (a tide, a
              filter sweep) from a soft new note on that partial's pitch -- both are a pitch that was not there.
  weighted    the same events, each weighted by how far it stands above the background at that moment:
              w = clip((SNR - FLOOR_DB) / (CEIL_DB - FLOOR_DB), 0, 1), where SNR is the power the event adds over
              its harmonics (A-weighted, the peak against the level before the rise) over the background, the median
              of the A-weighted power of the whole spectrum over the BACK_S before the event (a quantile estimate
              of the bed, as in quantile-based noise estimation). FLOOR_DB -20: an event with a hundredth of the
              bed's power, its power in one band against a bed spread over the ~25 critical bands of 40 Hz .. 16 kHz,
              lies near the masked threshold of a tone in noise (critical ratio, a few dB under the band's level).
              CEIL_DB 0: an event as strong as the whole bed is unmistakably foreground; in its own band it stands
              about 14 dB over the bed, where the Speech Intelligibility Index (ANSI S3.5) counts a band as fully
              audible (+15 dB). Between them the weight is linear in dB, as the SII's band audibility.
  repeats     the share of the new events whose pitch (+/- 33 cents) had an event in the REPEAT_S before.
Each event also keeps, for a look at what was counted: its pitch, its rise in dB and in seconds, its SNR and weight,
and how many of its first eight harmonics rose with it ("partials"; a pure tone has one).

What the detection follows (and why not something else):
  - pitch salience by harmonic summation (Klapuri, "Multiple fundamental frequency estimation by summing harmonic
    amplitudes", ISMIR 2006), on a spectrum resampled to thirds of a semitone; the weights 1/h;
  - onsets of tonal, non-percussive notes from a pitch representation rather than from broadband flux (Collins,
    "Using a pitch detector for onset detection", ISMIR 2005): the flux of the old measure misses soft attacks --
    the figures' 80 ms attacks did not register as events on 11.11.2026 -- and counts every re-strike;
  - the reference level taken as the maximum over the neighbouring pitch bins, so that vibrato and slow detuning do
    not rise (the maximum filter of SuperFlux, Boeck and Widmer, DAFx 2013);
  - the background as a running quantile of the power (Stahl, Fischer and Bippus, "Quantile based noise estimation
    for spectral subtraction and Wiener filtering", ICASSP 2000).
RISE_DB 10 is the range over which the early decay time is read (ISO 3382-1: the first 10 dB of a decay are what is
heard as the sound still ringing). MEMORY_S 3 s is the span of the "subjective present" (Poeppel, about three
seconds, within which successive sounds are heard as one gestalt). RISE_S 1 s lets the slow attacks of sung figures
count (a linear attack rises from silence by far more than 10 dB in its first second) and keeps a pad's swell of a
few seconds out. The thirds of a semitone and the +/- one-bin reference keep two pitches 25 cents apart apart only as
far as the analysis can (the engine's figure rule calls pitches within 25 cents one pitch).

    python Tools/presets/event_density.py --check
        synthetic signals: a re-strike in the tail must not count, a new pitch must, a repetition after silence must,
        a clearly separated repetition must, a tremolo pad and a noise bed must not; notes as loud as a noise bed
        weigh less than notes over a quiet one, notes far under it nothing. Exit 1 on failure (ctest
        `event_density`).
    python Tools/presets/event_density.py [--start S | --start-frac F] [--seconds S] [--json OUT] [--events] FILE...
        a line per file (and the events with --events)
    python Tools/presets/event_density.py --list LIST --cache DIR [--start-frac F] [--seconds S] [--min-length S]
                                          [--pause-every N] [--pause S] [--keep DIR]
        a list of "group<TAB>path" lines, one reader, a JSON per file in DIR (a restart continues where it stopped),
        a pause of --pause seconds every --pause-every files (for a hot drive); --keep keeps the decoded stretches
        (mono, 16 bit) and reads them from there next time
    python Tools/presets/event_density.py --table DIR...
        quartiles per group of every cached result in the folders

The process lowers its own priority (below normal) and keeps numpy to one thread.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from scipy import ndimage, signal  # noqa: E402

SR = 48000            # the rate everything is decoded to (as texture.py)
NFFT = 8192           # 171 ms: a semitone resolved from about 150 Hz, the lower notes through their harmonics
HOP = 480             # 10 ms
BPS = 36              # bins per octave: thirds of a semitone
MIDI_LO = 28.0        # the lowest bin, 41 Hz
MIDI_HI = 132.0       # the highest, 16.7 kHz
F0_LO = 33.0          # the lowest candidate pitch, 55 Hz
F0_HI = 96.0          # the highest, 2.1 kHz
HARMONICS = 8         # harmonics summed for a candidate's salience
RANGE_DB = 60.0       # salience under the file's 99th percentile minus this is the floor
RISE_DB = 10.0        # a pitch must rise this much over its own reference (see the file comment)
RISE_S = 1.0          # ... within this long
SMOOTH_S = 0.2        # the salience is averaged over this long first (a noisy bin's dips are no reference)
MERGE_S = 0.1         # rises closer than this are one event
BACK_S = 3.0          # the background: the median over this long before an event ...
BACK_GAP_S = 0.05     # ... ending this long before it
FLOOR_DB = -20.0      # weight 0 at and below this SNR
CEIL_DB = 0.0         # weight 1 at and above
REPEAT_S = 10.0       # a new event on a pitch that had one this recently is a repetition
MEMORY_S = 3.0        # a pitch is new when its peak stands RISE_DB over all it had in this long before its rise
FAST_S = 0.5          # otherwise (heard just before) the rise must be this quick ...
ONSET_TOL_S = 0.1     # ... and hold an onset of the broadband flux (old_onsets) within this much of it
PEAK_HARMONICS = 4    # one of the first four harmonics of a rising pitch must be a tonal peak at its peak ...
PEAK_NEAR = 3         # ... the largest within a semitone ...
PEAK_WIDE = 6         # ... and standing PEAK_DB over the smallest within two
PEAK_DB = 10.0        # (see PEAK_WIDE)
PEAK_RATIO = 10.0 ** (PEAK_DB / 10.0)   # PEAK_DB as a power ratio
PARTIAL_RISE_DB = 6.0     # an event's "partials": its harmonics that rose this much with it ...
PARTIAL_RANGE_DB = 30.0   # ... and stand within this of its strongest (a diagnostic, not a criterion)


def low_priority():
    """Lowers this process to below-normal priority on Windows (children started later inherit it)."""
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)


def decode(path, start=0.0, seconds=0.0):
    """The file as float mono (the mean of its channels) at 48 kHz, from @p start for @p seconds (0: to the end)."""
    cmd = ["ffmpeg", "-v", "error"]
    if start > 0.0:
        cmd += ["-ss", "%.3f" % start]
    cmd += ["-i", path]
    if seconds > 0.0:
        cmd += ["-t", "%.3f" % seconds]
    cmd += ["-ac", "2", "-ar", str(SR), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, creationflags=0x00004000 if os.name == "nt" else 0).stdout
    x = np.frombuffer(raw, dtype=np.float32).astype(np.float64)
    x = x[: len(x) // 2 * 2].reshape(-1, 2)
    return 0.5 * (x[:, 0] + x[:, 1])


def duration(path):
    """The file's length in seconds (0 when ffprobe cannot tell)."""
    p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                       capture_output=True, text=True)
    try:
        return float(p.stdout.strip())
    except ValueError:
        return 0.0


def old_onsets(m):
    """The old measure: texture.py's `events` (Family 86aee2e), verbatim -- onsets a minute from the spectral flux."""
    f, _, X = signal.stft(m, SR, window='hann', nperseg=2048, noverlap=1024, boundary=None, padded=False)
    p = np.abs(X) ** 2
    edges = 100.0 * 2.0 ** (np.arange(0, 21) / 3.0)   # thirds from 100 Hz to 10 kHz
    B = np.array([p[(f >= edges[i]) & (f < edges[i + 1])].sum(axis=0) for i in range(len(edges) - 1)])
    L = np.log10(B + 1e-9 * max(B.max(), 1e-30))
    flux = np.maximum(L[:, 1:] - L[:, :-1], 0.0).sum(axis=0)
    win = int(4.0 * SR / 1024)   # a running median over four seconds
    med = signal.medfilt(flux, win | 1)
    mad = np.median(np.abs(flux - med)) + 1e-9
    over = flux > med + 5.0 * 1.4826 * mad
    peaks = 0
    last = -10
    gap = int(0.15 * SR / 1024)
    for i in np.nonzero(over)[0]:
        if i - last > gap:
            peaks += 1
        last = i
    return peaks / (len(flux) * 1024.0 / SR / 60.0)


def flux_onset_times(m):
    """The times (s) of the onsets old_onsets() counts: the same flux, threshold and merging, the first frame of
    each peak (a frame's time is its centre)."""
    f, _, X = signal.stft(m, SR, window='hann', nperseg=2048, noverlap=1024, boundary=None, padded=False)
    p = np.abs(X) ** 2
    edges = 100.0 * 2.0 ** (np.arange(0, 21) / 3.0)
    B = np.array([p[(f >= edges[i]) & (f < edges[i + 1])].sum(axis=0) for i in range(len(edges) - 1)])
    L = np.log10(B + 1e-9 * max(B.max(), 1e-30))
    flux = np.maximum(L[:, 1:] - L[:, :-1], 0.0).sum(axis=0)
    med = signal.medfilt(flux, int(4.0 * SR / 1024) | 1)
    mad = np.median(np.abs(flux - med)) + 1e-9
    over = flux > med + 5.0 * 1.4826 * mad
    out = []
    last = -10
    gap = int(0.15 * SR / 1024)
    for i in np.nonzero(over)[0]:
        if i - last > gap:
            out.append(((i + 1) * 1024 + 1024) / SR)
        last = i
    return np.array(out)


def a_weight(f):
    """The A-weighting as a power factor at the frequencies @p f (as texture.py)."""
    f2 = np.maximum(f, 1.0) ** 2
    ra = (12194.0 ** 2 * f2 ** 2) / ((f2 + 20.6 ** 2) * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2))
                                    * (f2 + 12194.0 ** 2))
    return (ra / 0.7943) ** 2


def log_bins():
    """(centre frequencies of the thirds of a semitone, the matrix from FFT power to their power).

    Triangles in log frequency, each at least one FFT bin wide, every FFT bin's weights summing to one over the
    bins it falls in: the power is kept, only regrouped (in the bass several log bins share one FFT bin)."""
    n = int(round((MIDI_HI - MIDI_LO) * BPS / 12.0)) + 1
    midi = MIDI_LO + np.arange(n) * 12.0 / BPS
    fc = 440.0 * 2.0 ** ((midi - 69.0) / 12.0)
    ff = np.arange(NFFT // 2 + 1) * SR / NFFT
    lf = np.log2(np.maximum(ff, 1e-3))
    lc = np.log2(fc)
    half = np.maximum(1.0 / BPS, np.log2((fc + SR / NFFT) / fc))
    W = np.maximum(0.0, 1.0 - np.abs(lf[None, :] - lc[:, None]) / half[:, None])
    s = W.sum(axis=0)
    W[:, s > 0] /= s[s > 0]
    return fc, W.astype(np.float32)


def log_spectrogram(m):
    """The power in thirds of a semitone, shape (bins, frames), 10 ms frames of 171 ms."""
    fc, W = log_bins()
    win = signal.windows.hann(NFFT, sym=False).astype(np.float32)
    norm = 1.0 / float((win ** 2).sum())
    if len(m) < NFFT + HOP:
        return fc, np.zeros((len(fc), 0), dtype=np.float32)
    frames = (len(m) - NFFT) // HOP + 1
    x = m.astype(np.float32)
    out = np.empty((len(fc), frames), dtype=np.float32)
    step = 512
    for a in range(0, frames, step):
        b = min(frames, a + step)
        idx = (np.arange(a, b) * HOP)[:, None] + np.arange(NFFT)[None, :]
        X = np.fft.rfft(x[idx] * win[None, :], axis=1)
        P = (X.real ** 2 + X.imag ** 2) * norm
        out[:, a:b] = (P @ W.T).T
    return fc, out


def harm_offsets():
    """The bin offsets of harmonics 1 .. HARMONICS and their weights (1/h)."""
    off = np.array([int(round(BPS * np.log2(h))) for h in range(1, HARMONICS + 1)])
    g = 1.0 / np.arange(1, HARMONICS + 1)
    return off, g


def tonal_peak(col, bins):
    """Whether one of @p bins of the spectrum column @p col is a tonal peak: the largest within +/- PEAK_NEAR bins
    and PEAK_DB over the smallest within +/- PEAK_WIDE (a partial, as the peak picking of sinusoidal analysis takes
    it). The skirt of a partial that only grew louder, or the splash of an attack between the partials, is not."""
    n = len(col)
    for b in bins:
        if b >= n:
            break
        near = col[max(0, b - PEAK_NEAR):b + PEAK_NEAR + 1]
        wide = col[max(0, b - PEAK_WIDE):b + PEAK_WIDE + 1]
        if col[b] >= near.max() and col[b] >= PEAK_RATIO * max(float(wide.min()), 1e-30):
            return True
    return False


def rising_partials(P, bins, start, tp):
    """How many of the harmonic @p bins rose by PARTIAL_RISE_DB or more from frame @p start to @p tp and stand within
    PARTIAL_RANGE_DB of the strongest of them at @p tp (+/- one bin each): a note's partials rise together."""
    n = P.shape[0]
    bins = [b for b in bins if b < n]
    post = np.array([P[max(0, b - 1):b + 2, tp].max() for b in bins], dtype=np.float64)
    pre = np.array([P[max(0, b - 1):b + 2, start].max() for b in bins], dtype=np.float64)
    ok = (post >= pre * 10.0 ** (PARTIAL_RISE_DB / 10.0)) & (post >= post.max() * 10.0 ** (-PARTIAL_RANGE_DB / 10.0))
    return int(ok.sum())


def harmonic_kin(c, c2, off):
    """Whether candidates @p c and @p c2 (bins) are a harmonic apart either way (+/- one bin): then one of them
    shares its harmonics with the other, and in noise the weaker can cross the threshold a little after."""
    d = abs(int(c) - int(c2))
    return any(abs(d - o) <= 1 for o in off[1:])


def new_events(m, keep_events=False):
    """The new events of the mono signal @p m (see the file comment): a dict of the counts a minute (and the events
    themselves when @p keep_events)."""
    fc, P = log_spectrogram(m)
    T = P.shape[1]
    minutes = len(m) / SR / 60.0
    res = {"minutes": minutes}
    if T < 10:
        res.update({"new": 0.0, "weighted": 0.0, "repeats": 0.0, "events": [] if keep_events else None})
        return res
    off, g = harm_offsets()
    c_lo = int(round((F0_LO - MIDI_LO) * BPS / 12.0))
    c_hi = int(round((F0_HI - MIDI_LO) * BPS / 12.0))
    cands = np.arange(c_lo, c_hi + 1)
    S = np.zeros((len(cands), T), dtype=np.float64)
    for k, o in enumerate(off):
        idx = cands + o
        ok = idx < P.shape[0]
        S[ok] += g[k] * P[idx[ok]]
    S = ndimage.uniform_filter1d(S, size=max(1, int(round(SMOOTH_S * SR / HOP))), axis=1, mode="nearest")
    S = np.maximum(S, 0.0)                         # the running sum leaves -1e-12 next to digital silence
    L = 10.0 * np.log10(S + 1e-30)
    floor = np.percentile(L, 99) - RANGE_DB
    L = np.maximum(L, floor)
    Lmf = ndimage.maximum_filter1d(L, size=3, axis=0, mode="nearest")
    W = int(round(RISE_S * SR / HOP))
    mm = ndimage.minimum_filter1d(Lmf, size=W, axis=1, origin=(W - 1) // 2, mode="nearest")   # min of [t-W+1, t]
    ref = np.empty_like(mm)
    ref[:, 1:] = mm[:, :-1]                                                                    # min of [t-W, t-1]
    ref[:, 0] = np.inf
    cand_c, cand_t = np.nonzero(L - ref >= RISE_DB)
    WL = int(round(MEMORY_S * SR / HOP))
    fast = int(round(FAST_S * SR / HOP))
    flux_t = flux_onset_times(m)

    # One pass per candidate pitch over the frames where it rises enough: the window starts no earlier than the
    # peak of that pitch's last event, so a pitch that still rings (less than RISE_DB down) cannot rise again.
    rises = []                                     # (onset frame, cand index, pre frame, peak frame, partials)
    order = np.lexsort((cand_t, cand_c))
    cand_c, cand_t = cand_c[order], cand_t[order]
    starts = np.searchsorted(cand_c, np.arange(len(cands) + 1))
    for c in range(len(cands)):
        last_peak = -1
        row, rowmf = L[c], Lmf[c]
        for t in cand_t[starts[c]:starts[c + 1]]:
            if t <= last_peak:
                continue
            lo = max(t - W, last_peak, 0)
            seg = rowmf[lo:t]
            if len(seg) == 0:
                continue
            pre = lo + int(np.argmin(seg))
            if row[t] - rowmf[pre] < RISE_DB:
                continue
            start = lo + int(np.nonzero(seg <= rowmf[pre] + 3.0)[0][-1])     # where the rise leaves its floor
            tp = t
            while tp + 1 < T and row[tp + 1] > row[tp] and tp - t < 2 * W:
                tp += 1
            last_peak = tp
            if not tonal_peak(P[:, tp], cands[c] + off[:PEAK_HARMONICS]):
                continue
            before = rowmf[max(0, pre - WL):pre + 1]
            novel = row[tp] - before.max() >= RISE_DB
            if not novel:
                # Heard within the last seconds: only a clear re-articulation counts -- a quick rise with an
                # onset of the broadband flux in it (a beating or breathing partial has neither).
                a_s, b_s = (start * HOP + NFFT // 2) / SR - ONSET_TOL_S, (tp * HOP + NFFT // 2) / SR + ONSET_TOL_S
                k = np.searchsorted(flux_t, a_s)
                novel = tp - start <= fast and k < len(flux_t) and flux_t[k] <= b_s
            if novel:
                rises.append((t, c, pre, tp, (rising_partials(P, cands[c] + off, start, tp), tp - start)))
    rises.sort()

    # The background: the A-weighted power of the whole spectrum, its median over BACK_S before the event.
    aw = a_weight(fc).astype(np.float64)
    bed = (aw[:, None] * P).sum(axis=0)
    nb = int(round(BACK_S * SR / HOP))
    gap = int(round(BACK_GAP_S * SR / HOP))
    merge = int(round(MERGE_S * SR / HOP))
    rep = REPEAT_S * SR / HOP
    events = []
    i = 0
    last_on = {}                                    # candidate bin -> frame of its last event (for repeats)
    recent = []                                     # (frame, candidate, rise) of the events of the last RISE_S
    while i < len(rises):
        j = i
        while j + 1 < len(rises) and rises[j + 1][0] - rises[j][0] <= merge:
            j += 1
        best = None
        for t, c, pre, tp, nh in rises[i:j + 1]:
            e = 0.0
            for k, o in enumerate(off):
                b = cands[c] + o
                if b >= P.shape[0]:
                    break
                lo_b, hi_b = max(b - 1, 0), min(b + 2, P.shape[0])
                inc = P[lo_b:hi_b, tp].astype(np.float64) - P[lo_b:hi_b, pre]
                e += float((np.maximum(inc, 0.0) * aw[lo_b:hi_b]).sum())
            if best is None or e > best[0]:
                best = (e, t, c, pre, tp, nh)
        e, t, c, pre, tp, nh = best
        i = j + 1
        rise = float(L[c, tp] - Lmf[c, pre])
        if any(t - t2 <= W and rise < r2 and harmonic_kin(c, c2, off) for t2, c2, r2 in recent):
            continue                                # the late ghost of a pitch that rose just before
        recent = [x for x in recent if t - x[0] <= W] + [(t, c, rise)]
        a, b = max(0, t - gap - nb), max(1, t - gap)
        back = float(np.median(bed[a:b])) if b > a else float(bed[0])
        snr = 10.0 * np.log10(max(e, 1e-30) / max(back, 1e-30))
        w = float(np.clip((snr - FLOOR_DB) / (CEIL_DB - FLOOR_DB), 0.0, 1.0))
        repeat = any(t - last_on.get(cc, -1e18) <= rep for cc in (c - 1, c, c + 1))
        last_on[c] = t
        events.append({"t": round((t * HOP + NFFT // 2) / SR, 3), "hz": round(float(fc[cands[c]]), 2),
                       "rise_db": round(rise, 1), "snr_db": round(snr, 1), "w": round(w, 3),
                       "repeat": bool(repeat), "partials": nh[0], "rise_s": round(nh[1] * HOP / SR, 2)})
    n = len(events)
    res["new"] = n / minutes
    res["weighted"] = sum(ev["w"] for ev in events) / minutes
    res["repeats"] = (sum(ev["repeat"] for ev in events) / n) if n else 0.0
    res["events"] = events if keep_events else None
    return res


def measure(m, keep_events=False):
    """Both measures of the mono signal @p m: a dict (None when it is silent or shorter than 8 s, as texture.py)."""
    if len(m) < 8 * SR or float((m ** 2).mean()) < 1e-12:
        return None
    r = new_events(m, keep_events)
    r["onsets"] = old_onsets(m)
    if not keep_events:
        r.pop("events", None)
    return r


def line(name, r):
    """A file's measures as one line."""
    if r is None:
        return "%-40s (silent or too short)" % name[:40]
    return "%-40s onsets %6.1f  new %6.1f  weighted %6.1f  repeats %4.2f" % (
        name[:40], r["onsets"], r["new"], r["weighted"], r["repeats"])


# ---------------------------------------------------------------------------------------------------------------
# The check on synthetic signals
# ---------------------------------------------------------------------------------------------------------------

def tone(n, f0, t0, decay_s, end=None, amp=1.0, attack=0.02, release=0.05):
    """A harmonic tone (eight partials, 1/h) into a buffer of @p n samples from @p t0 s, decaying with time constant
    @p decay_s, ended at @p end s with a short release."""
    t = np.arange(n) / SR
    x = np.zeros(n)
    on = t >= t0
    tt = t[on] - t0
    env = np.minimum(1.0, tt / attack) * np.exp(-tt / decay_s)
    if end is not None:
        env *= np.clip(1.0 - (t[on] - end) / release, 0.0, 1.0)
    s = np.zeros(on.sum())
    for h in range(1, 9):
        if f0 * h < SR / 2:
            s += np.sin(2 * np.pi * f0 * h * tt) / h
    x[on] = amp * env * s
    return x


def pink(n, seed):
    """Pink noise of unit RMS."""
    rng = np.random.default_rng(seed)
    X = np.fft.rfft(rng.standard_normal(n))
    f = np.arange(len(X))
    X[1:] /= np.sqrt(f[1:])
    X[0] = 0.0
    y = np.fft.irfft(X, n)
    return y / np.sqrt((y ** 2).mean())


def check_signal(bed=0.0):
    """The check's 20 s: (signal, the onset times in s that must count, the times that must not); @p bed adds pink
    noise of that RMS (the notes peak at about 0.3)."""
    n = 20 * SR
    t = np.arange(n) / SR
    pad = 0.05 * (1.0 + 0.3 * np.sin(2 * np.pi * 0.5 * t)) * sum(np.sin(2 * np.pi * 98.0 * h * t) / h
                                                                  for h in range(1, 7))
    x = pad + 0.003 * pink(n, 1)
    notes = 0.3
    x = x + tone(n, 220.0, 2.0, 4.0, end=6.0, amp=notes)          # A3: new
    x = x + tone(n, 220.0, 3.0, 4.0, end=6.0, amp=notes)          # A3 again 1 s later, 2 dB down: a re-strike
    x = x + tone(n, 277.18, 4.0, 4.0, end=7.0, amp=notes)         # C#4 while A3 rings: a new pitch
    x = x + tone(n, 220.0, 9.0, 4.0, end=11.0, amp=notes)         # A3 after silence: counts (a repetition)
    x = x + tone(n, 659.26, 12.0, 0.8, end=12.25, amp=notes)      # E5, short
    x = x + tone(n, 659.26, 12.25, 0.8, end=14.0, amp=notes)      # ... struck again at once: a re-strike
    x = x + tone(n, 659.26, 15.0, 0.8, end=17.0, amp=notes)       # ... and after 24 dB of decay: counts
    if bed > 0.0:
        x = x + bed * pink(n, 2)
    return x, [2.0, 4.0, 9.0, 12.0, 15.0], [3.0, 12.25]


def run_check():
    """The synthetic check (ctest `event_density`): returns the exit code."""
    fails = []
    x, must, mustnot = check_signal()
    r = new_events(x, keep_events=True)
    ts = [e["t"] for e in r["events"]]
    print("check: events at %s (s); onsets the old way: %.1f a minute" % (", ".join("%.2f" % v for v in ts),
                                                                             old_onsets(x)))
    for e in r["events"]:
        print("  %6.2f s  %7.1f Hz  rise %5.1f dB  snr %6.1f dB  w %.2f  %s" % (
            e["t"], e["hz"], e["rise_db"], e["snr_db"], e["w"], "repeat" if e["repeat"] else "new pitch"))

    def near(v, ts_, tol=0.25):
        return [u for u in ts_ if abs(u - v) <= tol]
    for v in must:
        if len(near(v, ts)) != 1:
            fails.append("the onset at %.2f s should count once, found %d" % (v, len(near(v, ts))))
    for v in mustnot:
        if near(v, ts, 0.12):
            fails.append("the re-strike at %.2f s counted" % v)
    if len(ts) != len(must):
        fails.append("%d events, expected %d (a tremolo pad and a noise bed must give none)" % (len(ts), len(must)))
    kinds = {round(e["t"]): e["repeat"] for e in r["events"]}
    if kinds.get(9) is not True:
        fails.append("A3 at 9 s should be a repetition")
    if kinds.get(4) is not False:
        fails.append("C#4 at 4 s should be a new pitch")
    if any(e["w"] < 0.9 for e in r["events"]):
        fails.append("notes 15 dB over a quiet bed should weigh about 1")
    xl, _, _ = check_signal(bed=0.6)
    rl = new_events(xl, keep_events=True)
    print("check: the same notes in pink noise as loud as they are: %d events, %s" % (
        len(rl["events"]), ", ".join("%.2f s w %.2f" % (e["t"], e["w"]) for e in rl["events"])))
    if not rl["events"] or not all(0.4 <= e["w"] <= 0.95 for e in rl["events"]):
        fails.append("notes as loud as a noise bed should weigh between 0.4 and 0.95")
    xq, _, _ = check_signal(bed=3.0)
    rq = new_events(xq)
    print("check: the same notes under pink noise 20 dB louder: weighted %.2f a minute" % rq["weighted"])
    if rq["weighted"] > 0.1 * r["weighted"]:
        fails.append("notes under a much louder bed should weigh little (%.2f against %.2f)" % (rq["weighted"],
                                                                                              r["weighted"]))
    n = 20 * SR
    rn = new_events(0.1 * pink(n, 3) + 0.02 * np.random.default_rng(4).standard_normal(n))
    print("check: noise alone: %.2f new events a minute, weighted %.2f" % (rn["new"], rn["weighted"]))
    if rn["weighted"] > 1.0:
        fails.append("noise alone weighs %.2f events a minute" % rn["weighted"])
    for f in fails:
        print("FAIL: " + f)
    print("event_density --check: %s" % ("FAIL" if fails else "ok"))
    return 1 if fails else 0


# ---------------------------------------------------------------------------------------------------------------
# Lists, cache and tables
# ---------------------------------------------------------------------------------------------------------------

def stretch(path, start, start_frac, seconds, min_length=0.0):
    """Where to read @p path: (start s, seconds), or None when it is shorter than @p min_length s; --start-frac
    places the start at a share of its length (as the survey of 11.11.2026 did: 150 s from 35 %, a track shorter
    than 120 s left out)."""
    if start_frac is None and min_length <= 0.0:
        return start, seconds
    d = duration(path)
    if d < min_length:
        return None
    return (start_frac * d if start_frac is not None else start), seconds


def cache_name(group, path):
    """The cache file of one list entry: the group and a hash of the path (the names stay short and unique)."""
    h = hashlib.sha1(path.encode("utf-8")).hexdigest()[:12]
    return "%s_%s.json" % ("".join(ch if ch.isalnum() else "_" for ch in group), h)


def run_list(lst, cache, start, start_frac, seconds, min_length, pause_every, pause_s, keep=None):
    """Measures every entry of the list file @p lst not yet in @p cache, one at a time; returns the exit code. With
    @p keep, the decoded stretch is kept there (mono, 16 bit) and read from there the next time, so that a changed
    measure does not read the recordings again."""
    os.makedirs(cache, exist_ok=True)
    if keep:
        os.makedirs(keep, exist_ok=True)
    entries = []
    for ln in open(lst, encoding="utf-8"):
        ln = ln.rstrip("\r\n")
        if ln and not ln.startswith("#"):
            g, p = ln.split("\t", 1)
            entries.append((g, p))
    done = 0
    for k, (g, p) in enumerate(entries):
        out = os.path.join(cache, cache_name(g, p))
        if os.path.exists(out):
            continue
        t0 = time.time()
        where = stretch(p, start, start_frac, seconds, min_length)
        if where is None:
            rec = {"group": g, "path": p, "skipped": "shorter than %.0f s" % min_length, "result": None}
            m, t1 = np.zeros(0), time.time()
            r, t2 = None, t1
        else:
            m = None
            kept = os.path.join(keep, cache_name(g, p)[:-5] + ".npz") if keep else None
            if kept and os.path.exists(kept):
                z = np.load(kept)
                m = z["x"].astype(np.float64) * (float(z["peak"]) / 32767.0)
            else:
                m = decode(p, *where)
                if kept:
                    peak = max(float(np.abs(m).max()), 1e-9)
                    np.savez(kept, x=np.round(m / peak * 32767.0).astype(np.int16), peak=peak)
            t1 = time.time()
            r = measure(m)
            t2 = time.time()
            rec = {"group": g, "path": p, "start": round(where[0], 2), "seconds": round(len(m) / SR, 2),
                   "read_s": round(t1 - t0, 2), "measure_s": round(t2 - t1, 2), "result": r}
        with open(out + ".tmp", "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=1)
        os.replace(out + ".tmp", out)
        print("%4d/%d %s  (read %.1f s, measure %.1f s)" % (
            k + 1, len(entries), line(g + ": " + os.path.basename(p), r), t1 - t0, t2 - t1), flush=True)
        done += 1
        if pause_every > 0 and done % pause_every == 0:
            time.sleep(pause_s)
    return 0


def quartiles(v):
    """The first quartile, the median and the third of @p v (None when empty)."""
    if not v:
        return None
    return [round(float(np.percentile(v, q)), 2) for q in (25, 50, 75)]


def table(dirs):
    """Quartiles per group of the cached results in @p dirs, printed; returns the dict."""
    by = {}
    for d in dirs:
        for fn in sorted(os.listdir(d)):
            if not fn.endswith(".json"):
                continue
            rec = json.load(open(os.path.join(d, fn), encoding="utf-8"))
            r = rec.get("result")
            if r is None:
                continue
            for g in (rec["group"], "all"):
                by.setdefault(g, []).append(r)
    out = {}
    print("%-24s %4s  %-20s %-20s %-20s %-16s" % ("group", "n", "onsets (q1 med q3)", "new", "weighted", "repeats"))
    for g in sorted(by, key=lambda k: (k == "all", k)):
        rs = by[g]
        out[g] = {"count": len(rs)}
        cols = []
        for key in ("onsets", "new", "weighted", "repeats"):
            q = quartiles([r[key] for r in rs])
            out[g][key] = q
            cols.append("%5.1f %5.1f %5.1f" % tuple(q) if key != "repeats" else "%4.2f %4.2f %4.2f" % tuple(q))
        print("%-24s %4d  %-20s %-20s %-20s %-16s" % (g[:24], len(rs), *cols))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="the synthetic check (exit 1 on failure)")
    ap.add_argument("--list", help="a list of group<TAB>path lines")
    ap.add_argument("--cache", help="the folder of per-file results for --list")
    ap.add_argument("--table", nargs="+", help="quartiles per group of the cached results in these folders")
    ap.add_argument("--start", type=float, default=0.0, help="seconds into each file")
    ap.add_argument("--start-frac", type=float, default=None, help="start at this share of each file's length")
    ap.add_argument("--seconds", type=float, default=0.0, help="how long to read (0: to the end)")
    ap.add_argument("--keep", help="keep the decoded stretches here and read them from here (--list)")
    ap.add_argument("--min-length", type=float, default=0.0, help="leave out files shorter than this (s)")
    ap.add_argument("--pause-every", type=int, default=0, help="pause after this many files (--list)")
    ap.add_argument("--pause", type=float, default=5.0, help="seconds of each pause")
    ap.add_argument("--json", help="write the results of the FILEs here")
    ap.add_argument("--events", action="store_true", help="print and keep every event")
    ap.add_argument("files", nargs="*")
    a = ap.parse_args()
    low_priority()
    if a.check:
        return run_check()
    if a.list:
        if not a.cache:
            ap.error("--list needs --cache")
        return run_list(a.list, a.cache, a.start, a.start_frac, a.seconds, a.min_length, a.pause_every, a.pause,
                        a.keep)
    if a.table:
        res = table(a.table)
        if a.json:
            json.dump(res, open(a.json, "w", encoding="utf-8"), indent=1)
        return 0
    res = {}
    for f in a.files:
        where = stretch(f, a.start, a.start_frac, a.seconds, a.min_length)
        r = measure(decode(f, *where), keep_events=a.events) if where else None
        res[f] = r
        print(line(os.path.splitext(os.path.basename(f))[0], r), flush=True)
        if a.events and r:
            for e in r["events"]:
                print("  %7.2f s  %7.1f Hz  rise %5.1f dB in %4.2f s  snr %6.1f dB  w %.2f  partials %d  %s" % (
                    e["t"], e["hz"], e["rise_db"], e["rise_s"], e["snr_db"], e["w"], e["partials"],
                    "repeat" if e["repeat"] else "new"))
    if a.json:
        json.dump(res, open(a.json, "w", encoding="utf-8"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
