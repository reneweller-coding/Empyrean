"""The demos of Empyrean: a preset of each of its wonders as MP3, and one of them as a video by KaleidoscopeEnhanced
(17.10.2026, after Totality's Tools/demo/make_demos.py).

    python Tools/demo/make_demos.py [--only NAME ...] [--no-video] [--kaleidoscope DIR] [--exe RENDERER]
    python Tools/demo/make_demos.py --publish

Every demo below is rendered with the instrument's own renderer (empy_render: the preset, its conductor playing, the
seed fixed; its first minute left out, while the sound settles) into work/demos/ and encoded as an MP3 (ffmpeg, LAME VBR around 190 kbit/s, title and group in the tags,
a fade in and a fade out, brought to -16 LUFS by its gain alone as far as the true peak stays at -1 dBTP). The demo marked VIDEO
is also drawn by KaleidoscopeEnhanced (github.com/reneweller-coding/KaleidoscopeEnhanced) with its offline export
(-e: frame by frame from the sound, faster than real time, its settings in work/demos so the show's stay untouched);
a section every half minute lets its scenes change, as an ambient piece has no bars. The export is encoded once more for the web
(H.264 at most 6 Mbit/s, AAC 256 kbit/s, faststart) and a poster frame is cut from it.

--kaleidoscope (or KALEIDO_ROOT) names KaleidoscopeEnhanced's dist folder (bin/Kaleidoscope.exe beside Presets, ...),
by default Desktop/Kaleidoscope Enhanced/dist/KaleidoscopeVisualizer; its photos are the repository's Images two
folders up. ffmpeg must be on PATH, or FFMPEG names it.

Everything lands in work/demos/: <name>.wav (the render), <name>.mp3, and for the video <name>.mp4 and <name>.jpg.
--publish then puts the MP3s, the video and its poster on the GitHub release "demos" (gh; created on first use, never
marked as the latest release), where the README and the family's page link to them.
"""
import argparse
import glob
import os
import re
import shutil
import struct
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'work', 'demos')

# ---------------------------------------------------------------- this instrument
NAME = 'Empyrean'
RENDER = 'empy_render'
GENRE = 'Ambient'
FADE_IN = 3.0
MINUTES = 3
WARMUP = 60   # s rendered first and left out: the sound settles (long attacks, the conductor's first chords)
LOUDNESS = -16.0   # LUFS the demos are brought to, by their gain alone
# name, title, the group it shows, the renderer's arguments: three minutes of a preset each, its conductor playing
DEMOS = [
    ('lunar_hall', 'Lunar Hall', 'a sleep concert in just intonation',
     ['--preset', 'Lunar Hall', '--seed', '7']),
    ('teeming_garden', 'Teeming Garden', 'a Lenia field the partials follow',
     ['--preset', 'Teeming Garden', '--seed', '11']),
    ('thunderous_rainpool', 'Thunderous Rainpool', 'rain, thunder and a night chorus',
     ['--preset', 'Thunderous Rainpool', '--seed', '23']),
    ('convective_betelgeuse', 'Convective Betelgeuse', "a red giant's oscillations",
     ['--preset', 'Convective Betelgeuse', '--seed', '42']),
    ('sliding_whistler', 'Sliding Whistler', 'whistlers in the magnetosphere',
     ['--preset', 'Sliding Whistler', '--seed', '5']),
    ('steppe_night', 'Rumbling-Voice Steppe Night', 'giant throats singing kargyraa',
     ['--preset', 'Rumbling-Voice Steppe Night', '--seed', '3']),
]
VIDEO = ('convective_betelgeuse', 'SpaceAmbient')   # the demo that becomes a video, and Kaleidoscope's configuration
POSTER_AT = 40   # the video's frame that becomes its poster (seconds)
SECTION = 30     # s between the video's sections (its scene changes)


# ---------------------------------------------------------------- tools
def ffmpeg():
    """The ffmpeg to run: FFMPEG, or the one on PATH."""
    exe = os.environ.get('FFMPEG') or shutil.which('ffmpeg')
    if not exe:
        sys.exit('ffmpeg not found: put it on PATH or name it in FFMPEG')
    return exe


def aac():
    """The AAC encoder for the video's sound: Windows' own (aac_mf) where ffmpeg has it, else the native one."""
    p = subprocess.run([ffmpeg(), '-hide_banner', '-encoders'], capture_output=True, text=True, errors='replace')
    return ['-c:a', 'aac_mf', '-b:a', '256000'] if ' aac_mf ' in p.stdout else ['-c:a', 'aac', '-b:a', '256k']


def renderer(given):
    """The renderer: --exe, else the newest bin/<preset>/RENDER.exe of this checkout."""
    if given:
        return os.path.abspath(given)
    found = glob.glob(os.path.join(ROOT, 'bin', '*', RENDER + ('.exe' if os.name == 'nt' else '')))
    if not found:
        sys.exit('no %s in bin/: build it first or name it with --exe' % RENDER)
    return max(found, key=os.path.getmtime)


def wav_seconds(path):
    """The length of a WAV in seconds, from its header (PCM or float, any chunk order)."""
    with open(path, 'rb') as f:
        if f.read(4) != b'RIFF':
            raise ValueError('not a WAV: ' + path)
        f.read(8)
        rate = block = 0
        while True:
            head = f.read(8)
            if len(head) < 8:
                break
            cid, size = head[:4], struct.unpack('<I', head[4:])[0]
            if cid == b'fmt ':
                fmt = f.read(size)
                rate, block = struct.unpack('<I', fmt[4:8])[0], struct.unpack('<H', fmt[12:14])[0]
            elif cid == b'data':
                return size / float(rate * block) if rate and block else 0.0
            else:
                f.seek(size + (size & 1), 1)
    return 0.0


def headroom(path):
    """The gain in dB (never above 0) that would bring a file's true peak down to -1 dBTP."""
    p = subprocess.run([ffmpeg(), '-hide_banner', '-nostats', '-i', path, '-af', 'ebur128=peak=true', '-f', 'null', '-'],
                       capture_output=True, text=True, errors='replace')
    peaks = re.findall(r'Peak:\s+(-?[\d.]+|-inf) dBFS', p.stderr)
    if not peaks or peaks[-1] == '-inf':
        return 0.0
    return min(0.0, -1.0 - float(peaks[-1]))


def fades(seconds, gain=0.0):
    """The fades of a demo: in over FADE_IN seconds, out over the last eight; and its gain."""
    out = ['volume=%.2fdB' % gain] if abs(gain) >= 0.01 else []
    if FADE_IN > 0:
        out.append('afade=t=in:d=%.1f' % FADE_IN)
    out.append('afade=t=out:st=%.3f:d=8' % max(0.0, seconds - 8.0))
    return ','.join(out)


def render(exe, name, args):
    """Renders one demo into work/demos/<name>.wav: WARMUP seconds and the demo's minutes, the warm-up left out."""
    wav = os.path.join(OUT, name + '.wav')
    full = os.path.join(OUT, name + '.full.wav')
    cmd = [exe, '--seconds', str(WARMUP + MINUTES * 60)] + args + ['--out', full]
    print('render', name, ' '.join(args), flush=True)
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, errors='replace')
    if p.returncode == 0 and os.path.exists(full):
        subprocess.run([ffmpeg(), '-v', 'error', '-y', '-ss', str(WARMUP), '-i', full, '-t', str(MINUTES * 60),
                        '-c:a', 'pcm_f32le', wav], check=True)
        os.remove(full)
    if p.returncode != 0 or not os.path.exists(wav):
        sys.exit('render failed (%d):\n%s%s' % (p.returncode, p.stdout[-2000:], p.stderr[-2000:]))
    return wav


def loudness(path, before=''):
    """A file's integrated loudness (LUFS) and true peak (dBTP), after the filters @p before (its fades)."""
    chain = (before + ',' if before else '') + 'ebur128=peak=true'
    p = subprocess.run([ffmpeg(), '-hide_banner', '-nostats', '-i', path, '-af', chain, '-f', 'null', '-'],
                       capture_output=True, text=True, errors='replace')
    i = re.findall(r'I:\s+(-?[\d.]+) LUFS', p.stderr)
    peaks = re.findall(r'Peak:\s+(-?[\d.]+|-inf) dBFS', p.stderr)
    return (float(i[-1]) if i else -70.0), (float(peaks[-1]) if peaks and peaks[-1] != '-inf' else -70.0)


def mp3(wav, name, title, what):
    """The MP3 of a demo, with its tags and its fades, brought to LOUDNESS by its gain alone as far as its true peak
    allows (the bank is levelled by its loudest seconds as played, so a quiet stretch of ambient sits far below that);
    encoded once more a little quieter when the encoder's true peak came out above -1 dBTP. Returns the gain it took."""
    out = os.path.join(OUT, name + '.mp3')
    seconds = wav_seconds(wav)
    lufs, peak = loudness(wav, fades(seconds))   # as it is heard: a peak inside a fade is not one
    gain = min(LOUDNESS - lufs, -1.0 - peak)
    for _ in range(3):
        subprocess.run([ffmpeg(), '-v', 'error', '-y', '-i', wav, '-af', fades(seconds, gain), '-c:a', 'libmp3lame',
                        '-q:a', '2', '-metadata', 'title=%s (%s demo: %s)' % (title, NAME, what),
                        '-metadata', 'artist=' + NAME, '-metadata', 'album=%s demos' % NAME,
                        '-metadata', 'genre=' + GENRE,
                        '-metadata', 'comment=Rendered by %s; github.com/reneweller-coding/%s' % (RENDER, NAME), out],
                       check=True)
        more = headroom(out)
        if more >= 0.0:
            break
        gain += more - 0.2
    print('  mp3 %s (%.0f s, %.1f dB)' % (out, seconds, gain), flush=True)
    return gain


def video(wav, name, title, config, kaleido, gain=0.0):
    """The video of a demo: KaleidoscopeEnhanced's offline export from the faded sound, the web encode, the poster."""
    seconds = wav_seconds(wav)
    flac = os.path.join(OUT, name + '.video.flac')
    subprocess.run([ffmpeg(), '-v', 'error', '-y', '-i', wav, '-af', fades(seconds, gain), '-c:a', 'flac',
                    '-sample_fmt', 's32', '-bits_per_raw_sample', '24', '-ar', '48000', flac], check=True)
    raw = os.path.join(OUT, name + '.export.mp4')
    bindir = os.path.join(kaleido, 'bin')
    photos = os.path.normpath(os.path.join(kaleido, '..', '..', 'Images'))
    if not os.path.isdir(photos):
        photos = os.path.join(kaleido, 'Images')
    env = dict(os.environ)
    env['KALEIDO_SETTINGS'] = os.path.join(OUT, 'kaleidoscope_settings.ini')   # the show's settings stay its own
    # Sections every SECTION seconds, gently rising and falling: an ambient piece has no bars, but a section lets
    # KaleidoscopeEnhanced change its scene (one of its long lab scenes would otherwise hold all three minutes).
    tsv = os.path.join(OUT, name + '.cues.tsv')
    with open(tsv, 'w', encoding='utf-8', newline='\n') as f:
        f.write('# %s, %s: score cues for KaleidoscopeEnhanced -k (seconds, address, arguments)\n' % (NAME, title))
        kinds = [('Intro', 0.25), ('Groove', 0.4), ('Build', 0.5), ('Groove', 0.45), ('Break', 0.35), ('Groove', 0.4)]
        at, k = 0.0, 0
        while at < seconds - SECTION / 2:
            kind, energy = kinds[k % len(kinds)] if at < seconds - SECTION else ('Outro', 0.25)
            f.write('%.3f\t/phos/section\t%s\t%.2f\n' % (at, kind, energy))
            at += SECTION
            k += 1
    cmd = [os.path.join(bindir, 'Kaleidoscope.exe'), '-t', '0', '-c', config, '-f', photos, '-e', flac,
           '-g', '1920x1080', '-p', '30', '-j', 'h264,q=high,title,noblur', '-d', raw, '-k', tsv]
    print('  kaleidoscope', ' '.join(cmd[1:]), flush=True)
    with open(os.path.join(OUT, name + '.kaleidoscope.log'), 'w', encoding='utf-8', errors='replace') as log:
        code = subprocess.run(cmd, cwd=bindir, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    if code != 0 or not os.path.exists(raw) or os.path.getsize(raw) == 0:
        sys.exit('KaleidoscopeEnhanced made no video (code %d, see %s.kaleidoscope.log)' % (code, name))
    out = os.path.join(OUT, name + '.mp4')
    subprocess.run([ffmpeg(), '-v', 'error', '-y', '-i', raw, '-i', flac, '-map', '0:v:0', '-map', '1:a:0',
                    '-c:v', 'libx264', '-preset', 'slow', '-crf', '23', '-maxrate', '6M', '-bufsize', '12M',
                    '-pix_fmt', 'yuv420p'] + aac() + ['-shortest', '-movflags', '+faststart',
                    '-metadata', 'title=%s (%s demo)' % (title, NAME), '-metadata', 'artist=' + NAME,
                    '-metadata', 'comment=Music by %s, pictures by KaleidoscopeEnhanced (%s)' % (NAME, config), out],
                   check=True)
    jpg = poster(out, name)
    os.remove(flac)
    os.remove(raw)
    print('  video %s, poster %s' % (out, jpg), flush=True)
    return out


def poster(mp4, name):
    """The poster of the video: its frame at POSTER_AT seconds, 1280 pixels wide, as work/demos/<name>.jpg."""
    jpg = os.path.join(OUT, name + '.jpg')
    subprocess.run([ffmpeg(), '-v', 'error', '-y', '-ss', '%.1f' % POSTER_AT, '-i', mp4, '-frames:v', '1',
                    '-vf', 'scale=1280:-2', '-q:v', '3', jpg], check=True)
    return jpg


def publish():
    """The MP3s, the video and its poster on the release "demos" (gh), which is created the first time."""
    files = [os.path.join(OUT, n + '.mp3') for n, _, _, _ in DEMOS]
    files += [os.path.join(OUT, VIDEO[0] + e) for e in ('.mp4', '.jpg')]
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        sys.exit('not rendered yet: ' + ', '.join(os.path.basename(f) for f in missing))
    lines = ['Demos of %s, rendered by its own %s from the presets below, %d minutes each, the conductor playing -- '
             'nothing played in, nothing edited.' % (NAME, RENDER, MINUTES),
             '', '| demo | what it shows | file | renderer arguments |', '|---|---|---|---|']
    for n, title, what, args in DEMOS:
        lines.append('| %s | %s | `%s.mp3` | `--seconds %d %s` |' % (
            title, what, n, MINUTES * 60, ' '.join('"%s"' % x if ' ' in x else x for x in args)))
    lines += ['', 'The video (`%s.mp4`) is the %s demo with pictures by '
              '[KaleidoscopeEnhanced](https://github.com/reneweller-coding/KaleidoscopeEnhanced) (configuration %s), '
              'drawn offline from the sound. Made by `Tools/demo/make_demos.py`.'
              % (VIDEO[0], dict((n, t) for n, t, _, _ in DEMOS)[VIDEO[0]], VIDEO[1])]
    notes = os.path.join(OUT, 'release.md')
    with open(notes, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(lines) + '\n')
    if subprocess.run(['gh', 'release', 'view', 'demos'], cwd=ROOT, capture_output=True).returncode != 0:
        subprocess.run(['gh', 'release', 'create', 'demos', '--title', '%s demos' % NAME, '--notes-file', notes,
                        '--latest=false'], cwd=ROOT, check=True)
    else:
        subprocess.run(['gh', 'release', 'edit', 'demos', '--notes-file', notes], cwd=ROOT, check=True)
    subprocess.run(['gh', 'release', 'upload', 'demos', '--clobber'] + files, cwd=ROOT, check=True)
    print('published %d files on the release "demos"' % len(files))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--only', nargs='*', help='only these demos (by name)')
    ap.add_argument('--no-video', action='store_true', help='the MP3s only')
    ap.add_argument('--kaleidoscope', default=os.environ.get('KALEIDO_ROOT', os.path.join(
        os.path.expanduser('~'), 'Desktop', 'Kaleidoscope Enhanced', 'dist', 'KaleidoscopeVisualizer')),
        help="KaleidoscopeEnhanced's dist folder (bin/Kaleidoscope.exe beside Presets, ...); or KALEIDO_ROOT")
    ap.add_argument('--exe', help='the renderer (default: the newest bin/*/%s)' % RENDER)
    ap.add_argument('--publish', action='store_true', help='put what is rendered on the release "demos" and stop')
    a = ap.parse_args()
    if a.publish:
        publish()
        return
    os.makedirs(OUT, exist_ok=True)
    exe = renderer(a.exe)
    want_video = not a.no_video
    if want_video and not os.path.exists(os.path.join(a.kaleidoscope, 'bin', 'Kaleidoscope.exe')):
        sys.exit('no KaleidoscopeEnhanced at "%s": name its folder with --kaleidoscope or KALEIDO_ROOT, '
                 'or render the MP3s alone with --no-video' % a.kaleidoscope)
    for name, title, what, args in DEMOS:
        if a.only and name not in a.only:
            continue
        wav = render(exe, name, args)
        gain = mp3(wav, name, title, what)
        if want_video and name == VIDEO[0]:
            video(wav, name, title, VIDEO[1], a.kaleidoscope, gain)


if __name__ == '__main__':
    main()
