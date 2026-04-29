---
jupyter:
  jupytext:
    cell_metadata_filter: -all
    formats: md,ipynb
    main_language: bash
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.1
---

# FFmpeg audio tutorial (CLI-focused)

This tutorial assumes you already have a working `ffmpeg` and `ffplay` in your PATH.
Commands are shown for a single input file `input.wav` and can be adapted.

Basic pattern:

```bash
ffmpeg -i input.wav [global options] -af "filter1=...,filter2=..." [more options] output.wav
# or
ffmpeg -i input.wav -filter:a "filters..." output.wav
```

Filters in the same chain are comma-separated; multiple chains are separated by `;`. ([FFmpeg][1])

---

## Part I – Sound processing

### 1. Simple normalization with the `volume` filter

**Goal:** fixed gain change (e.g., “+5 dB”) or “set this file to a given loudness”.

#### 1.1 Change volume by a fixed amount

```bash
# Increase volume by 5 dB
ffmpeg -i input.wav -af "volume=5dB" output.wav

# Halve the volume (linear factor 0.5)
ffmpeg -i input.wav -af "volume=0.5" output_quieter.wav
```

Key option (audio filter `volume`): ([Super User][2])

* `volume=`

  * `volume=5dB` – gain in dB.
  * `volume=2.0` – linear gain factor.
  * You can also use expressions like `volume=0.8*volume`.

Use this when you already know the gain you want (e.g., from `volumedetect` below).

---

### 2. `volumedetect` + `loudnorm` (EBU R128 loudness)

**Typical use:** consistent perceived loudness across files (broadcast / publishing style). `loudnorm` implements EBU R128 loudness normalization. ([FFmpeg Trac][3])

#### 2.1 Analyze with `volumedetect` (peak & mean)

```bash
# Unix-like
ffmpeg -i input.wav -af "volumedetect" -f null - 2>&1 | tee volume.log

# Windows (NUL instead of /dev/null)
ffmpeg -i input.wav -af "volumedetect" -f null NUL 2>&1 | tee volume.log
```

Look for lines like:

```text
mean_volume: -18.0 dB
max_volume:  -3.0 dB
```

You can either:

* Use this info to decide a simple `volume=` gain, or
* Ignore it and let `loudnorm` do a full loudness-based normalization.

#### 2.2 One-pass `loudnorm` (simple mode)

```bash
ffmpeg -i input.wav -af "loudnorm=I=-16:TP=-1.5:LRA=11" loudnorm_1pass.wav
```

Key options (simplified): ([FFmpeg Trac][3])

* `I` – target integrated loudness in LUFS (e.g., `-16` or `-23` for broadcast).
* `TP` – maximum true peak in dBTP (e.g., `-1.5`).
* `LRA` – target loudness range (e.g., `11`).

**Usage tip:** For speech podcasts / YouTube, `I=-16:TP=-1.5:LRA=11` is a common starting point.

#### 2.3 Two-pass `loudnorm` (more accurate, recommended)

Pass 1 – measure:

```bash
ffmpeg -i input.wav -af "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json" -f null -
```

Copy the JSON stats (`input_i`, `input_tp`, `input_lra`, `input_thresh`, `offset`).

Pass 2 – apply with measured stats:

```bash
ffmpeg -i input.wav -af \
"loudnorm=I=-16:TP=-1.5:LRA=11:measured_I=-18.0:measured_TP=-3.0:measured_LRA=9.0:measured_thresh=-30:offset=1.5:linear=true:print_format=summary" \
loudnorm_2pass.wav
```

Replace the example values with the ones from your pass-1 output.

---

### 3. Speech-oriented normalization with `speechnorm` (best “out of the box” for speech)

`speechnorm` is designed specifically for speech: it increases quiet speech and gently compresses loud parts, with presets for “weak” to “extreme” amplification. ([FFmpeg][1])

#### 3.1 Basic usage

```bash
# Moderate, relatively slow amplification (good general speech preset)
ffmpeg -i input.wav -af "speechnorm=e=6.25:r=0.00001:l=1" speech_norm.wav
```

Key options (most important ones): ([FFmpeg][4])

* `e` – **expansion** strength (how much to boost quiet speech).

  * Typical examples:

    * `e=3` – weak & slow
    * `e=6.25` – moderate & slow
    * `e=12.5` – strong & fast
    * `e=25` or `e=50` – very strong / extreme
* `r` – **reaction** speed (how quickly gain changes).

  * Small values (`0.00001`) = slow, smooth.
  * Larger (`0.0001`) = faster reaction.
* `l` – **linked channels**:

  * `l=1` – link channels (stereo tracks move together).
  * `l=0` – treat each channel independently.

Start with `e=6.25:r=0.00001:l=1` for general conversational speech and adjust by ear; this is usually the simplest option that “just works” for ASR pre-processing.

---

### 4. Filtering & noise suppression

#### 4.1 Low-pass and high-pass filters

These are classic IIR filters that cut low or high frequencies. Useful as a first pass to remove rumble (<80 Hz) and harsh high-frequency noise. ([FFmpeg][5])

```bash
# Keep mainly the speech band ~200–3500 Hz
ffmpeg -i input.wav -af "highpass=f=200,lowpass=f=3500" bandlimited.wav
```

Key option for both `highpass` and `lowpass`:

* `f` – cutoff frequency in Hz (e.g., `f=200`).

---

#### 4.2 FFT-based denoiser `afftdn`

`afftdn` works in the frequency domain and lets you control how aggressively to reduce noise. ([FFmpeg][1])

```bash
ffmpeg -i input.wav -af "afftdn=nr=12:nf=-60" denoised_afftdn.wav
```

Key options (simplified):

* `nr` – **noise reduction** amount in dB (typical: `6–18`).
* `nf` – **noise floor** in dB; more negative = more aggressive (`-60` is a reasonable starting point).
* `om` – output mode; leave default for normal denoised output.

Use `afftdn` when you want a “classic” spectral denoiser with tunable aggressiveness.

---

#### 4.3 Neural denoiser `arnndn` (RNN noise reduction)

`arnndn` uses a trained neural network model (`.rnnn` file). It is very strong at removing stationary background noise, especially for speech. ([Super User][6])

You must download a compatible model file (e.g., `cb.rnnn`) and point the filter to it:

```bash
# Strong AI-based denoising
ffmpeg -i input.wav -af "arnndn=m=cb.rnnn" denoised_arnndn.wav
```

Key options:

* `m` / `model` – path to the model file, required.
* `mix` – blend of dry/wet signal:

  * `mix=1.0` – fully denoised (default).
  * `mix=0.5` – half denoised, half original.

Try `arnndn` when you have strong constant noise (fans, hum, traffic) and want a single “good default” noise filter.

---

#### 4.4 `dialoguenhance` – out-of-the-box dialogue booster

`dialoguenhance` enhances speech present in both stereo channels and outputs 3.0 (L, R, C) audio with an emphasized center channel for dialogue. Very useful with noisy stereo recordings or where speech is buried in music. ([FFmpeg][1])

```bash
ffmpeg -i input_stereo.wav -af "dialoguenhance=original=1:enhance=2:voice=4" enhanced_dialogue.wav
```

Key options:

* `original` – how much of the original center to retain (0–1, default `1`).
* `enhance` – how much to boost detected dialogue (0–3, default `1`).

  * `2` is a good starting point.
* `voice` – sensitivity of voice detection (2–32, default `2`).

  * Higher values can make detection more selective.

For ASR pipelines you might apply `dialoguenhance` first, then downmix to mono (e.g. `pan=mono|c0=c2`) before feeding the recognizer.

---

#### 4.5 Combining filters

You can chain filters in a single `-af`:

```bash
ffmpeg -i input.wav -af \
"highpass=f=120,lowpass=f=3800,arnndn=m=cb.rnnn,dialoguenhance=original=1:enhance=2:voice=4,speechnorm=e=6.25:r=0.00001:l=1" \
processed.wav
```

Filter order here:

1. **High-pass & low-pass** – remove obvious low/high junk first.
2. **arnndn** – heavy noise reduction.
3. **dialoguenhance** – pull out the speech.
4. **speechnorm** – final loudness & dynamics shaping for speech.

You can swap `arnndn` for `afftdn` if you prefer pure FFT denoising.

---

### 5. Visualizing a spectrogram via `ffplay`

You can create a scrolling spectrogram in real time by piping `ffmpeg` into `ffplay` using the `showspectrum` filter. ([Super User][7])

```bash
ffmpeg -i input.wav -filter_complex \
"showspectrum=mode=separate:color=intensity:slide=scroll:scale=cbrt:s=1280x720" \
-f nut - \
| ffplay -autoexit -f nut -
```

Explanation:

* `showspectrum=` – converts audio to a video spectrogram.

  * `mode=separate` – separate channels (L/R).
  * `color=intensity` – brighter = louder.
  * `slide=scroll` – scroll from right to left.
  * `scale=cbrt` – perceptual intensity scaling.
  * `s=1280x720` – resolution of the spectrogram video.
* `-f nut -` – use the NUT container to stream video over stdout.
* `ffplay -f nut -` – read the stream and display it.

You can also use `ffplay` directly:

```bash
ffplay input.wav -af "showspectrum=mode=separate:slide=scroll:scale=cbrt"
```

---

## Part II – Miscellaneous tasks

### 6. Converting to WAV (uncompressed)

#### 6.1 MP3 → WAV (16 kHz mono, suitable for ASR)

Based on the referenced gist. ([Gist][8])

```bash
ffmpeg -i input.mp3 -acodec pcm_s16le -ac 1 -ar 16000 output.wav
```

* `-acodec pcm_s16le` – 16-bit PCM (little-endian), standard uncompressed WAV.
* `-ac 1` – downmix to mono.
* `-ar 16000` – resample to 16 kHz.

#### 6.2 WMA → WAV

```bash
ffmpeg -i input.wma -acodec pcm_s16le -ac 1 -ar 16000 output.wav
```

Same options as above; only the input format changes.

---

### 7. Adjusting the sampling rate

You can set sampling rate directly via `-ar` (simplest) or use the `aresample` filter if you need more control. ([FFmpeg][9])

```bash
# Convert to 16 kHz mono PCM WAV
ffmpeg -i input.wav -ar 16000 -ac 1 -acodec pcm_s16le resampled.wav
```

If you want explicit control via a filter:

```bash
ffmpeg -i input.wav -af "aresample=16000" -acodec pcm_s16le resampled.wav
```

---

### 8. Seeking and cutting audio by timestamps

FFmpeg uses flexible time syntax (e.g., `HH:MM:SS`, `MM:SS`, or seconds). ([FFmpeg][5])

#### 8.1 Extract a segment from 00:01:30 to 00:02:10

```bash
ffmpeg -ss 00:01:30 -i input.wav -to 00:02:10 -c copy cut.wav
```

* `-ss` before `-i` – faster, but less sample-accurate.
* `-to` – stop time (relative to start of input).
* `-c copy` – stream copy (no re-encode); for file-accurate seeking you may omit `-c copy` and let FFmpeg re-encode.

For precise cuts with filters, you can use `atrim`:

```bash
ffmpeg -i input.wav -af "atrim=start=90:end=130,asetpts=N/SR/TB" trimmed.wav
```

* `start=90` – seconds from start.
* `end=130` – seconds from start.
* `asetpts=N/SR/TB` – reset timestamps to start at 0.

---

### 9. Concatenating two audio files

Simplest general method: use the `concat` filter with two inputs. ([FFmpeg][5])

```bash
ffmpeg -i part1.wav -i part2.wav -filter_complex \
"[0:a][1:a]concat=n=2:v=0:a=1[outa]" \
-map "[outa]" merged.wav
```

* `concat=n=2:v=0:a=1` – 2 inputs, no video, 1 audio output.
* All inputs must have the same format (sample rate, channels, sample format). If not, resample beforehand (e.g., `-ar 16000 -ac 1` on each file).

Alternative (for many files): concat demuxer (quick sketch):

```bash
# list.txt
file 'part1.wav'
file 'part2.wav'

ffmpeg -f concat -safe 0 -i list.txt -c copy merged.wav
```

(Works best when formats match exactly and no re-encoding is needed.)

---

## Further reading

* FFmpeg filter documentation (`ffmpeg-filters`) – canonical reference for all filters and options. ([FFmpeg][5])
* SuperUser answer on normalization (covers `volumedetect`, `volume`, `loudnorm`, `speechnorm`). ([Super User][2])
* “Convert mp3 to wave format using ffmpeg” gist – concise example of MP3→16kHz mono WAV. ([Gist][8])
* FFmpeg libav tutorial on GitHub – broader background on codecs, containers, and FFmpeg CLI usage. ([GitHub][10])

[1]: https://ffmpeg.org/ffmpeg-filters.html?utm_source=chatgpt.com "FFmpeg Filters Documentation"
[2]: https://superuser.com/a/971934 "How can I normalize audio using ffmpeg? - Super User"
[3]: https://trac.ffmpeg.org/wiki/AudioVolume?utm_source=chatgpt.com "Audio Volume Manipulation"
[4]: https://www.ffmpeg.org/doxygen/7.1/af__speechnorm_8c_source.html?utm_source=chatgpt.com "libavfilter/af_speechnorm.c Source File"
[5]: https://ffmpeg.org/ffmpeg-all.html?utm_source=chatgpt.com "ffmpeg Documentation"
[6]: https://superuser.com/questions/733061/reduce-background-noise-and-optimize-the-speech-from-an-audio-clip-using-ffmpeg?utm_source=chatgpt.com "Reduce background noise and optimize the speech from ..."
[7]: https://superuser.com/questions/294154/making-sound-spectrum-images-from-command-line?utm_source=chatgpt.com "audio - Making sound spectrum images from command line"
[8]: https://gist.github.com/vunb/7349145?utm_source=chatgpt.com "Convert mp3 to wave format using ffmpeg"
[9]: https://www.ffmpeg.org/ffplay-all.html?utm_source=chatgpt.com "ffplay Documentation"
[10]: https://github.com/leandromoreira/ffmpeg-libav-tutorial?utm_source=chatgpt.com "FFmpeg libav tutorial - learn how media works from basic ..."

