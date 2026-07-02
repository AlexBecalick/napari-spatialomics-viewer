# Viewer performance notes

This documents the performance work done on the viewer, the derived caches and
flags it introduced, and the remaining optional ideas to revisit if needed.

## Summary of what was optimized

The viewer targets very large SpatialData stores (e.g. MERSCOPE mosaics of
~105k x 115k px per channel, tens of millions of transcripts). Two classes of
slowness were addressed:

1. **UI freezes** — heavy builds used to run on the Qt GUI thread.
2. **Laggy pan/zoom** — coarse zoom levels were computed lazily from
   full-resolution data, so every pan re-read gigabytes.

### Threading (no more freezes)

All heavy builds now run in `napari` `thread_worker`s and only touch napari
layers back on the GUI thread:

- transcript density pyramid + in-memory point index build,
- label rasterization + label-outline pyramid build,
- image pyramid build,
- Cellpose value-overlay label prep + pyramid build + colour mapping.

Status updates from worker threads are marshalled to the GUI thread via a Qt
signal (`DatasetSwitcherWidget.status_message`).

### Materialized pyramids (smooth pan/zoom)

Single-scale images and labels have no stored pyramid, so coarse levels used to
be lazy stride/`coarsen` views over full resolution — reading a coarse tile
forced a full-resolution read (~4^k amplification). Now the viewer builds
**materialized** coarse pyramids, persisted into the zarr as private derived
caches, rechunked to ~1024² tiles:

- images: mean-downsampled (`_napari_compare_imgpyr__<key>__ds<N>`),
- label value overlays: max-pooled so label ids survive
  (`_napari_compare_labelpyr__<key>__ds<N>`),
- transcript density and label outlines were already materialized caches.

The full-resolution base is reused as level 0 (never duplicated), so a pyramid
adds only a fraction of the base size (~8 GB at 4x downsample for the reference
MERSCOPE element). Builds are one-time and cached; rebuild with
`--overwrite-derived-caches`.

Density build also switched from `np.add.at` to `np.bincount` (10-50x faster),
producing bit-identical output.

## Relevant flags

- `--transcripts-mode {streamed,capped}` (default `streamed`): stream the full
  transcript density + viewport detail on load instead of a fixed capped sample.
- `--transcript-density-contrast-max FLOAT` (default 8.0): upper contrast limit
  for transcript density layers; lower = brighter.
- `--image-pyramid-downsample INT` (default 4): downsample step between
  materialized image/label pyramid levels; 2 is smoothest but ~5x larger on disk.
- `--visible-channels "DAPI,PolyT"`: channels visible by default on image load
  (others load hidden/toggleable). Default: a DAPI-like channel, else the first.
- `--overwrite-derived-caches`: rebuild density/outline/image/label pyramid
  caches even when matching cache metadata exists.
- `--gl-core-profile` (experimental): request an OpenGL 4.1 core context instead
  of the macOS default legacy 2.1 context; see below.

The "Unload Transcripts" button in the dock fully tears down the streamed
transcript state so pan/zoom cannot bring the layers back (previously, deleting
the layers by hand left the camera callbacks alive and they reappeared).

## Environment / GPU

napari renders via VisPy -> OpenGL. On Apple Silicon this runs on Apple's legacy
OpenGL-on-Metal path (observed `renderer=Apple M4 Max ... version=2.1 Metal`).
This is GPU-accelerated but a legacy GL profile. It was not the bottleneck once
pyramids were materialized. Startup logs the machine arch, Qt binding, VisPy
backend, and (deferred until the GL context is live) the OpenGL
renderer/vendor/version — check these first when triaging.

## Optional ideas to revisit (not yet done)

These were intentionally left out because there was no demonstrated need or they
carry regression risk. Revisit if a specific interaction still feels heavy.

1. **OpenGL 4.1 core profile.** Try `--gl-core-profile` and confirm via the
   `OpenGL: version=` startup log that it reports 4.1. A/B test pan/zoom feel.
   Some VisPy visuals assume legacy GL, so if anything renders wrong, drop the
   flag. Expected benefit is small now that I/O is fixed.

2. **PySide6 / PyQt6 binding.** PyQt5 is currently used. PySide6 or PyQt6 track
   Apple Silicon more actively and may negotiate a better GL context by default.
   This is an environment change (reinstall + retest), so measure against the
   current baseline before committing. `qtpy` already abstracts the binding, so
   the code should not need changes.

3. **Points layer tuning.** The streamed transcript *detail points* are capped
   (default 300k) and currently render fine. If they ever feel heavy at extreme
   counts, set `layer.antialiasing = 0` and drop per-point borders on the
   assigned/unassigned point layers (`_add_points_layer`). Note this slightly
   changes point appearance (hard-edged, no border), so it is a deliberate
   trade-off, not a free win.

4. **Persisted label pyramid for outlines.** The label *outline* cache is
   already materialized, but building it from a single-scale 12-gigapixel label
   array streams the full labels once (~minutes, one-time, threaded). If that
   first build is painful, a shared materialized base-label pyramid could feed
   both the outline and value-overlay caches so the full labels are read once
   rather than once per derived cache.
