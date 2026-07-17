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

### Responsive, cancellable loading

All heavy builds now run in `napari` `thread_worker`s and only touch napari
layers back on the GUI thread:

- per-gene transcript point-store build,
- label rasterization + label-outline pyramid build,
- image pyramid build,
- Cellpose value-overlay label prep + pyramid build + colour mapping.

Dataset metadata is prepared off the GUI thread too. Image channels put a lazy
preview on screen first and replace it with the materialized pyramid when ready;
cell clicks draw their boundary and a loading panel immediately while transcript
counts and image intensities are calculated in the background. Repeated identical
requests are deduplicated, obsolete work is cooperatively cancelled on dataset
switch/reload, and zarr-heavy work shares a small I/O pool rather than starting
an unbounded number of competing reads.

Inspection hit-testing is also deferred from mouse-down to mouse-release. A
gesture moving more than four canvas pixels is classified as a pan and skips all
Points-layer and cell-polygon picking, so those synchronous queries cannot delay
the first movement of the view.

napari's asynchronous slicer is enabled by default for image and segmentation
layers. Lazy viewport reads happen outside the GUI thread and stale requests are
cancelled as newer camera positions arrive. This improves frame pacing and makes
rapid image zooms less likely to expose unloaded black edge tiles.

Some Qt/VisPy combinations can finish progressive startup with valid layers and
camera extents but a stale blank OpenGL viewport. Because manually resizing a
side dock reliably refreshes all three layout layers, the viewer performs the
same operation imperceptibly after each startup layer batch: the Viewer Controls
dock grows by one pixel for one frame and is restored. A few delayed retries
cover asynchronous slicing and texture upload that settle after layer insertion.

Status/progress updates from worker threads are marshalled to the GUI thread via
Qt signals (`ViewerControlPanel.status_message` / `ViewerControlPanel.progress_message`).
A busy progress bar with a stage label below the tabs shows which stage is
running (image pyramids, cell masks, transcripts, …).

### Materialized pyramids (smooth pan/zoom)

Single-scale images and labels have no stored pyramid, so coarse levels used to
be lazy stride/`coarsen` views over full resolution — reading a coarse tile
forced a full-resolution read (~4^k amplification). Now the viewer builds
**materialized** coarse pyramids, persisted into the zarr as private derived
caches, rechunked to ~1024² tiles:

- images: mean-downsampled (`_napari_compare_imgpyr__<key>__ds<N>`),
- label value overlays: max-pooled so label ids survive
  (`_napari_compare_labelpyr__<key>__ds<N>`),
- label outlines, coverage-downsampled and rechunked to bounded ~1024² display
  tiles. Fractional uint8 coverage keeps nearest-neighbour outlines crisp while
  preventing them from becoming overwhelmingly thick at coarse pyramid levels.
  A square-root opacity curve compensates for the lost coverage at each level,
  keeping ordinary boundaries visible without returning to fully opaque blocks.

Pyramids continue to a ~1024-pixel overview level, including shallow pyramids
provided by upstream stores. That small whole-tissue fallback reduces data and
texture upload at the widest zooms. Existing derived image/outline caches with
the older 4096-pixel stopping point are detected as stale and rebuilt once.

The full-resolution base is reused as level 0 (never duplicated), so a pyramid
adds only a fraction of the base size (~8 GB at 4x downsample for the reference
MERSCOPE element). Builds are one-time and cached; rebuild with
`--overwrite-derived-caches`.

### Bounded transcript memory and constant-time toggles

Transcript ingestion is a two-pass streaming build. The first pass retains only
gene/cell counts; the second fills a fixed render array and, when necessary, uses
an exact uniform sample without first concatenating every coordinate. Gene names
in the exact per-cell index are dictionary encoded with compact integer codes.
The displayed Points layers keep fixed coordinate/color buffers and checkbox
changes update napari's per-point `shown` mask, avoiding repeated concatenate,
copy, and GPU-buffer replacement work.

Cell inspection reuses the exact compact index built during transcript loading,
so a render cap never changes a selected cell's gene counts.

### Reusable dataset sessions

Prepared dataset metadata, transcript arrays, and label display specs are held in
an LRU session cache. Switching back to a recent dataset can restore them without
re-reading/re-grouping. The default RAM budget is 20% of physical memory capped
at 8 GiB; inactive sessions are evicted first.

## Relevant flags

- `--skip-images` / `--skip-cellpose` / `--skip-proseg` / `--skip-transcripts`:
  suppress the *startup* auto-load of that layer type (load it manually from its
  tab afterwards) — useful on low-RAM systems.
- `--image-pyramid-downsample INT` (default 4): downsample step between
  materialized image/label pyramid levels; 2 is smoothest but ~5x larger on disk.
- `--visible-channels "DAPI,PolyT"`: channels visible by default on image load
  (others load hidden/toggleable). Default: a DAPI-like channel, else the first.
- `--overwrite-derived-caches`: rebuild density/outline/image/label pyramid
  caches even when matching cache metadata exists.
- `--session-cache-gb FLOAT`: RAM budget for inactive prepared dataset sessions;
  `0` disables cross-dataset reuse.
- `--background-io-workers INT` (default 2): maximum concurrent zarr-heavy
  builders. Increase only if profiling shows the storage device is underused.
- `--disable-async-slicing`: compatibility fallback for napari's background
  image/segmentation viewport slicing.
- `--gl-core-profile` (experimental): request an OpenGL 4.1 core context instead
  of the macOS default legacy 2.1 context; see below.

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

3. **Persisted label pyramid for outlines.** The label *outline* cache is
   already materialized, but building it from a single-scale 12-gigapixel label
   array streams the full labels once (~minutes, one-time, threaded). If that
   first build is painful, a shared materialized base-label pyramid could feed
   both the outline and value-overlay caches so the full labels are read once
   rather than once per derived cache.

## Measuring transcript-store changes

The synthetic benchmark reports build time, retained compact-array size, and RSS
change. Its defaults exercise one million source points; use the second command
for a larger stress run:

```bash
python scripts/benchmark_transcript_store.py --points 1000000
python scripts/benchmark_transcript_store.py --points 10000000 --render-cap 2000000
```
