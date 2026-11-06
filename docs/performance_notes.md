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
- `--gene-max-render-points INT`: per platform. MERSCOPE/Xenium: cap on the
  points rendered (default 40,000,000; 0 = no cap). Atera: the whole-slide
  sample budget; unset = AUTO (half the RAM over 200 B per point, about
  172M on a 64 GB Mac), 0 = every transcript. The memory guard clamps any
  Atera budget to 85% of RAM, and a MemoryError halves it.
- `--atera-transcript-mode {full,lazy}` (default `full`): `full` streams every
  file into a nested whole-slide sample; `lazy` shows a 5 um density image and
  loads only the ticked genes.
- `--atera-viewport-max-points INT` (default 20,000,000): cap on the
  transcripts streamed for the visible field at full density.

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

## Atera lazy transcript engine

Atera stores hold about 1.4e9 transcripts in ~400 per-tile parquet files, cut
by gene within each tile and sorted by feature code. They are never loaded as
one table. `atera_transcripts.py` (pure numpy/pyarrow, no Qt) validates
MerXen's `merxen_atera_points_index.json` against the parquet footers. It then
serves batched gene reads (each row group is read once per call), whole-tile
and window reads, and a one-pass overview: exact per-feature totals and
assigned counts, a 5 um density raster over real genes, and the exact
code-range map (`code -> (row group, row_start, row_end)`).

The overview and the code-range map are cached outside the store, under
`atera-overview-v1/` and `atera-codemap-v1/` in the transcript cache
directory. They are keyed by the index JSON content and by each file's size
and mtime. ctime is not part of the key, because MerXen's hardlink churn
changes it. Until a code-range map exists, gene reads prune row groups with
the `feature_name` string statistics, which over-read because control probes
sort among the gene names.

Measured on the real store: 409 files, 1,409,601,228 rows, 24,677-entry
shared dictionary. Warm OS page cache, 16-core Mac, pyarrow 24, 4-thread
pool, `use_threads=False` per read.

| Operation | Time | Notes |
| --- | --- | --- |
| Index load and validation | 0.20 s | footers plus one dictionary batch per file |
| Code-range map build | 1.4 s | 599,029 runs (about 6 MB) |
| GFAP, string statistics | 0.20 s | 61 row groups, 41.8M rows decoded, 1.24M kept |
| GFAP, code-range map | 0.09 s | 32 row groups, 27.9M decoded |
| 24 random genes, code-range map | 1.5 s | 477 distinct row groups, 456M decoded |
| Largest tile (2,1) | 0.5 s | 87.2M rows |
| 1 mm2 window | 0.36 s | 70M decoded, 20M kept |
| Overview pass (5 um bins) | 8.5-9.4 s | 2610 x 1810 raster, about 1.2 GB peak RSS |
| Overview cache save / load | 0.17 s / 0.32 s | load includes index validation |

With `use_threads=True` on each read, the overview pass takes 6.8 s; with an
8-thread pool it takes 5.9 s. Both are left off, so lazy reads stay bounded
next to the viewer's image and label loaders.

Cold-cache timings were taken without `sudo purge`. For each operation the
points parquet (24.1 GB) was copied with `ditto --nocache` to a fresh scratch
directory on the same internal SSD. `mincore` confirmed that only 0.007 GB of
the copy was resident before each run. Each run is a new process: it loads the
index, then does one operation. The gene reads use a code-range map built on
another cold copy. The "Read" column is how much of the copy was resident
afterwards, so it is about what the operation read from disk.

| Operation (cold) | Time | Read | Warm |
| --- | --- | --- | --- |
| Index load and validation | 0.50-1.02 s | footers | 0.20 s |
| Code-range map build | 1.56 s | 0.27 GB | 1.4 s |
| GFAP, string statistics | 0.24 s | 0.45 GB | 0.20 s |
| GFAP, code-range map | 0.12 s | 0.32 GB | 0.09 s |
| 24 random genes, code-range map | 1.67 s | 3.9 GB | 1.5 s |
| Largest tile (2,1) | 0.47 s | 0.86 GB | 0.5 s |
| 1 mm2 window | 0.39 s | 0.70 GB | 0.36 s |
| Overview pass (5 um bins) | 8.8 s | 9.9 GB | 8.5-9.4 s |

On this internal NVMe SSD, about 1.1 GB/s or more, the passes are bound by
decoding rather than I/O. A cold first open costs only the index footers,
about 0.5-0.8 s more than a warm one. A store on a slower external or network
drive will be I/O bound: the overview reads about 10 GB, so at 100-200 MB/s
it would take about 1-2 minutes. That is why the overview runs in a
background worker and is cached outside the store.

## Atera transcripts in the viewer (M7)

An Atera store opens in one of two transcript modes; both are set from the
CLI and from the Gene inspector ("Atera transcripts" mode, "Point budget" with
a Reload button, "Viewport cap" applied at once).

`full` (default) streams every parquet file once, in a background worker on
the engine's 4-thread pool, and keeps a deterministic hash sample: every row
whose splitmix64 key of (file, row) is below `budget / total`. Samples nest
(a smaller budget is a subset of a larger one) and a budget at or above the
store total keeps every row. The sample becomes an ordinary
`GenePointStore` (vectorised build, uint16 radix sort), so every
gene-inspector feature works as for Xenium/MERSCOPE. The same pass yields the
overview and code-range map. The sample is cached under `atera-sample-v1/`
(the largest one per store, 18 B/row, up to 400M rows); a smaller budget
filters it by key without touching the store. While the budget leaves rows
out, a persistent `Genes | viewport` layer tops the visible field up to full
density with exactly the rows the sample skipped (ticked genes, or every
real gene under the All-transcript overview). It reads a window 50% larger
than the view, only when the density raster estimates at most the viewport
cap, single-flight and latest-wins.

`lazy` shows the 5 um density image of every real gene, loads ticked genes at
full resolution with one batched `read_genes` per debounce window, and uses the
viewport layer for the other genes under the overview.

Measured on the real store (1,409,601,228 rows; 64 GB / 16-core M4 Max; warm
OS page cache; napari 0.7.0 in a visible 1500x950 window, images skipped,
cell outlines loaded). "Cold" is a transcript-cache miss: the whole store is
streamed. "Warm" is a cache hit (the cached sample is at least as large).
Pan times are the median and p90 of 30 synchronous repaints while the camera
moves by 2% of the view.

Headless load (engine pass, sample and `GenePointStore`, no layers):

| Budget | Points | Cold | Warm | Peak RSS cold / warm |
| --- | --- | --- | --- | --- |
| 40M | 40.0M | 15.3 s | 1.8 s | 5.0 / 3.3 GB |
| 100M | 100.0M | 15.1 s | 3.3 s | 8.7 / 6.3 GB |
| 200M | 200.0M | 19.0 s | 6.6 s | 14.6 / 11.6 GB |
| AUTO | 171.8M | 17.4 s | 5.6 s | 12.2 / 10.1 GB |

In the viewer (load = process start to layers ready, including napari start-up
and the cell outlines; one run per budget):

| Budget | Load | RSS loaded | Peak RSS | Pan, nothing ticked | Tick a gene | Pan, 3 genes | All-transcript overview on | Pan, overview | 250 um field at full density | Pan in that field |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 40M (warm) | 10.2 s | 8.9 GB | 14.8 GB | 8 / 9 ms | 0.8-1.0 s | 9 / 18 ms | 13.1 s | 94 / 212 ms | 2.9 s, 7.3M points | 45 / 129 ms |
| 100M (warm) | 15.5 s | 17.7 GB | 23.7 GB | 8 / 8 ms | 1.4-3.1 s | 19 / 26 ms | 28.2 s | 244 / 394 ms | 3.0 s, 6.9M points | 185 / 307 ms |
| 200M (cold) | 34.8 s | 14.6 GB | 21.8 GB | 8 / 11 ms | 1.5-6.0 s | 20 / 32 ms | 38.1 s | 954 / 1131 ms | 4.0 s, 6.4M points | 184 / 566 ms |
| AUTO 171.8M (warm) | 25.2 s | 27.3 GB | 35.1 GB | 8 / 9 ms | 1.5-4.4 s | 21 / 35 ms | 29.5 s | 921 / 1209 ms | 3.9 s, 6.6M points | 313 / 492 ms |
| lazy | 5.1 s | 1.9 GB | 9.6 GB | 4 / 13 ms | async: 0.04 s to tick, 0.6 s to read and group 3 genes (7.5M points) | 40 / 50 ms | 2.5 s | 20 / 26 ms | 2.7 s, 7.5M points | 13 / 30 ms |

RSS readings on macOS move with memory compression, so the RSS columns are
indicative only. Between the 40M and AUTO runs the loaded RSS grows by about
140 B per point (store and napari layers); the vertex buffer adds about 56 B
per shown point. That is where the AUTO formula's 200 B per point comes from.
Every transcript (1.41B points, about 280 GB) does not fit on this machine,
and the memory guard clamps it to about 292M.

What this means in practice:
- Ticked genes stay fluid at every budget (about 10-35 ms per frame), and
  toggling a gene costs about 1-6 s at 100-200M because its symbol group's
  colours are re-uploaded.
- The All-transcript overview draws every loaded point. It pans at about
  10 fps at 40M, 4 fps at 100M and about 1 fps at AUTO; turning it on
  uploads every point (13-38 s). For a fluid all-gene view, dial the budget
  back to about 40M, or use lazy mode, where the overview is the density image.
- Creating the layers for a large sample runs on the Qt thread: about 10-15 s
  at AUTO, during which the window does not respond. The read, sample and
  store build stay in the background. This, and the 1 fps All-transcript
  overview at AUTO, are open against the M7 acceptance ("nothing blocks image
  or mask loads", "a pan storm stays responsive"); the queued M10 splits the
  layer creation into per-group chunks on a QTimer chain with a determinate
  progress bar, and caps the All-transcript overview at a nested 40M
  sub-sample. Until then, a 40M budget or the lazy mode avoids both.
- A MemoryError while creating the layers (not only while sampling) tears the
  partial layers down and reloads at half the budget, saying so in the
  status bar.
- Lazy mode keeps the rows of genes read at full resolution in a 1 GiB LRU
  (about 14-22 B per row). Only ticked genes are napari points: unticking a
  gene removes its points from the layers, a re-tick restores them from the
  LRU without a read, and the LRU is dropped when the inspector is torn down.
- The viewport needs about 0.5 s to read a 500 um window and about 2 s to
  swap 7M points into its layer, with one vispy upload. Pans inside the
  padded window reuse its rows.

Not measured: a store on a slower or external drive (the cold pass is about
10 GB of reads), and GPUs other than this Apple M4 Max.

## Atera cell inspector (M8)

A cell click on an Atera store never builds a whole-store transcript index
(about 6.4 GB) and never runs a dask scan. It paints in two phases:

1. One row of `tables/table_original/X`, read through `indptr` (the var names
   and the `instance_id -> row` lookup are cached per table). These are gene
   transcripts: `X` holds no control features.
2. The cell's exact assigned transcripts. Each of the cell's files is read
   in two steps. First only `original_assignment` is decoded and the matching
   rows are found. Then `x`, `y` and `feature_name` are read for just the row
   groups with hits, and only those rows are kept. The image statistics and
   the stored annotation arrive with this phase. Control features are counted
   separately from the genes.

The read covers only the tiles holding the cell's transcripts, so the first
question was whether cells span tiles. Measured over all 409 files:

- 91,429 cells have assigned transcripts.
- 907 of them (about 1%) have them in more than one tile, at most 4.
- Choosing tiles by the cell's bounding box would still have found all of
  them (0 misses), because tile boxes are tile-wide.

To be safe, the overview pass builds an exact cell -> tile map
(`CellTileMap`). It is cached inside the overview entry and as a
`atera-celltiles-v1/` sidecar. An overview cached before M8 gets the map once,
from an assignment-only pass: 3.2 s warm, then 0.03 s from the sidecar.
Collecting the ids adds about 1.5-2 s to a warm overview pass (8.2 s
without, 9.7-10.3 s with). A read that cannot be exact returns nothing, and
the inspector keeps the table counts with a note. That happens when the index
is not validated, or when a file's size or mtime changed since the engine
opened.

Real store, warm unless noted:

| Step | Cell 1001 | Others |
| --- | --- | --- |
| Table row | 17,199 transcripts, 5,940 genes; 37 ms for the first call (lookup build), then 8-10 ms | 7-10 ms |
| Exact read | 17,202 rows (3 controls), 22 files, 85 row groups, 0.30-0.35 s | 0.26-0.35 s |
| Exact read, cold | 0.38 s (1.3 GB of files copied with `ditto --nocache`) | cell 45000: 0.32 s |
| Real viewer, click to table paint | 0.40 s (0.30 s of it is the first overlay build) | cell 177: 0.18 s |
| Real viewer, click to exact rows | 0.85 s | 0.77 s |

Cell 177 straddles two tiles.

Presentation is capped at WTA scale. Below the caps the output is unchanged:

- The pie shows the top 24 genes plus one neutral "Other" slice.
- The gene list shows 200 rows plus an "N more genes" footer.
- Each cell draws at most 2,000 transcript links, evenly spaced and
  deterministic. The summary says when links are capped.

Not in this effort: vendor Azimuth as a second cell-type/UMAP source
(`vendor_azimuth_*`, `X_umap_vendor`). Also not done: reusing the viewport's
rows for a cell inside it, because the sample drops cell ids and the per-cell
read is already about 0.3 s.
