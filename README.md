# napari-compare-xenium-merscope

Napari-based viewer for visualising MERSCOPE and Xenium SpatialData `.zarr`
outputs, either as one standalone dataset or as a simultaneously displayed,
pre-aligned pair.

The viewer opens a single Napari window with a tabbed control panel (tabs across
the top of the right-hand dock): **Gene inspector**, **Cell segmentation**,
**Cell type labels**, **Per cell statistics**, **Draw tissue annotations**,
**Images**, **Dataset loader**, **UMAP**, and **Zarr metadata**. With
a supplied dataset it automatically loads all image channels, the Cellpose, ProSeg and
ProSeg hybrid cell segmentations, and all transcripts (rendered as per-gene coloured
points). A busy progress bar with a stage label below the tabs shows what is
loading (image pyramids, cell masks, transcripts, …).

Standalone mode keeps one dataset active at a time. Image channels appear first
as lazy previews while optimized pyramids build in the background, missing
segmentation rasters can be generated and saved to the store, and recent
prepared sessions remain in an LRU cache for fast return visits. Paired mode is
a strict read-only view of already materialized MerXen alignment output; it does
not build registration products or write derived artifacts into either source
store. Prepared transcripts can still use the viewer's external user cache.

## Install

Begin by starting in the root folder of the repository and create a new conda environment from the provided environment file:

```bash
conda env create -f environment.yml
conda activate napari-compare-xenium-merscope
```

For updating an existing Python environment after pulling any new changes to the repo:

```bash
pip install -e .
```

### macOS application bundle

On macOS, installation also provides the tooling used to create a lightweight
application bundle at `~/Applications/Napari Compare Xenium MERSCOPE.app`. The
first `napari-compare-xenium-merscope` launch creates or refreshes this bundle
for the active Python environment, then runs the viewer through its native
Mach-O launcher. The bundle contains an `Info.plist` and `.icns` icon, allowing
the macOS Dock and third-party Dock replacements such as Sidebar to identify the
viewer as an application rather than as a generic Python console script.

The bundle uses the installed environment in place; it does not duplicate the
napari or SpatialData dependencies. Rebuild it explicitly at any time with:

```bash
napari-compare-xenium-merscope-install-macos-app
```

Launching the `.app` from Finder, the Dock, or Sidebar opens with no dataset
loaded and selects the **Dataset loader** tab. Choose one of the two paired
layouts or open a standalone store there by selecting each dataset's
`spatialdata.zarr` folder. The empty canvas points toward these controls and
keeps napari's rotating **Did you know?** tip. The loader retains the ten most
recently opened entries: a standalone store is one entry, while a MERSCOPE and
Xenium pair is saved as one logical entry together with its view mode. Select an
entry and click **Open selected recent dataset** (or double-click it) to reopen
it. Dataset paths passed explicitly on the command line still load immediately.

### Self-contained desktop installers

Self-contained application installers are built by the
`Package desktop applications` GitHub Actions workflow. They include Python,
napari, Qt, SpatialData, and this viewer, so end users do not need to create a
conda environment or run `pip install`.

- macOS produces separate Apple Silicon (`arm64`) and Intel (`x86_64`) `.dmg`
  files. Each contains a native `.app` with the project icon and an Applications
  shortcut for drag-and-drop installation.
- Windows produces an x86-64 Setup `.exe`. It installs for the current user,
  adds a Start Menu entry, and optionally creates a desktop shortcut.
- Linux produces an amd64 `.deb`. It installs the application under `/opt`, a
  command under `/usr/bin`, and a desktop-menu entry with the project icon.

Run the workflow manually from the repository's **Actions** page to obtain its
downloadable artifacts. It does not create a GitHub Release or publish the
installers automatically. Native packages are built and smoke-tested on their
target operating systems; PyInstaller is not a cross-compiler.

Pushing a version tag such as `v0.3.0` builds these same installers and publishes
a GitHub Release with checksums and release notes. See [RELEASING.md](RELEASING.md).

For local package builds, first install the pinned build environment:

```bash
python -m pip install -r packaging/requirements-build.txt
```

Then run `bash packaging/macos/build_macos.sh` on macOS,
`packaging/windows/build_windows.ps1` from PowerShell on Windows, or
`bash packaging/linux/build_linux.sh` on Debian/Ubuntu Linux. The macOS build
script can use `MACOS_SIGNING_IDENTITY` for Developer ID signing and
`MACOS_NOTARY_PROFILE` for an existing `notarytool` keychain profile. Without
project-owned signing credentials, macOS and Windows downloads may show an
unknown-developer or unknown-publisher warning.

## Then Run

Launch without a dataset to choose one in the viewer:

```bash
napari-compare-xenium-merscope
```

Or supply both dataset paths and opt into a paired layout explicitly:

```bash
napari-compare-xenium-merscope \
  --merscope-zarr /path/to/merscope.zarr \
  --xenium-zarr /path/to/xenium.zarr \
  --paired-view side-by-side
```

Open the same pair as a stacked overlay with:

```bash
napari-compare-xenium-merscope \
  --merscope-zarr /path/to/merscope.zarr \
  --xenium-zarr /path/to/xenium.zarr \
  --paired-view stacked-overlay
```

`--paired-view` is used only when both dataset paths are supplied; one path
continues to open the corresponding store in standalone mode. Supplying both
paths without `--paired-view` preserves the original behavior: both stores are
available, but only `--initial-dataset` is displayed at startup.

You can also open just one dataset:

```bash
napari-compare-xenium-merscope --merscope-zarr /path/to/merscope.zarr
napari-compare-xenium-merscope --xenium-zarr /path/to/xenium.zarr
```

### Paired viewing

The **Dataset loader** offers two explicit choices: **Load new paired dataset
side-by-side** and **Load new paired dataset stacked overlay**. Both choices ask
for the MERSCOPE store first and the Xenium store second, then keep both loaded
in the same Napari window with one shared control panel.

- **Side-by-side** places MERSCOPE on the left and Xenium on the right. Pan,
  zoom, and camera navigation are linked, and changing the visibility of an
  image, segmentation, transcript, or other logical layer also changes its
  counterpart. Cell and transcript clicks are resolved at the same aligned
  coordinate and highlighted in both datasets. The Images, Cell segmentation,
  Gene inspector, and shared cell-type controls load, unload, or update both
  platforms together.
- **Stacked overlay** draws both platforms in their common coordinate system on
  one canvas. Every underlying layer has an unambiguous platform-qualified name,
  such as `MERSCOPE | Image | DAPI` or `XENIUM | Segmentation | Cellpose`, and
  each platform layer can be shown or hidden independently.

Use **Switch to side-by-side view** or **Switch to Stacked overlay view** to
change presentation without reopening the stores or rebuilding the layers. The
independent visibility choices made in stacked mode are restored when returning
to it. Per-cell statistic overlays and tissue-annotation creation/export are
unavailable while a pair is active because those tools currently have
single-dataset semantics. The UMAP view works with pairs; see
[Paired UMAPs](#paired-umaps).

Paired viewing is deliberately a strict, read-only consumer of MerXen v2
materialized data. The moving MERSCOPE store must contain a complete version-2
`merxen_alignment` manifest, and the fixed Xenium store must contain its matching
`merxen_alignment_pair_reference`. Their pair identity, transform, source
fingerprint, common coordinate system, fixed grid, schemas, and all declared
artifacts must agree and be present. The viewer does not calculate a
registration, rasterize shapes, generate image/label pyramids or outlines, or
write either store in paired mode. If validation reports that a pair is missing,
incomplete, stale, or mismatched, rerun the upstream MerXen materialization and
then reopen both stores.

### Default startup and low-RAM flags

In either workflow, the viewer loads all available image channels, the Cellpose,
ProSeg and ProSeg hybrid segmentations (shown as lazy label outlines), and all
transcripts as per-gene points by default. Per-cell statistic overlays are **not** loaded
automatically and are unavailable in paired mode. Missing segmentations are
skipped. Outline colours match between MERSCOPE and Xenium: **Cellpose green**,
**ProSeg hybrid magenta**, and **ProSeg cyan**.

On low-memory systems you can suppress the startup load of any layer type; each
`--skip-*` flag only skips the *automatic* load — you can still load that layer
manually from its tab.

```bash
napari-compare-xenium-merscope \
  --merscope-zarr /path/to/merscope.zarr \
  --xenium-zarr /path/to/xenium.zarr \
  --skip-images \       # don't auto-load images (load from the Images tab)
  --skip-cellpose \     # don't auto-load the Cellpose mask
  --skip-proseg \       # don't auto-load the ProSeg mask
  --skip-proseg-hybrid \ # don't auto-load the ProSeg hybrid mask
  --skip-transcripts    # don't auto-load transcripts (load from the Gene inspector tab)
```

### Tabs

The tab buttons across the top have a white background and bold black text so
they stand out as the main controls, and they wrap onto extra rows on a narrow
dock so every full name stays readable.

- **Gene inspector** — the transcript view (see below). **Load / reload
  transcripts** restores or builds the per-gene points; **Unload transcripts**
  frees them.
  With a pair active, these actions and the gene filters apply to both platforms.
- **Cell segmentation** — a list of every segmentation key; currently-loaded
  segmentations are shown in **green**. Select one or more and use **Load
  selected cell segmentation** / **Unload selected cell segmentation**. Cellpose
  and ProSeg load automatically at startup. In paired mode, each logical key
  controls its MERSCOPE and Xenium layers together.
- **Per cell statistics** — Channel / Statistic / Colormap dropdowns plus
  **Load / Unload per-cell statistic overlay** (MERSCOPE Cellpose
  quantification). This tab is unavailable in paired mode.
- **Draw tissue annotations** — the cortical-depth drawing tools (see below).
  Selecting this tab automatically expands napari's layer controls so the active
  drawing layer can be edited. Creation, validation, and export controls are
  unavailable in paired mode.
- **Images** — a list of every individual image **channel** (across all image
  elements, for both MERSCOPE and Xenium); currently-loaded channels are shown in
  **green**. Use **Load selected image(s)**, **Load all images**, and **Unload
  selected image(s)**. In paired mode, a channel is one logical slot whose
  available MERSCOPE and Xenium layers are controlled together.
- **Dataset loader** — the MERSCOPE/XENIUM switcher and **Reload Dataset** for a
  standalone session; paired loads disable those single-dataset controls. Use
  **Load new paired dataset side-by-side** or **Load new paired dataset stacked
  overlay** (browse for a MERSCOPE store then a Xenium store), or **Load new
  standalone MERSCOPE / Xenium dataset** (browse for a single store). While a
  pair is active, use the two **Switch to... view** buttons to change layouts.
  Each browser expects the store's `spatialdata.zarr` folder. **Recently viewed**
  lists up to ten logical entries and persists between launches.
- **UMAP** — **Open UMAP window** shows the stored UMAP of the cells in an
  optional floating window, with **Cell type source**, **Colour by**, and point
  size controls, plus **Draw region of interest** for lassoing cells in either
  view (see [UMAP view](#umap-view)). With a pair active it shows both
  platforms' UMAPs, one above the other.

The left dock is simplified for the spatial-transcriptomics workflow. **Layer
controls** starts collapsed and can be expanded or collapsed with the arrow in
its title. The create-points, create-shapes, create-labels, console, 2D/3D,
axis, and grid buttons are hidden. Delete remains available in standalone mode
but is disabled while a paired grid is active; use the shared unload controls
instead. **Reset view to original state** remains available in both modes.

In standalone mode, segmentations display as memory-efficient label outlines.
If a matching label element is already present in the SpatialData store, it is
loaded lazily. If it is missing, the viewer rasterizes the selected polygon
layer chunk by chunk, saves it back to `labels/<shape-key>_labels` in the zarr,
and then displays the label outlines as a lazy multiscale image. Future
standalone runs reuse the saved label layer and only rebuild the lightweight
outline view. Paired mode only reads the materialized labels or outlines named
by the validated MerXen contract and never takes this cache-generation path.

Clicking inside a cell mask highlights that cell (repeated clicks accumulate
highlights). This is gated on the **ProSeg** (cell-inspector) layer's visibility:
hiding that layer in the napari layer list disables click-to-highlight and clears
any current highlights, and showing it again re-enables clicking.

The transcript renderer still uses one real napari Points layer for each marker
symbol behind the scenes, but those implementation layers are hidden from the
layer list. A standalone dataset is presented as one **Genes** row; a paired
dataset is presented as separate **MERSCOPE Genes** and **XENIUM Genes** rows.
Their visibility controls are linked in side-by-side mode and independent in
stacked overlay mode. Shared genes use the same marker and colour in both
platforms. The aggregate rows sit immediately beneath the final visible native
layer. The real gene-layer blocks remain pinned to the bottom of napari's model,
so transcript points sit *below* image and mask layers; toggle image visibility
if you need the spots in front.

The label-cache generation options below apply to standalone mode only. They do
not repair or add files to a paired dataset:

```bash
# Rebuild standalone cached labels instead of reusing existing labels.
napari-compare-xenium-merscope ... --overwrite-labels

# Use smaller chunks while generating standalone labels.
napari-compare-xenium-merscope ... --label-chunk-size 1024
```

Napari's label-contour display width can be adjusted where applicable. This is
a display option and does not create or modify materialized paired outlines:

```bash
# Thicker label outlines in napari.
napari-compare-xenium-merscope ... --label-contour-width 2
```

## Transcripts (Gene inspector)

Transcripts render as real points coloured by gene — not a density raster or a
viewport-sampled subset — because a typical section (~10-20M transcripts) fits in
memory as points. The **Gene inspector** tab loads the full gene panel for the
current dataset alphabetically and gives every gene a unique **colour + marker
shape** shown as a large icon beside its name. Assigned and unassigned transcripts
are shown by default; control/blank probes are hidden. Use **Load / reload
transcripts** to restore or build the panel and **Unload transcripts** to free the
points.

The first transcript load for a dataset builds an external persistent cache;
later loads restore that prepared point store instead of streaming the complete
transcript table twice. Cached coordinates and cell-assignment indexes are
memory-mapped, while display colours remain mutable. The cache is automatically
invalidated when the source point files, selected columns, render cap, random
seed, cell-index option, or marker reference changes. It never writes into the
SpatialData zarr (including in paired mode).

By default the cache is stored in the platform user-cache directory: under
`~/Library/Caches/napari-compare-xenium-merscope/transcripts` on macOS,
`$XDG_CACHE_HOME/napari-compare-xenium-merscope/transcripts` (or
`~/.cache/...`) on Linux, and
`%LOCALAPPDATA%/napari-compare-xenium-merscope/transcripts` on Windows. Set
`NAPARI_COMPARE_TRANSCRIPT_CACHE_DIR` to use another external directory. Delete
that directory to force all transcript caches to be rebuilt.

### pciSeq assignments from MerXen

MerXen's pciSeq finalization publishes separately selectable cell-type and
transcript-assignment sources. Choose a **pciSeq** source for the segmentation
used in the fit; the original clustering and native transcript assignments stay
available. A paired session can display a pciSeq source on just the platform
where it exists, with the other side marked unavailable.

pciSeq cell labels use the all-cell annotation table, including cells excluded
by clustering QC. Its UMAP uses the original clustered subset: a spatially
labelled cell can have no embedding point. The model's **Zero** expression class
is retained as a distinct label, while cells excluded from fitting have no
inferred type. These results are estimates, not independent validation of the
reference labels.

Transcript assignments join the original measured positions by exact transcript
IDs and canonical mask IDs. The viewer never displays the raster-frame
coordinates from pciSeq's CSV export. Source selection switches the transcript
store, its per-cell lookup, and the matching segmentation together. The cache
includes the selected source and analysis identity, so switching assignment
methods cannot restore an index belonging to another result. The assignment
summary distinguishes modelled background from transcripts that were not
modelled, such as genes absent from the reference.

For the moving platform in a paired session, the viewer joins those assignments
onto the store's materialized aligned transcript positions by transcript ID.
This preserves nonrigid alignment without changing the native pciSeq points.
Missing, duplicate, or gene-conflicting IDs are rejected rather than joined to
another transcript.

For an older completed MerXen run, publish its exports once with:

```bash
merxen publish-pciseq-viewer --config pciseq_config.json --results-dir pciseq_results
```

The viewer reads these results from the SpatialData store and does not require
pciSeq or the reference expression matrix in its own environment.

### Transcript controls

Genes are grouped by marker shape into at most 14 napari Points layers, so per-gene
toggles stay fast even with hundreds of genes. Each layer keeps fixed point
buffers and toggles update a per-point visibility mask, avoiding large array and
GPU-buffer rebuilds. Transcript ingestion is bounded and streaming; if the render
cap samples the visible points, cell-click gene summaries remain exact through a
separate compact dictionary-encoded index. The panel offers:

- a per-gene checkbox (with the colour/shape marker) and a **Show all genes** toggle,
- a **Spot size** slider (world microns; spots keep a real size and scale with zoom),
- **Hide background (unassigned) spots** to drop transcripts not assigned to a cell,
- **Show control / blank probes** to include negative-control codewords (hidden by
  default), and a filter box to search the list.

### Grouping genes by cell type

If the SpatialData store carries a **cell-type marker reference** — a
`cell_type_marker_reference` table element (or `broad_cell_type` / `fine_cell_type`
columns / a `cell_type_marker_reference` entry in a table's `uns`) mapping each
non-control gene to a broad and fine cell type — the Gene inspector uses it to
group the list. **Order genes by** offers three modes:

- **Broad cell type** (default when a reference is present): genes are grouped
  under broad-type headings in alphabetical order. Each broad type gets its own
  distinct colour; genes within a group share that colour as slightly different
  shades (with different marker shapes) so they stay individually distinguishable.
- **Fine cell type**: genes are grouped by fine subtype, ordered by broad then
  fine. Fine subtypes of the same broad type get nearby hues (and so read as
  related), while unrelated broad types stay far apart in colour.
- **A–Z**: one flat, alphabetical list that keeps whichever colours (broad or
  fine) were last applied and annotates each gene with its broad / fine type.

Marker shapes are stable across all three modes, so only the colours change when
switching — the same transcript keeps its shape. Without a reference the panel
falls back to the flat alphabetical rainbow (only **A–Z** is available).

Reference discovery is panel-based rather than sample-name-based. The viewer
checks, in order, an embedded `cell_type_marker_reference` table, JSON/CSV
sidecars beside the store (and one directory above it), and the bundled
catalogue in `src/napari_compare_xenium_merscope/resources/`. Bundled entries
are accepted only when their fingerprint exactly matches the store's complete
non-control gene panel. Hover over an ordering button to see the selected source
and coverage. Partially mapped biological genes appear under **Unclassified**,
separately from control/codeword probes.

The catalogue contains portable JSON and CSV references for the P-series
MERSCOPE and Xenium panels, NMV2P35-18, ag7, and the Vizgen 815-gene mouse-brain
panel. Rebuild the portable files with `scripts/build_marker_reference_catalog.py`
and inject one into one or more SpatialData stores with
`scripts/inject_marker_reference.py`. Set
`NAPARI_COMPARE_MARKER_REFERENCE_DIR` to add an external catalogue directory.

### Cell inspector

Hovering over the canvas reports world `x`/`y`, the cell ID under the cursor,
and its broad and fine cell-type annotations in napari's status bar. This
readout remains active whether or not the Cell inspector dock is open. Clicking
a cell adds its exact transcript and image summary to that dock; the broad and
fine annotations appear directly below the Cell ID using the same colours as
the corresponding entries in the **Cell type labels** panel.

```bash
napari-compare-xenium-merscope ... \
  --gene-spot-size 2.0 \            # initial world-micron spot size
  --gene-hide-background \          # start with background spots hidden
  --gene-show-controls \            # start with control/blank probes shown
  --gene-max-render-points 40000000 # subsample cap for very large panels
```

Loading jobs are cancellable and duplicate requests are coalesced. These optional
limits tune reuse and storage concurrency:

```bash
napari-compare-xenium-merscope ... \
  --session-cache-gb 4 \
  --background-io-workers 2
```

See [docs/performance_notes.md](docs/performance_notes.md) for cache details and
the synthetic transcript-store benchmark.

### Zoomed-out appearance (spots and outlines)

Transcript spots have a real micron size and scale with zoom. napari otherwise
clamps every spot to a **2px minimum on screen**, so when zoomed out millions of
spots stay 2px and blanket the view into a solid mass. Two defaults address this:

- `--gene-min-canvas-px` (default `0`) drops that minimum so spots shrink with
  zoom instead of staying a fixed 2px. Raise it (e.g. `1`) if you want spots more
  visible at extreme zoom-out.
- `--gene-antialiasing` (default `1`) lets sub-pixel spots contribute partial
  opacity, so dense regions read darker than sparse ones and tissue structure
  (e.g. cortical layering) shows through when zoomed out. Set `--gene-antialiasing 0`
  for hard-edged spots if antialiasing costs too much on your GPU.

Label outlines use crisp `nearest` interpolation at every zoom by default, with
no filter change during navigation. Their cached coarse levels store fractional
coverage of the full-resolution boundary, so very thin outlines fade
proportionally instead of becoming fully opaque, oversized blocks when napari
switches pyramid levels. A square-root opacity curve keeps partially covered
boundaries readable and reduces the brightness difference between ordinary
lines and intersections. `--label-interpolation linear` remains available as
an explicit compatibility option.

Lazy raster viewport reads run asynchronously by default, so obsolete image and
segmentation requests are cancelled instead of blocking camera input. Use
`--disable-async-slicing` only as a compatibility fallback.

### Scale bar

Once a dataset has loaded, a scale bar appears in the bottom-right corner of the
canvas. It remains hidden on the empty launch screen. Its length is fixed at
about a quarter of the canvas width; as you zoom, the **bar stays the same size**
and only its **label** changes to show how many microns it currently spans (large
bold white text with a black outline).

## Zarr metadata

The **Zarr metadata** button at the top of the right-hand controls opens a
read-only text tree of the active SpatialData object. Expand or collapse branches
to inspect images, labels, points, shapes, tables, and attributes, including table
`obs`, `var`, and nested `uns` entries. Elements omitted from the viewer's loaded
object are browsed directly from the backing Zarr store, with their stored encoding
and attributes visible. Lazy transcript frames show their column types.

Images, matrices, and table columns show shape, data type, and other available
metadata only; the explorer never renders images or reads array values. Long
collections are paged in groups of 100; expand **Show more…** to continue. Reads run in the background as branches open. **Collapse all** folds the
tree, and **Refresh** rebuilds it from the current object and store. In paired
mode, MERSCOPE and XENIUM appear one below the other, with independent expansion
and collapse in either layout.

## UMAP view

The **UMAP** tab opens a floating window with the UMAP embedding MerXen stores
in each clustering table (`obsm['X_umap']` of `table_*_clustering_squidpy`).
Only that array and the cell-id column are read, so opening it takes about a
second even for ~170k cells. The window can be closed and reopened at any time
and stays above the viewer.

- **Cell type source** picks which clustering's UMAP is shown (ProSeg hybrid,
  ProSeg reseg, ProSeg in-mask, Cellpose, or the original segmentation). Only
  sources with a stored UMAP are offered.
- **Colour by** offers stored cell-type levels (including **Broad cell type**,
  **Fine cell type** and the Allen hierarchy described below), or a
  **Per-cell statistic** (Channel / Statistic / Colormap, as on the Per cell
  statistics tab). Cell-type colours are identical to the **Cell type labels**
  masks. Statistics use the same 1st–99th percentile clipping as the spatial
  overlay, so a cell has the same colour in both views. The statistics are
  measured on the Cellpose masks, so choosing one switches the UMAP to the
  Cellpose clustering.
- With **Follow the Cell type labels and Per cell statistics tabs** ticked
  (the default), the UMAP follows those tabs. Clicking a source, **Broad**, or
  **Fine** on Cell type labels, or **Load per-cell statistic overlay** on Per
  cell statistics, recolours the UMAP to match. Cell types unticked on the
  Cell type labels tab are dimmed in the UMAP.
- Selection is linked both ways. Cells clicked in the spatial view are ringed
  in the UMAP in their highlight colour. Clicking a point in the UMAP selects
  that cell in the spatial view: it gets a boundary highlight and a Cell
  inspector panel, and the view pans to it (untick **Centre the spatial view
  on cells picked in the UMAP** to turn off the panning). When the UMAP's source
  uses different masks from the cell inspector (e.g. a Cellpose UMAP while
  clicks pick ProSeg cells), the selection is matched by location: the cell
  whose mask contains the other cell's centre.

Scroll to zoom, drag to pan, and double-click (or **Reset view**) to fit the
embedding. Hovering a point shows its cell ID and its label or value.

### Allen / MAPMYCELLS cell-type hierarchy

MerXen's reference-based annotation and `map_first` clustering outputs are
supported in **Cell type labels** and **UMAP**, including the default
`table_MOSAIK_proseg_hybrid_clustering_squidpy` table joined to
`MOSAIK_proseg_hybrid` masks. Named runs such as
`table_MOSAIK_proseg_hybrid_clustering_squidpy_mapfirst` are also discovered.
When several runs coexist, the viewer prefers an unsuffixed table with Allen
annotation columns, then an annotated `_mapfirst` table, then other annotated
suffixes in alphabetical order (for example `_m8` before `_m8_state`). Legacy
unsuffixed tables remain the fallback. Labels and UMAP always use the same
selected table; its name appears under the UMAP title. Other segmentations can
expose the same levels when their own clustering tables contain them.

Choose **ProSeg hybrid** under **Cell type source**, then a stored **Colour by**
level. **Allen · final assignment** displays `ct_final_name`, the deepest
confident assignment selected by MerXen. Depending on species and output,
additional choices include lineage, broad type, class, neurotransmitter,
supercluster, subclass, SEA-AD subclass, and optional report-only cluster or
supertype. These read their corresponding `ct_<level>_name` columns. Individual
level names are displayed as stored, including candidates that MerXen may mark
as low confidence; use **final assignment** for its resolved calls.

Only levels with at least one non-missing label are offered. Missing cells stay
transparent in the spatial fill and appear as **Unannotated** in UMAP; the
viewer never substitutes a coarser label for a missing Allen level. Older stores
retain their Broad/Fine choices. Stores without these annotations or a UMAP
cannot select the corresponding new options.

The selected level drives the spatial fill, UMAP colours, hover labels, type
filters, and region-of-interest summaries. In paired mode the same type has the
same colour in both platforms and both views. Spatial fills offer levels shared
by both stores; paired UMAPs can show a level available in either store, with
unannotated points in the other. Changing source or dataset removes unavailable
choices and falls back to an available level.

### Regions of interest

Click **Draw region of interest** in the UMAP tab, then hold the mouse button
and drag around cells, either in the spatial view or in the UMAP window.
Releasing the button closes the outline with a straight line from the last
point back to the first. Each press of the button draws one region; the
button then switches off again so dragging pans as usual (click it again
while drawing to cancel).

- **Which cells.** A region drawn in the spatial view takes the cells whose
  mask centre lies inside it. A region drawn in the UMAP takes the points
  inside it. Only cells with a UMAP row from the current cell type source
  count, so both views show the same set. Outlines that cross themselves
  keep both loops.
- **Spatial view.** The region's cells get a thick white outline around
  their masks, drawn as a single raster layer (`Region of interest | cells`)
  however many cells are selected. No Cell inspector panels, transcript
  links or per-cell shapes are added, so selecting tens of thousands of
  cells stays fast. A region drawn in the spatial view also keeps its drawn
  outline (`Region of interest | outline`).
- **UMAP.** Cells outside the region are muted grey and drawn behind it. A
  panel on the right shows a donut chart of the region's make-up by the
  current level (broad or fine cell type) with the total in the middle, and
  a table of counts and percentages per type. When colouring by a per-cell
  statistic, it also compares the region's mean with the whole UMAP.
- **Changing the source.** A spatial region is re-evaluated against the new
  source's cells and masks. A region drawn in the UMAP is cleared, because
  its outline only means something on the embedding it was drawn on.

**Clear region of interest** (in the tab or the UMAP window) removes the
region from both views. Individual cell selections are independent of
regions and keep working as before.

### Paired UMAPs

With a MERSCOPE/Xenium pair open, the UMAP window shows two UMAPs, one above
the other: **MERSCOPE** on top and **Xenium** below. Each one comes from its
own store's clustering table for the chosen cell type source. The two UMAPs are
not harmonised, so each is only ever compared with its own spatial view.

- **Colours.** Both UMAPs use the palette of the paired Cell type labels fill,
  built over both stores' labels, so a cell type has the same colour in both
  UMAPs and both spatial panes. The Cell type labels tab's source, level and
  ticks sync as in standalone mode. Per-cell statistics are unavailable for
  pairs.
- **Selecting cells.** A spatial click already selects the cells of both
  stores at the same aligned point; each UMAP rings its own store's cell.
  Clicking a point in one UMAP selects only that store's cell.
- **Regions drawn in the spatial view.** An outline drawn in either pane (or
  on the stacked canvas) is drawn identically in both panes. The stores share
  the aligned MerXen coordinate system, so the outline covers the same tissue
  in both. Each store's cells inside it are outlined on its own masks,
  emphasised in its own UMAP, and summarised in its own pie chart.
- **Regions drawn in a UMAP.** These apply to that modality only. Circling
  cells in the MERSCOPE UMAP outlines only MERSCOPE cells in the spatial view,
  shows only the MERSCOPE pie, and clears any region shown for Xenium.

The region layers are registered with the paired layout as
`<PLATFORM> | Annotation | Region of interest | …` (outline and cells), so they
follow side-by-side/stacked switching like every other paired layer. Clicking
**Draw region of interest** does the one-off preparation up front (a second or
so on large pairs), so the drawing itself stays smooth.

## Data Assumptions

In standalone mode, MERSCOPE images are placed in microns with the inverse of
the micron-to-mosaic-pixel transform, taken from the first of:

1. `--merscope-transform-path`, when that file exists;
2. `micron_to_mosaic_pixel_transform.csv` inside the zarr directory, then beside
   it (the zarr's parent folder);
3. the `global` transformation of `shapes/merscope_cell_boundaries` in the zarr
   (read from metadata only). The SpatialData MERSCOPE reader stores the same
   matrix there, so MerXen's published stores need no CSV.

The log names the source used. Only when none is usable does the viewer fall
back to `0.108 um/px` with no translation, with a warning: images are then
offset from masks and transcripts unless the mosaic origin is (0, 0) um.
Viewer-built label caches rasterized under a transform that places them more
than 1 um from the current one are rebuilt.

In standalone mode, Xenium inputs may include `experiment.xenium` either inside
the zarr directory or beside it. If missing, the viewer falls back to `0.2125
um/px`.

Standalone SpatialData stores should contain points for transcripts, images for
channel display, and either vector `shapes` or raster `labels` for
segmentations. Paired mode does not use the standalone transform fallbacks or
infer equivalent elements: it requires the exact common coordinate system,
fixed grid, and materialized artifact mappings declared by a valid MerXen v2
pair.

## Cortical-depth annotations for MerXen

For a complete step-by-step workflow, see
[docs/cortical_depth_annotation_guide.md](docs/cortical_depth_annotation_guide.md).

The same tab also supports named polygon objects for MerXen's
`distance_from_object` stage. Enter an object name (for example, `Amyloid
plaques`), create its polygon layer, draw one polygon per object, then validate
and export a combined object GeoJSON. Existing object GeoJSON can be loaded to
continue editing while preserving IDs. See
[docs/distance_object_annotation_guide.md](docs/distance_object_annotation_guide.md).

The **Draw tissue annotations** tab holds the cortical-depth tools. Click
**Create Drawing Layers** to add one editable napari Shapes layer for each
MerXen cortical-depth input role. Draw in the same visible coordinate frame as
the SpatialData cells/transcripts, then click **Validate Annotations** or
**Export Combined GeoJSON**.

The exporter writes a single GeoJSON `FeatureCollection` with MerXen role
labels. Napari stores shape vertices as `(y, x)`; export converts them back to
GeoJSON `[x, y]` and does not otherwise transform, scale, flip, normalize, or
rotate coordinates.

Use **Current Tissue Piece** to choose the `tissue_piece_id` for each cortical
piece. New pia, WM, exclusion, and ribbon shapes inherit the selected piece ID.
Use **New Piece** for another independent tissue piece and **Apply Piece To
Selection** to relabel selected existing shapes.

Draw the required inputs first:

1. **Tissue edge**: on `side`, draw exactly one tissue-edge path. It may be an
   open U-shaped path with sharp corners or a closed box-like path around the
   tissue. It is global, not piece-specific.
2. **Pial boundary**: on `pia`, draw one open path per tissue piece along the
   pial surface. This is cortical depth 0 for depth pieces. A pial-only piece is
   allowed and is exported as `piece_mode: mask_qc_only`.
3. **Gray/white boundary**: on `wm`, draw one open path for each tissue piece
   that has gray/white matter. This is cortical depth 1. A WM line without a pia
   line for the same `tissue_piece_id` is invalid.

Optional inputs improve QC and geometry handling:

4. **Snap to edge**: use **Snap Boundaries To Edge** after drawing the edge,
   pia, and optional WM paths. This moves pia/WM endpoints to their nearest
   position on the single tissue edge.
5. **Exclusion polygons**: on `exclusion`, draw simple polygons around tears,
   folds, vessels, debris, or artifacts to subtract from the cortical ribbon.
   Keep exclusions inside the ribbon when possible and avoid touching the pia or
   WM boundaries unless the artifact really opens to the tissue edge.
6. **Full cortical ribbon**: on `ribbon`, draw a complete simple polygon or
   polygons for a piece when edge + pia + optional WM does not unambiguously
   define the usable cortical ribbon. Pial-only pieces with ambiguous geometry
   require an explicit ribbon polygon.

Validation blocks export when required lines are missing, lines are closed or
empty, polygon coordinates are invalid, or coordinates contain non-finite
values. It also blocks multiple edge lines and WM-only pieces, and warns about
pial/WM crossings, endpoints far from the tissue edge, overlapping piece
polygons, and exclusions touching pia/WM.

Use the exported combined GeoJSON in MerXen samplesheets with:

```csv
xenium_cortical_depth_annotation_geojson
merscope_cortical_depth_annotation_geojson
```

Generic single-platform rows may also use:

```csv
cortical_depth_annotation_geojson
```

For the separate-file export, point MerXen at the generated files with
platform-prefixed columns such as `xenium_pial_boundary_geojson`,
`xenium_wm_boundary_geojson`, `xenium_side_boundaries_geojson`,
`xenium_exclusion_masks_geojson`, and `xenium_cortical_ribbon_geojson`; use
`merscope_` for MERSCOPE rows.

## Development

```bash
pytest
```

### Saved dataset rotation

Use the Rotation slider or angle field below the layer list to rotate the display.
Click **Save** beside the angle to restore it the next time you open that dataset.
Angles are stored in local viewer settings using the full dataset path; an aligned
MERSCOPE/Xenium pair has its own shared saved angle. Moving a store to a new path
starts with zero rotation. To clear a saved rotation, set the angle to 0° and save.
Source images, coordinates, and Zarr stores are never modified. Region drawing
follows the rotated display while selecting cells in their original coordinate frame.
