# napari-compare-xenium-merscope

Napari-based viewer for visualising MERSCOPE and Xenium SpatialData `.zarr` outputs.

The viewer opens a single Napari window with a tabbed control panel (tabs across
the top of the right-hand dock): **Gene inspector**, **Cell segmentation**, **Per
cell statistics**, **Draw tissue annotations**, **Images**, and **Dataset**. On
startup with a supplied dataset it automatically loads all image channels, the
Cellpose and ProSeg cell segmentations, and all transcripts (rendered as per-gene coloured points). A busy
progress bar with a stage label below the tabs shows what is loading (image
pyramids, cell masks, transcripts, …). Image channels appear first as lazy
previews while optimized pyramids build in the background; switching datasets
keeps recent prepared sessions in an LRU cache for fast return visits.

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
loaded and selects the **Dataset loader** tab. Choose a paired MERSCOPE/Xenium
dataset or a standalone store there by selecting the dataset's
`spatialdata.zarr` folder. The empty canvas points toward these controls and
keeps napari's rotating **Did you know?** tip. The loader also retains the ten
most recently opened dataset folders; select an entry and click **Open selected
recent dataset** (or double-click it) to reopen it. Dataset paths passed
explicitly on the command line still load immediately.

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

Or supply dataset paths to load them immediately:

```bash
napari-compare-xenium-merscope \
  --merscope-zarr /path/to/merscope.zarr \
  --xenium-zarr /path/to/xenium.zarr
```

You can also open just one dataset:

```bash
napari-compare-xenium-merscope --merscope-zarr /path/to/merscope.zarr
napari-compare-xenium-merscope --xenium-zarr /path/to/xenium.zarr
```

### Default startup and low-RAM flags

By default the viewer eagerly loads everything: all image channels, the Cellpose
and ProSeg segmentations (shown as lazy label outlines), and all transcripts as
per-gene points. Per-cell statistic overlays are **not** loaded automatically.

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
  --skip-transcripts    # don't auto-load transcripts (load from the Gene inspector tab)
```

### Tabs

The tab buttons across the top have a white background and bold black text so
they stand out as the main controls, and they wrap onto extra rows on a narrow
dock so every full name stays readable.

- **Gene inspector** — the transcript view (see below). **Load / reload
  transcripts** rebuilds the per-gene points; **Unload transcripts** frees them.
- **Cell segmentation** — a list of every segmentation key; currently-loaded
  segmentations are shown in **green**. Select one or more and use **Load
  selected cell segmentation** / **Unload selected cell segmentation**. Cellpose
  and ProSeg load automatically at startup.
- **Per cell statistics** — Channel / Statistic / Colormap dropdowns plus
  **Load / Unload per-cell statistic overlay** (MERSCOPE Cellpose quantification).
- **Draw tissue annotations** — the cortical-depth drawing tools (see below).
  Selecting this tab automatically expands napari's layer controls so the active
  drawing layer can be edited.
- **Images** — a list of every individual image **channel** (across all image
  elements, for both MERSCOPE and Xenium); currently-loaded channels are shown in
  **green**. Use **Load selected image(s)**, **Load all images**, and **Unload
  selected image(s)**.
- **Dataset loader** — the MERSCOPE/XENIUM switcher for the currently open
  stores, **Reload Dataset**, and buttons to open a different dataset: **Load new
  paired dataset** (browse for a MERSCOPE store then a Xenium store) or **Load new
  standalone MERSCOPE / Xenium dataset** (browse for a single store). Each
  browser expects the store's `spatialdata.zarr` folder. **Recently viewed**
  lists up to ten stores and persists between launches.

The left dock is simplified for the spatial-transcriptomics workflow. **Layer
controls** starts collapsed and can be expanded or collapsed with the arrow in
its title. The create-points, create-shapes, create-labels, console, 2D/3D,
axis, and grid buttons are hidden; delete and **Reset view to original state**
remain available.

Segmentations display as memory-efficient label outlines. If a matching label
element is already present in the SpatialData store, it is loaded lazily. If it
is missing, the viewer rasterizes the selected polygon layer chunk by chunk,
saves it back to `labels/<shape-key>_labels` in the zarr, and then displays the
label outlines as a lazy multiscale image. Future runs reuse the saved label
layer and only rebuild the lightweight outline view.

Clicking inside a cell mask highlights that cell (repeated clicks accumulate
highlights). This is gated on the **ProSeg** (cell-inspector) layer's visibility:
hiding that layer in the napari layer list disables click-to-highlight and clears
any current highlights, and showing it again re-enables clicking.

The transcript renderer still uses one real napari Points layer for each marker
symbol behind the scenes, but the layer list presents them as one **Genes** row.
The row sits immediately beneath the final visible native layer, and its
visibility control shows or hides every gene layer together. The real gene layer
block remains pinned to the bottom of napari's model, so transcript points sit
*below* image and mask layers; toggle image visibility if you need the spots in
front.

```bash
# Rebuild cached labels instead of reusing existing labels.
napari-compare-xenium-merscope ... --overwrite-labels

# Use smaller chunks while generating labels.
napari-compare-xenium-merscope ... --label-chunk-size 1024

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
transcripts** to rebuild the panel and **Unload transcripts** to free the points.

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

The reference is a standalone data artefact produced outside this package (the
viewer only *reads* it). For the P7513 panel it was written both as
`cell_type_marker_reference.{csv,json}` beside the `.zarr` and as the
`cell_type_marker_reference` table element inside the store.

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

Label outlines are rasterised, so a downsampled multiscale level can make thin
outlines look thick and blocky (merging into blocks of colour) when zoomed out.
`--label-interpolation` (default `linear`) anti-aliases the outlines on screen so
they stay thin and fade rather than blocking up; use `nearest` for crisper — but
blockier when zoomed out — outlines. In the default `linear` mode the outlines
automatically switch to crisp `nearest` interpolation once you zoom in far enough
that the field of view is ≤ ~150 µm (so single-cell close-ups stay sharp), and
back to `linear` when you zoom out. All of these are display-only settings with no
material performance cost (antialiasing aside).

### Scale bar

Once a dataset has loaded, a scale bar appears in the bottom-right corner of the
canvas. It remains hidden on the empty launch screen. Its length is fixed at
about a quarter of the canvas width; as you zoom, the **bar stays the same size**
and only its **label** changes to show how many microns it currently spans (large
bold white text with a black outline).

## Data Assumptions

MERSCOPE inputs may include `micron_to_mosaic_pixel_transform.csv` inside the zarr directory. If missing, the viewer falls back to `0.108 um/px`.

Xenium inputs may include `experiment.xenium` either inside the zarr directory or beside it. If missing, the viewer falls back to `0.2125 um/px`.

The SpatialData stores should contain points for transcripts, images for channel display, and either vector `shapes` or raster `labels` for segmentations.

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
