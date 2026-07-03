# napari-compare-xenium-merscope

Napari-based viewer for visualising MERSCOPE and Xenium SpatialData `.zarr` outputs.

The viewer opens a single Napari window with a tabbed control panel (tabs across
the top of the right-hand dock): **Gene inspector**, **Cell segmentation**, **Per
cell statistics**, **Draw tissue annotations**, **Images**, and **Dataset**. On
startup it automatically loads all image channels, the Cellpose and ProSeg cell
segmentations, and all transcripts (rendered as per-gene coloured points). A busy
progress bar with a stage label below the tabs shows what is loading (image
pyramids, cell masks, transcripts, …).

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

## Then Run

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
- **Images** — a list of every individual image **channel** (across all image
  elements, for both MERSCOPE and Xenium); currently-loaded channels are shown in
  **green**. Use **Load selected image(s)**, **Load all images**, and **Unload
  selected image(s)**.
- **Dataset loader** — the MERSCOPE/XENIUM switcher for the currently open
  stores, **Reload Dataset**, and buttons to open a different dataset: **Load new
  paired dataset** (browse for a MERSCOPE store then a Xenium store) or **Load new
  standalone MERSCOPE / Xenium dataset** (browse for a single store).

Segmentations display as memory-efficient label outlines. If a matching label
element is already present in the SpatialData store, it is loaded lazily. If it
is missing, the viewer rasterizes the selected polygon layer chunk by chunk,
saves it back to `labels/<shape-key>_labels` in the zarr, and then displays the
label outlines as a lazy multiscale image. Future runs reuse the saved label
layer and only rebuild the lightweight outline view.

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
toggles stay fast even with hundreds of genes. The panel offers:

- a per-gene checkbox (with the colour/shape marker) and a **Show all genes** toggle,
- a **Spot size** slider (world microns; spots keep a real size and scale with zoom),
- **Hide background (unassigned) spots** to drop transcripts not assigned to a cell,
- **Show control / blank probes** to include negative-control codewords (hidden by
  default), and a filter box to search the list.

```bash
napari-compare-xenium-merscope ... \
  --gene-spot-size 2.0 \            # initial world-micron spot size
  --gene-hide-background \          # start with background spots hidden
  --gene-show-controls \            # start with control/blank probes shown
  --gene-max-render-points 40000000 # subsample cap for very large panels
```

## Data Assumptions

MERSCOPE inputs may include `micron_to_mosaic_pixel_transform.csv` inside the zarr directory. If missing, the viewer falls back to `0.108 um/px`.

Xenium inputs may include `experiment.xenium` either inside the zarr directory or beside it. If missing, the viewer falls back to `0.2125 um/px`.

The SpatialData stores should contain points for transcripts, images for channel display, and either vector `shapes` or raster `labels` for segmentations.

## Cortical-depth annotations for MerXen

For a complete step-by-step workflow, see
[docs/cortical_depth_annotation_guide.md](docs/cortical_depth_annotation_guide.md).

The **Draw tissue annotations** tab holds the cortical-depth tools. Click
**Create Drawing Layers** to add one editable napari Shapes layer for each
MerXen cortical-depth input role. Draw in the same visible coordinate frame as
the SpatialData cells/transcripts, then click **Validate Annotations** or
**Export Combined GeoJSON**. **Export Separate GeoJSONs** writes the same
validated annotations as one file per role.

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
