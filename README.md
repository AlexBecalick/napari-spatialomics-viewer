# napari-compare-xenium-merscope

Standalone Napari viewer for comparing MERSCOPE and Xenium SpatialData `.zarr` outputs.

The viewer opens a single Napari window with a dataset switcher. It can show image channels, segmentation layers from `shapes` or `labels`, and assigned/unassigned transcript points.

## Install

The most reproducible setup mirrors the environment used by the original MOSAIK viewer:

```bash
conda env create -f environment.yml
conda activate napari-compare-xenium-merscope
```

If you already created the environment before `spatialdata==0.7.3a1` was pinned, update it in place:

```bash
conda activate napari-compare-xenium-merscope
pip install --upgrade --pre "spatialdata==0.7.3a1"
```

For an existing Python environment:

```bash
pip install -e .
```

## Run

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

Useful options:

```bash
napari-compare-xenium-merscope \
  --merscope-zarr /path/to/merscope.zarr \
  --xenium-zarr /path/to/xenium.zarr \
  --startup-mode fast \
  --max-transcripts 200000 \
  --max-shapes-per-layer 20000
```

Fast startup loads transcripts and lists available segmentations, but does not
add polygon layers by default. Use **Load Selected Segmentations (Capped)** for
a capped preview, **Load Selected Segmentations (All: bbox)** for selected full
segmentation layers in a lightweight bounding-box representation, or
**Load All Segmentations (Capped)** when you want a bounded
overview of every segmentation key.

For exact polygon boundaries without loading a whole Cellpose layer into RAM,
select one or more segmentation keys and use **Stream Selected Polygons**. This
adds a bounding-box overview and refreshes an exact polygon detail layer when
the current view is small enough. Tune this with:

```bash
napari-compare-xenium-merscope ... \
  --stream-shapes-max-view-size 2500 \
  --stream-shapes-max-polygons 2500
```

For the most memory-efficient full segmentation display, select one or more
shape keys and use **Load Selected Labels (Outlines)**. If a matching label
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

Very large Cellpose layers can contain tens of millions of boundary vertices.
The uncapped selected-load button uses `--full-shape-render-mode bbox` by
default to avoid materializing every vertex in a Napari `Shapes` layer. Other
options are:

```bash
# Lightest full-layer view.
napari-compare-xenium-merscope ... --full-shape-render-mode centroid

# Approximate full outlines with at most 64 vertices per polygon.
napari-compare-xenium-merscope ... \
  --full-shape-render-mode path \
  --shape-max-vertices-per-polygon 64

# Exact full outlines; highest memory use.
napari-compare-xenium-merscope ... --full-shape-render-mode path
```

## Inspect genes

Click **Inspect Genes** in the Transcripts section of the right dock to open the
**Gene Inspector** panel. It loads the full gene panel for the current dataset,
alphabetically, and gives every gene a unique **colour + marker shape** shown as a
large icon beside its name. Unlike the default assigned/unassigned view, this
renders *all* transcripts as real points coloured by gene (no density raster or
viewport sampling) — feasible because a typical section is ~10-20M transcripts.
Opening it takes over the transcript display; **Close Gene Inspector** restores the
default density/assigned-unassigned view.

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

The right-side dock has a **Cortical Depth Annotations** section. Click
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
