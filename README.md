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

## Data Assumptions

MERSCOPE inputs may include `micron_to_mosaic_pixel_transform.csv` inside the zarr directory. If missing, the viewer falls back to `0.108 um/px`.

Xenium inputs may include `experiment.xenium` either inside the zarr directory or beside it. If missing, the viewer falls back to `0.2125 um/px`.

The SpatialData stores should contain points for transcripts, images for channel display, and either vector `shapes` or raster `labels` for segmentations.

## Development

```bash
pytest
```
