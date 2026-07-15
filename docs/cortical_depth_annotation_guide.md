# Cortical Depth Annotation Guide

This guide explains how to draw and export cortical-depth annotations for the
MerXen `compute_cortical_depth` stage from the napari comparison viewer.

The exported GeoJSON uses the same 2D coordinate system as the SpatialData cell
centroids and shapes. The viewer converts napari `(y, x)` shape vertices to
GeoJSON `[x, y]` coordinates, but it does not scale, rotate, flip, normalize, or
otherwise transform coordinates.

## Core Concepts

MerXen uses these annotation roles:

| Layer | Required | Geometry | Meaning |
|-------|----------|----------|---------|
| `side` | yes | one LineString | The global tissue edge. Draw exactly one edge line for the whole section. |
| `pia` | yes, per tissue piece | open LineString | Pial/surface boundary. This is cortical depth 0 for depth pieces. |
| `wm` | optional, per tissue piece | open LineString | Gray/white matter boundary. This is cortical depth 1 for depth pieces. |
| `exclusion` | optional, per tissue piece | Polygon | Tears, folds, vessels, debris, or artifacts to subtract from the ribbon. |
| `ribbon` | optional, per tissue piece | Polygon | Explicit usable cortical ribbon mask. Use when line geometry is ambiguous. |

Each independent cortical/tissue region should have a `tissue_piece_id`.
The viewer stores this ID on pia, WM, exclusion, and ribbon shapes. The tissue
edge is global and does not need a piece ID.

Pieces with both pia and WM are exported with `piece_mode: depth` and MerXen
computes Laplace and equivolumetric depth. Pieces with pia but no WM are
exported with `piece_mode: mask_qc_only`; MerXen keeps them as mask/QC regions
but does not compute depth values for cells inside them.

## Step 1: Open The Dataset

1. Start the napari comparison viewer with the relevant Xenium or MERSCOPE
   SpatialData zarr.
2. Load the image, segmentation outlines, or transcript/cell layers that make
   the cortical tissue boundaries easiest to see.
3. Confirm you are annotating in the coordinate frame used by the SpatialData
   cell centroids/shapes. Do not manually rescale or rotate annotation layers.

## Step 2: Create Drawing Layers

1. In the right-side dock, open the **Draw tissue annotations** tab.
2. Click **Create Drawing Layers**.
3. The viewer creates editable layers for `pia`, `wm`, `side`, `exclusion`, and
   `ribbon`.
4. The `pia` layer becomes active first. You can switch layers in napari as
   needed.

## Step 3: Draw The Tissue Edge

1. Select the `side` layer.
2. Draw exactly one path representing the tissue edge or artificial analysis
   boundary for the whole annotated section.
3. The edge may be:
   - an open U-shaped line,
   - an open line with sharp corners,
   - a closed box-like path around the tissue,
   - a curved or irregular line that follows a cut edge.
4. Do not draw multiple separate edge lines. Validation blocks export if more
   than one edge line exists.

The edge line is used to construct piece polygons when no explicit `ribbon`
polygon is supplied. It is also used for side-boundary QC and near-edge flags.
It is okay for the edge line to extend beyond the points where pia or WM touch
it. Those overhanging edge segments are expected when drawing by hand.

## Step 4: Select Or Create A Tissue Piece

1. In **Current Tissue Piece**, keep `piece_1` for the first tissue piece.
2. Before drawing another independent piece, click **New Piece**. The viewer
   creates the next ID, such as `piece_2`.
3. Draw all pia, WM, exclusion, and ribbon shapes for that piece while its piece
   ID is selected.
4. If you drew a shape with the wrong piece ID:
   - select that shape on its napari layer,
   - choose the correct **Current Tissue Piece**,
   - click **Apply Piece To Selection**.

Use one `tissue_piece_id` for each independent cortical region that should be
processed separately by MerXen.

## Step 5: Draw The Pial Boundary

1. Select the `pia` layer.
2. Confirm **Current Tissue Piece** is set to the piece you are drawing.
3. Draw one open path along the pial/surface boundary for that piece.
4. The pia endpoints should touch, or be very close to, the single tissue edge.
5. Avoid closed loops, branches, duplicate loops, and self-crossings.

For a depth piece, the pia is the depth 0 boundary. For a pial-only piece, the
pia marks the surface boundary of a mask/QC-only region.

## Step 6: Draw The White Matter Boundary When Present

1. If the tissue piece has a visible gray/white matter boundary, select the
   `wm` layer.
2. Confirm **Current Tissue Piece** is the same piece ID used for that piece’s
   pia line.
3. Draw one open path along the gray/white boundary.
4. The WM endpoints should touch, or be very close to, the same global tissue
   edge.
5. The WM line should belong to the same cortical region as the pia line.
6. Avoid WM lines that cross the pia line.

Do not draw a WM line without an associated pia line for the same
`tissue_piece_id`. Validation blocks WM-only pieces.

## Step 7: Snap Boundary Endpoints To The Edge

1. After drawing the tissue edge, pia lines, and any WM lines, click
   **Snap Boundaries To Edge**.
2. The viewer moves each pia/WM endpoint to the nearest point on the global
   tissue-edge line.
3. Inspect the results and undo/redraw if the nearest edge point was not the
   intended connection point.

Snapping is recommended because MerXen constructs piece masks from line
topology. Clean endpoint contact makes polygon construction and QC more robust.

## Step 8: Draw Exclusion Polygons

1. Select the `exclusion` layer.
2. Confirm **Current Tissue Piece** matches the piece containing the artifact.
3. Draw simple polygons around tears, folds, vessels, debris, or obvious
   artifacts.
4. Keep exclusions fully inside the intended cortical ribbon when possible.
5. Avoid touching pia or WM boundaries unless the artifact truly opens to the
   tissue edge.
6. Avoid exclusions that sever a depth piece into disconnected parts unless this
   is intentional and expected during QC.

Invalid or self-intersecting exclusion polygons block export.

## Step 9: Draw An Explicit Ribbon Polygon When Needed

Use the `ribbon` layer when the tissue-edge plus pia plus optional WM lines do
not unambiguously define the usable cortical region.

Draw a ribbon polygon when:

1. the true tissue cut edge is curved or irregular and line polygonization might
   choose the wrong region,
2. the piece is pial-only and edge + pia could form more than one possible
   polygon,
3. multiple pieces sit close together and their inferred polygons may overlap,
4. straight or automatic closures would include non-tissue,
5. validation reports ambiguous pial-only geometry.

For depth pieces, draw the pia and WM on or just inside the ribbon polygon
boundary so MerXen can rasterize the Dirichlet boundaries inside the mask.

## Single-Piece Workflow

For one cortical region with pia and WM:

1. Click **Create Drawing Layers**.
2. Draw one `side` tissue-edge line.
3. Keep **Current Tissue Piece** as `piece_1`.
4. Draw one `pia` line for `piece_1`.
5. Draw one `wm` line for `piece_1`.
6. Click **Snap Boundaries To Edge**.
7. Add optional `exclusion` polygons for `piece_1`.
8. Add an optional `ribbon` polygon for `piece_1` if the inferred polygon is
   not reliable.
9. Validate and export.

## Multiple-Piece Workflow

For multiple independent tissue pieces:

1. Draw one global `side` tissue-edge line for the whole section.
2. Set **Current Tissue Piece** to `piece_1`.
3. Draw `pia`, optional `wm`, optional `exclusion`, and optional `ribbon`
   shapes for `piece_1`.
4. Click **New Piece** to create `piece_2`.
5. Draw all shapes for `piece_2`.
6. Repeat for each independent tissue piece.
7. Click **Snap Boundaries To Edge**.
8. Validate and inspect warnings about overlapping piece polygons or endpoints
   far from the edge.
9. Export the combined GeoJSON.

Important rules for multiple pieces:

1. Use one piece ID per independent region.
2. A WM line must share the same `tissue_piece_id` as its associated pia line.
3. Pial-only pieces are allowed and become `mask_qc_only`.
4. Do not use separate edge lines for each piece. There must be exactly one
   global edge line.
5. Use explicit `ribbon` polygons for pieces whose inferred mask is ambiguous.

## Validation

Click **Validate Annotations** before exporting.

Validation blocks export when:

1. no tissue-edge line is present,
2. more than one tissue-edge line is present,
3. no pial boundary is present,
4. a piece has WM but no pia,
5. multiple pia or WM segments for one piece cannot merge into one continuous
   line,
6. pia or WM paths are closed,
7. polygons are invalid, self-intersecting, empty, or contain non-finite
   coordinates,
8. pial-only geometry cannot be turned into exactly one polygon and no explicit
   ribbon polygon was supplied.

Validation warns when:

1. pia and WM cross or touch within a depth piece,
2. pia/WM endpoints are far from the tissue edge,
3. exclusion polygons touch or overlap pia/WM,
4. candidate piece polygons overlap,
5. multiple possible depth-piece polygons are found.

Warnings do not always block export, but they should be inspected. If a warning
reflects incorrect geometry, fix the drawing before saving.

If validation fails during **Export Combined GeoJSON**, the viewer offers to
save the current annotations anyway for debugging. This forced export is not
guaranteed to run in MerXen, but it preserves the drawn geometry and includes a
top-level `napari_compare_validation` report with the errors and warnings.

## Export

Export the combined MerXen annotation file:

1. Click **Export Combined GeoJSON**.
2. Choose a filename such as `xenium_cortical_depth_annotations.geojson` or
   `merscope_cortical_depth_annotations.geojson`.
3. Add the exported path to the MerXen samplesheet column:
   - `xenium_cortical_depth_annotation_geojson`,
   - `merscope_cortical_depth_annotation_geojson`,
   - or generic `cortical_depth_annotation_geojson` for single-platform rows.

## Troubleshooting

If export says the edge is missing:

1. Select the `side` layer.
2. Draw exactly one tissue-edge path.
3. Validate again.

If export says there are multiple edge lines:

1. Select the `side` layer.
2. Delete extra edge shapes.
3. Keep one global line that covers the relevant tissue boundaries.

If a WM-only piece is reported:

1. Select the relevant WM shape.
2. Check its `tissue_piece_id`.
3. Draw a pia line with the same piece ID, or relabel the WM shape to the
   correct piece.

If endpoints are far from the tissue edge:

1. Click **Snap Boundaries To Edge**.
2. Inspect the snapped endpoints.
3. Redraw the tissue edge or boundary line if snapping picked the wrong point.

If pial-only geometry is ambiguous:

1. Select the `ribbon` layer.
2. Set **Current Tissue Piece** to the pial-only piece.
3. Draw a complete ribbon polygon for that piece.
4. Validate again.

If piece polygons overlap:

1. Confirm each pia, WM, exclusion, and ribbon shape has the intended
   `tissue_piece_id`.
2. Use **Apply Piece To Selection** to fix mislabeled shapes.
3. Draw explicit ribbon polygons for close or complex pieces if needed.
