# Object annotations for MerXen

The **Draw tissue annotations** tab can create registered polygon inputs for
MerXen's `distance_from_object` stage. Use these tools for discrete objects such
as amyloid plaques or tau tangles after the image/cell registration has already
been applied.

## Draw and export

1. Activate the MERSCOPE or Xenium dataset whose coordinate frame you want to
   annotate.
2. Load the registered image layer that makes the objects visible.
3. Open **Draw tissue annotations**.
4. Enter a descriptive object name, for example `Amyloid plaques`, then click
   **Create Named Object Layer**.
5. Draw one closed polygon around each object. Create another named layer for a
   different object type, such as `Tau tangles`.
6. Click **Validate Object Annotations**, correct any reported invalid or
   self-intersecting polygons, then click **Export Object GeoJSON**.

Each polygon becomes one GeoJSON feature with the user-provided `object_type`
and a stable `object_id`. Napari coordinates are converted from `(y, x)` to
GeoJSON `(x, y)` during export.

## Reload and continue editing

Click **Load Object GeoJSON** and select a previous export. The viewer recreates
one polygon layer per object type and preserves existing object IDs. Drawing
more polygons on those layers assigns non-conflicting IDs at the next export.

Loading replaces the shapes for object types present in the file. Other named
object layers already open for the active dataset are left unchanged, so use
**Validate Object Annotations** before exporting the combined set.

## MerXen samplesheet

Use platform-specific columns when paired sections have different annotations:

```text
merscope_distance_object_annotation_geojson
xenium_distance_object_annotation_geojson
```

For a single-platform row, `distance_object_annotation_geojson` is also
accepted. The annotation coordinates must already match the selected cell
tables; MerXen does not register them in the distance stage.

The cortical-depth drawing tools remain separate. MerXen uses their resulting
`cortical_depth_annotation` table column to restrict near/far pseudobulk
analysis to `grey_matter` cells.
