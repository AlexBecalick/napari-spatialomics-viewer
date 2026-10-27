# Manual registration guide

This guide covers the **Manual registration** tab: placing landmark pairs between two images of
the same or adjacent tissue sections, and inspecting a registration result.

## 1. What this is for

You have two images of the same tissue section, or of adjacent (serial) sections. The
*reference* image (for example the Xenium section) stays fixed. The *moving* image (for example
a scan of a laser-capture-microdissection section) is aligned onto it.

MerXen does the heavy work in two workflows, `manual_registration_prepare.nf` and
`manual_registration_register.nf` (see MerXen `docs/stages/manual-registration.md`). The viewer is
where you place landmarks and check results:

1. MerXen **prepare** writes a *prepared store*, `<id>_prepared.zarr`, holding both images.
2. You open it in the viewer, place landmark pairs and save them into the store.
3. MerXen **register** reads your landmarks, fits the transform and writes a *result store*,
   `<run_id>_registered.zarr`, which you open in the viewer to check the registration.

## 2. Opening a store

- **Dataset loader ▸ Manual Registration ▸ Load image data**, then browse to the store folder.
- **Recently viewed**: entries named "MANUAL REGISTRATION — ..." reopen a store you used before.
- From the command line: `napari-compare-xenium-merscope --registration-store <path>`.
- If you pick a registration store with one of the standalone loaders, the viewer notices it and
  opens it in the Manual registration tab instead.

A **prepared** store is editable: you can crop, align and place landmarks. A **result** store is
read-only. Opening a store replaces whatever is currently displayed. If the current registration
session has unsaved changes, the viewer asks whether to save a draft, discard (a local draft is kept
and offered the next time) or cancel.

While a registration session is open the dataset switcher, Reload, segmentation, transcript,
per-cell statistic and annotation tools are disabled, display rotation is set to 0 degrees and
disabled, and the layer list cannot delete or reorder layers. Channels of the two images are in the
layer list; show or hide them there.

## 3. The Manual registration tab

| Section | What it does |
|---|---|
| **Display** | **Reference**, **Moving** or **Both** chooses which image is shown. **Flick (F)** switches between the two images. **Auto-switch after each click** moves the view to the other image after you click a landmark. |
| **Crop** | Draws a rectangle on each image (**Draw**, then drag) or lets you edit its edges as numbers; **Clear** removes it. Use crops to exclude tile-seam borders and coverslip artefacts. |
| **Coarse alignment** | Rotation (a field plus −90, −10, −1, −0.1, +0.1, +1, +10, +90 buttons and a **180°** button), the offset of the moving centre from the reference centre (fields plus arrow buttons with a step size), **Flip H**, **Flip V** and **Reset**. **Use fit as coarse alignment** copies the current landmark fit into these controls (needs at least 3 enabled pairs). |
| **Landmarks** | **Landmark mode** turns clicks on the image into landmark points. The table lists the pairs with their residuals; each pair can be enabled or disabled. **Undo** and **Delete selected** edit the list. The **Outlier floor** sets the distance above which a pair is flagged as an outlier. |
| **Preview fit** | Shows the moving image under the fitted transform (needs at least 3 enabled pairs). Switch it off to return to the coarse alignment. |
| **Save** | **Save draft** stores the landmarks any time. **Finish labelling** marks the labelling as complete; it needs at least 8 enabled pairs and no half-placed pair. |

## 4. Recommended workflow

1. **Crops (optional).** Exclude borders that contain duplicated or missing tiles and coverslip
   artefacts. Crops are rectangles in each image's own pixel frame and can be edited later. MerXen
   snaps them inward to multiples of 64 px.
2. **Coarse alignment.** Use **180°** for upside-down sections, **Flip H** / **Flip V** for mirrored
   ones, then rotate (±90, ±10, ±1, ±0.1) and shift with the arrow buttons until both images roughly
   overlap in **Both** mode.
3. **Landmarks.** Switch **Landmark mode** on. Click a point on the reference image; the view flips to
   the moving image (auto-switch); click the matching point. Press **F** to flick between the images
   to check a position, or hold **Shift+F** to peek at the other image.
4. **Watch the residuals.** The fit and leave-one-out (LOO) residuals update after every pair, with
   warnings for pairs that disagree. Disable (do not delete) a few good pairs to keep them as
   held-out checks.
5. **Save.** **Save draft** any time. **Finish labelling** needs at least 8 enabled pairs and no
   half-placed pair.

## 5. Serial sections

Nuclei do not correspond between serial 5 µm sections, so do not match individual cells. Choose
tissue-scale landmarks: vessel branch points, layer boundaries, white-matter edges and distinctive
tissue-edge features. Spread them over the whole section and aim for 16 to 24 pairs.

Press **serial (30 µm)** next to the outlier floor. **Preview fit** then flags the same outliers as
the `serial_section` run does. The floor is saved with the landmarks and the register workflow
reports it. `merxen manual-registration-fit --preset serial_section` is the authoritative check.

In MerXen run the `serial_section` preset, and also `same_section` for comparison; the two runs are
kept in separate folders.

## 6. Viewing a result

Open `<run_id>_registered.zarr`. The tab shows the reference image and the registered moving image
as two image layers, plus three landmark overlays:

- `Reference landmarks`,
- `Registered moving landmarks` (after the affine fit and the optic-flow refinement),
- `Landmark-fit positions` (hidden by default).

The residual table lists, per landmark, the fit, affine and final residual in µm. A banner above it
names the preset, the refinement and flow status, and the tissue correlation per stage.

## 7. Keyboard and mouse

| Input | Action |
|---|---|
| **F** | Flick between the reference and the moving image |
| **Shift+F** (hold) | Peek at the other image while the keys are held |
| **Ctrl+Z** | Undo (cancels a half-placed pair first) |
| **Escape** | Cancel a half-placed pair or an armed crop drawing; otherwise it leaves Landmark mode |
| Left click | Place a landmark point (in Landmark mode) |
| Drag | Pan |

Ctrl+Alt+T (transform mode) and the layer-deletion shortcuts are disabled in a session.

## 8. Saving, conflicts and recovery

- **Another session changed the store.** If the landmarks on disk changed since you loaded them,
  the viewer offers to reload (discarding your changes here) or to cancel and keep editing.
- **The store is busy.** Another program is writing to it; try again in a moment.
- **Local drafts.** While you edit, an unsaved-work draft is kept in your user's application-data
  folder. If a newer draft exists when you reopen a store, the viewer offers to restore it.
- **History.** Every previous saved payload is kept in `<store>.landmark-history/`.
- **Landmarks placed on different images.** If the stored landmarks do not match the images in the
  store, the viewer says so and offers to start fresh.
- **Written by a newer version.** If the store or the stored landmarks were written by a newer
  MerXen or viewer (a higher schema version), the viewer refuses to open them; update the viewer or
  MerXen.

## 9. Coordinate conventions

- Landmark points are `(x, y)` in full-resolution pixels of the uncropped image; an integer is a
  pixel centre.
- Matrices map moving to reference (the Xenium Explorer convention).
- The CSV files MerXen writes use the full (uncropped) frames.
- Positive rotation is counter-clockwise on screen.
