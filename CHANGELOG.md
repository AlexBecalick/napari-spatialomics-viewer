# Changelog

All notable changes to the viewer are recorded here. The format follows
[Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/), and the viewer
uses [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).
[Versioning and releases](docs/versioning.md) says what counts as a major,
minor or patch change for this viewer and how a release is made.

A pull request that changes what the viewer does, what it reads or writes, or
how it is installed adds its entry under [Unreleased] (CI checks this; the
`skip-changelog` label exempts a PR with nothing users would notice). A release
turns [Unreleased] into the release's section, and the release's GitHub notes
are that section.

## [Unreleased]

## [0.5.0] - 2026-11-13

Manual landmark registration, standalone 10x Atera stores, hand-curated cell
groups and human-in-the-loop Cellpose annotation, and the first release made
from this changelog. Version 0.4.0 was set in `pyproject.toml` but never
released; its manual-registration work ships here. Builds and saved files that
report 0.4.0 come from `main` between 0.3.0 and this release.

### Highlights

- **Manual registration.** Place landmark pairs between two MerXen tissue images in a new tab, with live residuals, draft saving and a read-only result view.
- **Atera stores.** Open whole-transcriptome 10x Atera stores standalone, at about 1.4 billion transcripts and 18k genes, with lazy transcripts, a searchable gene inspector and saved atlas predictions.
- **Cell groups.** Click Cellpose cells into named groups and export them with their per-channel statistics.
- **HITL segmentation crops.** Draw crops with live cell estimates, then annotate and track Cellpose correction packages for MerXen training.
- **Versioned releases.** `--version`, this changelog, and releases published from it ([versioning](docs/versioning.md)).

### Upgrading

- The new manual-registration and HITL features need recent MerXen outputs (see Requirements). Existing MERSCOPE, Xenium and paired stores open as before.
- A developer install (`pip install -e .`) reports its version from the install metadata: re-run it after pulling so `--version` shows 0.5.0.

### Added

#### Manual landmark registration

- A new **Manual registration** tab and a **Manual Registration ▸ Load image data** section in the Dataset loader open a MerXen `<id>_prepared.zarr` (to place landmarks) or `<run_id>_registered.zarr` (to inspect a result). `--registration-store <path>` opens a store at startup ([guide](docs/manual_registration_guide.md)).
- Crops on each image to exclude tile-seam borders and coverslip artefacts.
- Coarse alignment with large rotations and flips: a 180° button, Flip H / Flip V, rotation in steps from ±90° down to ±0.1°, and shifts in selectable steps.
- Flicker comparison between the two images (F, hold Shift+F to peek) and auto-switching after each click.
- Landmark pairs with live fit and leave-one-out residuals, warnings, and held-out (disabled) pairs.
- Draft and finish saving with conflict detection, a history of previous payloads, and local drafts offered after an interrupted session.
- A read-only result view with the registered image, landmark overlays, a residual table and a banner with the preset, refinement and flow status.

#### Atera stores

- A **Load new standalone Atera dataset** button, `--atera-zarr` and `--atera-spec-path` open MerXen 10x Atera (whole-transcriptome) stores, with transcript, gene-panel and cell-inspector paths that scale to about 1.4 billion transcripts and 18k genes. Recently viewed remembers Atera stores.
- Detection from MerXen metadata: a store opened through another platform's button, a recent entry or the CLI is re-labelled only when its root `merxen_schema.platform` or `merxen_native_latest.json` says so, with a status-bar message.
- Pixel size from `atera/experiment.spatial` (or the override), else the root `merxen_atera` attributes or the cell-boundary scale, with a warning. A store with none of these is refused.
- The vendor cell outlines load at startup. The nucleus mask is listed once, and a nucleus click selects its cell through `cell_instance_id`. MerXen cell types, the UMAP and per-cell image statistics join on the vendor cells. The vendor masks, label aliases and payload are never modified.
- Transcripts are read lazily from the parquet tiles. **Whole-slide sample** mode (default) keeps a nested sample sized from RAM and fills in the visible field at full density; **Lazy** mode shows a 5 µm density image and reads only the ticked genes. The overview, code-range map, sample and cell-to-tile map are cached outside the store.
- `--atera-overview-max-points` (default 40M, `0` turns it off) caps the all-genes view at a nested sub-sample of the loaded sample; ticked genes keep the whole point budget, and a zoomed field stays complete.
- Gene inspector selection mode for Atera and any panel above 2,000 features: search first, up to 32 colour slots, control features folded under one heading, and an all-transcript overview switch.
- Cell inspector without a whole-store index: table counts first, then the cell's exact transcripts (about 0.3 s per cell), shown as a ranked bar legend beside a top-10 pie.
- Saved SEA-AD subclass and Allen WHB supercluster predictions, separate from accepted labels, with a shared bootstrap cut-off for spatial and UMAP colouring. Zero shows every available prediction, including expressed spatial cells outside clustering QC. Bootstrap support measures stability, not biological probability.

#### Cell groups

- A **Cell groups** tab (standalone mode): with **Pick cells for group** armed, a click toggles the Cellpose cell under the cursor in the selected named group. Groups may overlap and are outlined in their own colours.
- Export writes one CSV row per (group, cell) with identity columns, centroid and area in microns, and every per-channel statistic read from `table_MOSAIK_cellpose_image_quantification`. Groups can be reloaded from an export.

#### HITL segmentation crops

- A **HITL seg crops** tab draws rectangular regions for a prefilled Cellpose correction package on native Atera and Xenium views, shows the estimated Cellpose cell count (green from 800 to 1,200) and saves the regions ([README](README.md#hitl-segmentation-crops)).
- **Save areas and prepare annotation** writes a managed `hitl-<timestamp>-<id>/` package; **Annotate cell masks** opens each crop in a separate Cellpose 4.2.1.1 environment and tracks saved progress. **Open annotation package…** resumes later, and older packages are upgraded with their mask edits and evaluation splits preserved.
- Embedded transcript density is loaded automatically for viewing and for Atera HITL inputs; separate `transcript_density.zarr` files remain supported for older stores.
- Xenium stores prepared by MerXen show their **Transcript density** as a red annotation guide. The Xenium HITL inputs stay DAPI + 18S RNA.

#### Other

- Determinate progress bars on every platform, including Atera streaming, sample restore, eager and paired builds, masks, images and gene-layer creation.
- `--version` prints the installed viewer version.

### Changed

- The Manual registration tab is appended after Zarr metadata; the existing tab order is unchanged. Recently viewed remembers registration stores, and the standalone loaders open a registration store in that tab instead of as a standalone dataset.
- Image-only stores without transcripts skip the transcript autoload.
- Display rotation is set to 0° and disabled in a registration session; registration layers are never rotated by it.
- The gene panel is a virtualised list: a 24.7k-row panel populates in about 0.04 s instead of 224 s.
- The Xenium/MERSCOPE eager transcript build is vectorised (about 5–7 times faster on a large section) with bit-identical output, and no longer fails on stores of 1e9 or more transcripts.
- Paired viewing stays MERSCOPE + Xenium only. An Atera store given to the paired loader is refused before anything is unloaded.
- pyarrow is a declared dependency and is bundled in the installers.
- napari is pinned below 0.8.
- Releases are made from this changelog: merging a release PR tags `vX.Y.Z`, builds the installers and publishes the release with this file's section as its notes. Patch versions are now released and tagged like any other ([versioning](docs/versioning.md)).

### Fixed

- Chunked gene-layer builds finish correctly across a dataset switch, and the outgoing dataset no longer overwrites the loading status or evicts sessions mid-switch.
- A partially applied Atera sample partition is rolled back on `MemoryError`, so the out-of-memory retry never builds points from misaligned columns.
- Gene-inspector and Atera controls no longer clip at the panel's 240 px minimum width or sit under the macOS overlay scrollbar.
- The stored-means cell is named by its source (for example "Atera cell" or "Cellpose cell") on every platform.
- HITL crop preparation recognises native image sources independently of display names, and writes package text as UTF-8.

### Requirements

- Manual registration needs stores written by a MerXen version with the manual-registration workflows, schema version 1. The viewer refuses newer schema versions.
- HITL region GeoJSON (`merxen_hitl_regions_version: 2`) holds rectangles: export it with MerXen's rectangle-enabled `cellpose-hitl-export`; older square-only exporters refuse it.

### Limitations

- Not available for Atera: cortical-depth and distance-from-object annotations, and pciSeq sources (their MerXen exports are defined for MERSCOPE and Xenium; the disabled controls say so), vendor Azimuth cell types and UMAP, and a bundled whole-transcriptome marker reference (the gene list is ordered A–Z unless the store embeds one).
- See [performance notes](docs/performance_notes.md) for measured timings and memory use.

## [0.3.0] - 2026-10-13

Compare aligned MERSCOPE and Xenium datasets in one napari window with shared
controls, linked cell inspection, and spatial/UMAP region selection.

### Added

#### Paired viewing

- **Side-by-side** mode with MERSCOPE on the left and Xenium on the right, linked navigation, paired layer visibility, and shared image, segmentation, gene, and cell-type controls.
- **Stacked overlay** mode displays both platforms in their common coordinate system, with independent visibility for platform-qualified layers.
- Switching between layouts without reopening stores or rebuilding layers. Independent stacked-overlay visibility settings are preserved.
- Cells and transcripts are inspected at the same aligned coordinate in both datasets.
- Paired sessions launch from the Dataset loader or with `--paired-view side-by-side` / `--paired-view stacked-overlay` and both dataset paths.

#### UMAPs, regions, and cell types

- Linked spatial and UMAP selections, including separate MERSCOPE and Xenium UMAPs with consistent cell-type colours across platforms.
- Regions drawn in spatial views or UMAPs highlight the selected cells and show cell-type counts and percentages. Spatial regions apply to both aligned stores; a UMAP region applies to its own platform.
- MerXen cell-type hierarchies and hybrid masks, including discovery of named map-first annotation tables.
- Shared dataset rotation controls and marker-reference catalog tooling with panel-matched reference support.

### Changed

- The storage stack matches MerXen: SpatialData 0.8, Zarr 3, and AnnData 0.12 (below 0.13).

### Fixed

- Cell-type gene ordering is restored across derived stores, cell-type masks render on first selection, and stale cell-type ID joins are prevented.
- Intel macOS packaging keeps compatible Numba, llvmlite, and NumPy binary versions.

### Requirements

- Paired viewing requires matching, complete **MerXen v2 materialized stores**: a MERSCOPE `merxen_alignment` manifest and the corresponding Xenium `merxen_alignment_pair_reference`. Paired mode validates the pair and consumes it read-only; it does not perform registration or repair missing artifacts.

### Limitations

- Per-cell statistic overlays and tissue-annotation creation/export are unavailable while a pair is active. Each platform's UMAP comes from its own store; the embeddings are not harmonised.

## [0.2.0] - 2026-07-17

### Added

- Object polygon annotations.
- A **Cell type labels** tab colours cell masks by type.
- Cell types in the cell inspector and the cursor status.
- Cancellable, progressive background loading, and cached prepared dataset sessions for fast switching.
- Cell inspection statistics are computed asynchronously.
- Tagged desktop releases built by GitHub Actions.

### Changed

- Transcripts stream into compact bounded stores and are filtered with visibility masks; raster pan and zoom are more responsive.

### Removed

- The "Export Separate GeoJSONs" cortical-depth export.

### Fixed

- Cell-type mask colouring uses true instance ids, the cache format and the full-resolution grid.
- Inspection hit-testing is skipped while panning, a blank canvas is refreshed after progressive loading, and a data-loading visibility bug is fixed.

## [0.1.0] - 2026-07-14

First public release of the interactive napari viewer for Xenium and MERSCOPE
SpatialData datasets.

### Added

- Standalone or paired MERSCOPE and Xenium SpatialData stores.
- Interactive transcript and gene-expression inspection.
- Image channels and cell-segmentation results.
- Per-cell statistics, calculated and visualised.
- Tissue annotations.
- Launch without a dataset and select data through the Dataset loader.
- Native desktop packages for macOS, Windows, and Linux.

[Unreleased]: https://github.com/AlexBecalick/napari-spatialomics-viewer/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/AlexBecalick/napari-spatialomics-viewer/compare/v0.3.0...v0.5.0
[0.3.0]: https://github.com/AlexBecalick/napari-spatialomics-viewer/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/AlexBecalick/napari-spatialomics-viewer/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/AlexBecalick/napari-spatialomics-viewer/releases/tag/v0.1.0
