"""Xenium density is discoverable for viewing without changing HITL inputs."""

import hashlib
import json

import pytest

from napari_compare_xenium_merscope.density_image import embedded_density_record
from napari_compare_xenium_merscope.platforms import platform_spec


def test_xenium_density_binding_is_accepted_as_a_viewer_image(tmp_path):
    key = 'xenium_transcript_density'
    (tmp_path / 'images' / key).mkdir(parents=True)
    record = {'platform': 'XENIUM', 'fingerprint': 'test-record'}
    binding = {'schema_version': 1, 'image_key': key, 'record': record}
    binding['binding_sha256'] = hashlib.sha256(
        json.dumps(binding, sort_keys=True).encode()
    ).hexdigest()
    metadata = tmp_path / 'zarr.json'
    metadata.write_text(json.dumps({'attributes': {'merxen_transcript_density': binding}}))
    assert embedded_density_record(tmp_path) == record
    spec = platform_spec('XENIUM')
    assert dict(spec.image_channel_aliases)[(key, 'transcript_density')] == 'Transcript density'
    assert spec.channel_colormap('transcript_density') == 'red'
    binding['image_key'] = 'unexpected_image'
    metadata.write_text(json.dumps({'attributes': {'merxen_transcript_density': binding}}))
    with pytest.raises(ValueError, match='incomplete or inconsistent'):
        embedded_density_record(tmp_path)


def test_xenium_crop_export_keeps_density_out_of_model_arrays(tmp_path):
    import numpy as np
    import spatialdata as sd
    from spatialdata.models import Image2DModel, Labels2DModel

    from napari_compare_xenium_merscope.hitl_crops import CropRectangle, HitlCropContext
    from napari_compare_xenium_merscope.hitl_package import (
        AnnotationExportSettings,
        export_local_annotation_package,
    )

    source = tmp_path / 'review.zarr'
    yy, xx = np.indices((512, 512), dtype=np.float32)
    labels = np.zeros((512, 512), np.uint32)
    labels[100:130, 100:130] = 1
    key = 'xenium_transcript_density'
    record = {'platform': 'XENIUM', 'fingerprint': 'viewer-only'}
    binding = {'schema_version': 1, 'image_key': key, 'record': record}
    binding['binding_sha256'] = hashlib.sha256(
        json.dumps(binding, sort_keys=True).encode()
    ).hexdigest()
    sd.SpatialData(
        images={
            'morphology_focus': Image2DModel.parse(
                np.stack([yy, xx]), dims=('c', 'y', 'x'), c_coords=['DAPI', '18S']
            ),
            key: Image2DModel.parse(
                np.full((1, 512, 512), 1000, np.float32),
                dims=('c', 'y', 'x'), c_coords=['transcript_density']
            ),
        },
        labels={'MOSAIK_cellpose_labels': Labels2DModel.parse(labels, dims=('y', 'x'))},
        attrs={'merxen_transcript_density': binding},
    ).write(source)
    context = HitlCropContext(
        'XENIUM', 'morphology_focus', 'HD1', 512, 512, 0.5, 2048, 64, 32, 768,
        hashlib.sha256((source / 'zarr.json').read_bytes()).hexdigest(), '2' * 64,
    )
    package = export_local_annotation_package(
        source, context, [CropRectangle(64, 64, 320, 320).vertices()],
        AnnotationExportSettings(('DAPI', '18S'), 'joint', None, None),
    )
    assert package.manifest['model_channels'] == ['DAPI', '18S']
    assert package.manifest['transcript_density'] is None
    raw = np.load(package.root / 'raw/crop_001.npy', allow_pickle=False)
    model = np.load(package.root / 'model_input/crop_001.npy', allow_pickle=False)
    assert raw.shape == (2, 320, 320)
    assert model.shape == (3, 320, 320)
    assert not model[2].any()
