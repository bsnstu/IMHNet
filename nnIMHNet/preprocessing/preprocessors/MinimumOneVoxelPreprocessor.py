"""Keep every resampled axis at least one voxel for thin-volume data."""
import numpy as np
from nnunetv2.preprocessing.preprocessors import default_preprocessor as base


class MinimumOneVoxelPreprocessor(base.DefaultPreprocessor):
    def run_case_npy(self, *args, **kwargs):
        original = base.compute_new_shape
        def minimum_shape(*shape_args, **shape_kwargs):
            return np.maximum(1, original(*shape_args, **shape_kwargs))
        base.compute_new_shape = minimum_shape
        try:
            return super().run_case_npy(*args, **kwargs)
        finally:
            base.compute_new_shape = original
