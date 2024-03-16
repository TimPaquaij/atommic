# coding=utf-8
__author__ = "Tim Paquaij"

from atommic.collections.multitask.rso.parts.transforms import RSOMRIDataTransforms

__all__ = ["ObjectdetectionMRIDataTransforms"]


class ObjectdetectionMRIDataTransforms(RSOMRIDataTransforms):
    """Transforms for the MRI segmentation task.

    .. note::
        Extends :class:`atommic.collections.multitask.rs.parts.transforms.RSMRIDataTransforms`.
    """
