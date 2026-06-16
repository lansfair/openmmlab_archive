# Copyright (c) OpenMMLab. All rights reserved.
from .dior import DIORDataset  # noqa: F401, F403
from .dota import DOTAv2Dataset  # noqa: F401, F403
from .dota import DOTADataset, DOTAv15Dataset, DIOR_DOTADataset
from .hrsc import HRSCDataset  # noqa: F401, F403
from .transforms import *  # noqa: F401, F403

__all__ = [
    'DOTADataset', 'DOTAv15Dataset', 'DOTAv2Dataset', 'HRSCDataset',
    'DIORDataset', 'DIOR_DOTADataset'
]
