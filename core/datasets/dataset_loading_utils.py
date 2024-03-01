import itertools
from enum import Enum
from typing import Dict, List, Optional

import torch

from .vq_dataset import VQSMPLXMotionDataset


class AIST_GENRE(Enum):
    GBR = "Break"
    GPO = "Pop"
    GLO = "Lock"
    GMH = "Middle Hip-hop"
    GLH = "LA style Hip-hop"
    GHO = "House"
    GWA = "Waack"
    GKR = "Krump"
    GJS = "Street Jazz"
    GJB = "Ballet Jazz"


dataset_names = [
    "animation",
    "humanml",
    "perform",
    "GRAB",
    "idea400",
    "humman",
    "beat",
    "game_motion",
    "music",
    "aist",
    "fitness",
    "moyo",
    "choreomaster",
    "dance",
    "kungfu",
    "EgoBody",
    "HAA500",
]


def load_dataset(
    dataset_args,
    model_args,
    dataset_names=dataset_names,
    split: str = "train",
    weight_scale: Optional[List[int]] = None,
    dataset_class=VQSMPLXMotionDataset,
):
    if weight_scale is None:
        weight_scale = [1] * len(dataset_names)
    assert len(dataset_names) == len(weight_scale), "mismatch in size"
    dataset_list = []
    weights = []
    for dataset_name in dataset_names:
        dataset_list.append(
            dataset_class(
                dataset_name,
                dataset_args=dataset_args,
                model_args=model_args,
                split=split,
            )
        )

    concat_dataset = torch.utils.data.ConcatDataset(dataset_list)

    if split != "train" or len(dataset_names) == 1:
        return concat_dataset, None, None

    for i, ds in enumerate(dataset_list):
        weights.append(
            [weight_scale[i] * concat_dataset.__len__() / (ds.__len__())] * ds.__len__()
        )

    weights = list(itertools.chain.from_iterable(weights))

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=weights, num_samples=len(weights)
    )

    return concat_dataset, sampler, weights
