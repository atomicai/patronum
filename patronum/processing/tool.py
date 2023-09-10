import logging
import numbers

import numpy as np
import torch
from torch.utils.data import TensorDataset

from patronum.etc import flatten_list

logger = logging.getLogger(__name__)


def convert_features_to_dataset(features):
    """
    Converts a list of feature dictionaries (one for each sample) into a PyTorch Dataset.

    :param features: A list of dictionaries. Each dictionary corresponds to one sample. Its keys are the
                     names of the type of feature and the keys are the features themselves.
    :Return: a Pytorch dataset and a list of tensor names.
    """
    # features can be an empty list in cases where down sampling occurs
    if len(features) == 0:
        return None, None
    tensor_names = list(features[0].keys())
    all_tensors = []
    for t_name in tensor_names:
        try:
            # Checking whether a non-integer will be silently converted to torch.long
            check = features[0][t_name]
            if isinstance(check, numbers.Number):
                base = check
            # extract a base variable from a nested lists or tuples
            elif isinstance(check, list):
                base = list(flatten_list(check))[0]
            # extract a base variable from numpy arrays
            else:
                base = check.ravel()[0]
            if not np.issubdtype(type(base), np.integer):
                logger.warning(
                    "Problem during conversion to torch tensors:\n"
                    "A non-integer value for feature '%s' with a value of: "
                    "'%s' will be converted to a torch tensor of dtype long.",
                    t_name,
                    base,
                )
        except:
            logger.debug(
                "Could not determine type for feature '%s'. Converting now to a tensor of default type long.", t_name
            )

        # Convert all remaining python objects to torch long tensors
        cur_tensor = torch.as_tensor(np.array([sample[t_name] for sample in features]), dtype=torch.long)

        all_tensors.append(cur_tensor)

    dataset = TensorDataset(*all_tensors)
    return dataset, tensor_names


__all__ = ["convert_features_to_dataset"]
