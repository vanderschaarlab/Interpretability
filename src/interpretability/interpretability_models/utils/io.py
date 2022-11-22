import os
import dill as pkl
from copy import deepcopy
import torch

cwd = os.path.abspath(".")

pkl.settings["recurse"] = True


def save_explainer(explainer, save_path):
    save_path = os.path.join(cwd, save_path)
    with open(save_path, "wb") as f:
        pkl.dump(explainer, f)


def load_explainer(save_path, join_to_cwd_to_save_path=True):
    if join_to_cwd_to_save_path:
        with open(os.path.join(cwd, save_path), "rb") as f:
            return pkl.load(f)
    else:
        with open(save_path, "rb") as f:
            return pkl.load(f)


def check_attribute_eq(attribute, explainer, explainer_from_file):
    print(f"Comparing {attribute}")
    if isinstance(getattr(explainer, attribute), torch.Tensor):
        exp = torch.equal(
            getattr(explainer, attribute), getattr(explainer_from_file, attribute)
        )
    else:
        exp = getattr(explainer, attribute) == getattr(explainer_from_file, attribute)
    if not exp:
        try:
            assert getattr(explainer, attribute) != deepcopy(
                getattr(explainer, attribute)
            )
            print(f"\t{attribute} not comparible")
        except AssertionError as e:
            print(f"\t{attribute} is not equal")
            raise e
