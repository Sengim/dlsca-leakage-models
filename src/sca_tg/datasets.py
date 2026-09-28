"""Dataset registry for the scripts: --dataset name -> (loader, default file)."""

import os

from sca_tg.ascadr import load_ascadr
from sca_tg.ches_ctf import load_ches_ctf
from sca_tg.eshard import load_eshard

DATASETS = {
    "ascadr": (load_ascadr, "ascad-variable.h5"),  # also ascad-variable-desync50.h5
    "eshard": (load_eshard, "eshard.h5"),
    "ches_ctf": (load_ches_ctf, "ches_ctf.h5"),
}


def load_dataset(name, file_path, **kwargs):
    return DATASETS[name][0](file_path, **kwargs)


def default_path(name):
    if name == "ascadr":
        return os.environ.get("ASCADR_PATH", DATASETS[name][1])
    return DATASETS[name][1]
