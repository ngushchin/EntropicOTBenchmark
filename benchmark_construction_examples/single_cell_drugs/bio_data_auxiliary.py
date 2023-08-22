### Code from https://github.com/bunnech/cellot (Learning Single-Cell Perturbation Responses using Neural Optimal Transport)

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel
from torch.utils.data import DataLoader, Dataset
import anndata
from sklearn.model_selection import train_test_split
from scipy import sparse
from collections import MutableMapping
from itertools import groupby


def mmd_distance(x, y, gamma):
    xx = rbf_kernel(x, x, gamma)
    xy = rbf_kernel(x, y, gamma)
    yy = rbf_kernel(y, y, gamma)

    return xx.mean() + yy.mean() - 2 * xy.mean()


def compute_scalar_mmd(target, transport, gammas=None):
    if gammas is None:
        gammas = [2, 1, 0.5, 0.1, 0.01, 0.005]

    def safe_mmd(*args):
        try:
            mmd = mmd_distance(*args)
        except ValueError:
            mmd = np.nan
        return mmd

    return np.mean(list(map(lambda x: safe_mmd(target, transport, x), gammas)))



def load_data(config, **kwargs):
    data_type = config.get("data.type", "cell")
    if data_type in ["cell", "cell-merged", "tupro-cohort"]:
        loadfxn = load_cell_data

    elif data_type == "toy":
        loadfxn = load_toy_data

    else:
        raise ValueError

    return loadfxn(config, **kwargs)

def load_cell_data(
    config,
    data=None,
    split_on=None,
    return_as="loader",
    include_model_kwargs=False,
    pair_batch_on=None,
    **kwargs
):

    if isinstance(return_as, str):
        return_as = [return_as]

    assert set(return_as).issubset({"anndata", "dataset", "loader"})
    config.data.condition = config.data.get("condition", "drug")
    condition = config.data.condition

    if data is None:
        if config.data.type == "cell":
            data = read_single_anndata(config, **kwargs)
        else:
            raise ValueError

    if config.data.get("select") is not None:
        keep = pd.Series(False, index=data.obs_names)
        for key, value in config.data.select.items():
            if not isinstance(value, list):
                value = [value]
            keep.loc[data.obs[key].isin(value)] = True
            assert keep.sum() > 0

        data = data[keep].copy()

    if "dimension_reduction" in config.data:
        genes = data.var_names.to_list()
        name = config.data.dimension_reduction.name
        if name == "pca":
            dims = config.data.dimension_reduction.get(
                "dims", data.obsm["X_pca"].shape[1]
            )

            data = anndata.AnnData(
                data.obsm["X_pca"][:, :dims], obs=data.obs.copy(), uns=data.uns.copy()
            )
            data.uns["genes"] = genes

    if "ae_emb" in config.data:
        # load path to autoencoder
        assert config.get("model.name", "cellot") == "cellot"
        path_ae = Path(config.data.ae_emb.path)
        model_kwargs = {"input_dim": data.n_vars}
        config_ae = load_config(path_ae / "config.yaml")
        ae_model, _ = load_autoencoder_model(
            config_ae, restore=path_ae / "cache/model.pt", **model_kwargs
        )

        inputs = torch.Tensor(
            data.X if not sparse.issparse(data.X) else data.X.todense()
        )

        genes = data.var_names.to_list()
        data = anndata.AnnData(
            ae_model.eval().encode(inputs).detach().numpy(),
            obs=data.obs.copy(),
            uns=data.uns.copy(),
        )
        data.uns["genes"] = genes

    # cast to dense and check for nans
    if sparse.issparse(data.X):
        data.X = data.X.todense()
    assert not np.isnan(data.X).any()

    dataset_args = dict()
    model_kwargs = {}

    model_kwargs["input_dim"] = data.n_vars

    if config.get("model.name") == "cae":
        condition_labels = sorted(data.obs[condition].cat.categories)
        model_kwargs["conditions"] = condition_labels
        dataset_args["obs"] = condition
        dataset_args["categories"] = condition_labels

    if "training" in config:
        pair_batch_on = config.training.get("pair_batch_on", pair_batch_on)

    if split_on is None:
        if config.model.name == "cellot":
            # datasets & dataloaders accessed as loader.train.source
            split_on = ["split", "transport"]
            if pair_batch_on is not None:
                split_on.append(pair_batch_on)

        elif (config.model.name == "scgen" or config.model.name == "cae"
              or config.model.name == "popalign"):
            split_on = ["split"]

        else:
            raise ValueError

    if isinstance(split_on, str):
        split_on = [split_on]

    for key in split_on:
        assert key in data.obs.columns

    if len(split_on) > 0:
        splits = {
            (key if isinstance(key, str) else ".".join(key)): data[index]
            for key, index in data.obs[split_on].groupby(split_on).groups.items()
        }

        dataset = nest_dict(
            {
                key: AnnDataDataset(val.copy(), **dataset_args)
                for key, val in splits.items()
            },
            as_dot_dict=True,
        )

    else:
        dataset = AnnDataDataset(data.copy(), **dataset_args)

    if "loader" in return_as:
        kwargs = dict(config.dataloader)
        kwargs.setdefault("drop_last", True)
        loader = cast_dataset_to_loader(dataset, **kwargs)

    returns = list()
    for key in return_as:
        if key == "anndata":
            returns.append(data)

        elif key == "dataset":
            returns.append(dataset)

        elif key == "loader":
            returns.append(loader)

    if include_model_kwargs:
        returns.append(model_kwargs)

    if len(returns) == 1:
        return returns[0]

    return tuple(returns)


def read_single_anndata(config, path=None):
    if path is None:
        path = config.data.path

    data = anndata.read(path)

    if "features" in config.data:
        features = read_list(config.data.features)
        data = data[:, features].copy()

    # select subgroup of individuals
    if "individuals" in config.data:
        data = data[
            data.obs[config.data.individuals[0]].isin(config.data.individuals[1])
        ]

    # label conditions as source/target distributions
    # config.data.{source,target} can be a list now
    transport_mapper = dict()
    for value in ["source", "target"]:
        key = config.data[value]
        if isinstance(key, list):
            for item in key:
                transport_mapper[item] = value
        else:
            transport_mapper[key] = value

    data.obs["transport"] = data.obs[config.data.condition].apply(transport_mapper.get)

    if config.data["target"] == "all":
        data.obs["transport"].fillna("target", inplace=True)

    mask = data.obs["transport"].notna()
    assert "subset" not in config.data
    if "subset" in config.datasplit:
        for key, value in config.datasplit.subset.items():
            if not isinstance(value, list):
                value = [value]
            mask = mask & data.obs[key].isin(value)

    # write train/test/valid into split column
    data = data[mask].copy()
    if "datasplit" in config:
        data.obs["split"] = split_cell_data(data, **config.datasplit)

    return data


def split_cell_data_train_test(
    data, groupby=None, random_state=0, holdout=None, subset=None, **kwargs
):

    split = pd.Series(None, index=data.obs.index, dtype=object)
    groups = {None: data.obs.index}
    if groupby is not None:
        groups = data.obs.groupby(groupby).groups

    for key, index in groups.items():
        trainobs, testobs = train_test_split(index, random_state=random_state, **kwargs)
        split.loc[trainobs] = "train"
        split.loc[testobs] = "test"

    if holdout is not None:
        for key, value in holdout.items():
            if not isinstance(value, list):
                value = [value]
            split.loc[data.obs[key].isin(value)] = "ood"

    return split


def split_cell_data_train_test_eval(
    data,
    test_size=0.15,
    eval_size=0.15,
    groupby=None,
    random_state=0,
    holdout=None,
    **kwargs
):

    split = pd.Series(None, index=data.obs.index, dtype=object)

    if holdout is not None:
        for key, value in holdout.items():
            if not isinstance(value, list):
                value = [value]
            split.loc[data.obs[key].isin(value)] = "ood"

    groups = {None: data.obs.loc[split != "ood"].index}
    if groupby is not None:
        groups = data.obs.loc[split != "ood"].groupby(groupby).groups

    for key, index in groups.items():
        training, evalobs = train_test_split(
            index, random_state=random_state, test_size=eval_size
        )

        trainobs, testobs = train_test_split(
            training, random_state=random_state, test_size=test_size
        )

        split.loc[trainobs] = "train"
        split.loc[testobs] = "test"
        split.loc[evalobs] = "eval"

    return split


def split_cell_data_toggle_ood(data, holdout, key, mode, random_state=0, **kwargs):

    """Hold out ood sample, coordinated with iid split

    ood sample defined with key, value pair

    for ood mode: hold out all cells from a sample
    for iid mode: include half of cells in split
    """

    split = split_cell_data_train_test(data, random_state=random_state, **kwargs)

    if not isinstance(holdout, list):
        value = [holdout]

    ood = data.obs_names[data.obs[key].isin(value)]
    trainobs, testobs = train_test_split(ood, random_state=random_state, test_size=0.5)

    if mode == "ood":
        split.loc[trainobs] = "ignore"
        split.loc[testobs] = "ood"

    elif mode == "iid":
        split.loc[trainobs] = "train"
        split.loc[testobs] = "ood"

    else:
        raise ValueError

    return split


def split_cell_data(data, name="train_test", **kwargs):
    if name == "train_test":
        split = split_cell_data_train_test(data, **kwargs)
    elif name == "toggle_ood":
        split = split_cell_data_toggle_ood(data, **kwargs)
    elif name == "train_test_eval":
        split = split_cell_data_train_test_eval(data, **kwargs)
    else:
        raise ValueError

    return split.astype("category")

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
        
def nest_dict(d, sep=".", as_dot_dict=False):
    result = DotDict() if as_dot_dict else dict()
    for k, v in d.items():
        # for each key split_rec splits keys to form recursively nested dict
        split_rec(k, v, result, sep=sep, as_dot_dict=as_dot_dict)
    return result


class AnnDataDataset(Dataset):
    def __init__(
        self, adata, obs=None, categories=None, include_index=False, dim_red=None
    ):
        self.adata = adata
        self.adata.X = self.adata.X.astype(np.float32)
        self.obs = obs
        self.categories = categories
        self.include_index = include_index

    def __len__(self):
        return len(self.adata)

    def __getitem__(self, idx):
        value = self.adata.X[idx]

        if self.obs is not None:
            meta = self.categories.index(self.adata.obs[self.obs].iloc[idx])
            value = value, int(meta)

        if self.include_index:
            return self.adata.obs_names[idx], value

        return value
    
    
class DotDict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
def split_rec(k, v, out, sep=".", as_dot_dict=False):
    # splitting keys in dict, calling recursively to break items on '.'
    k, *rest = k.split(sep, 1)
    if rest:
        split_rec(rest[0], v, out.setdefault(k, DotDict() if as_dot_dict else dict()))
    else:
        out[k] = v
        
        
def cast_dataset_to_loader(dataset, **kwargs):
    # check if dataset is torch.utils.data.Dataset
    if isinstance(dataset, Dataset):
        return DataLoader(dataset, **kwargs)

    batch_size = kwargs.pop('batch_size', 1)
    flat_dataset = flat_dict(dataset)

    minimum_batch_size = {
        group: min(*map(lambda x: len(flat_dataset[x]), keys), batch_size)
        for group, keys
        in groupby(flat_dataset.keys(), key=lambda x: x.split('.')[0])
    }

    min_bs = min(minimum_batch_size.values())
    if batch_size != min_bs:
        logging.warn(f'Batch size adapted to {min_bs} due to dataset size.')

    loader = nest_dict({
        key: DataLoader(
            val,
            batch_size=minimum_batch_size[key.split('.')[0]],
            **kwargs)
        for key, val
        in flat_dataset.items()
    }, as_dot_dict=True)

    return loader

def flat_dict(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flat_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)