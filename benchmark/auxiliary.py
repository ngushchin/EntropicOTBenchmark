from os import environ, listdir, makedirs
from os.path import expanduser, isdir, join, splitext


# adapted from https://github.com/scikit-learn/scikit-learn/blob/8c9c1f27b/sklearn/datasets/_base.py#L37
def get_data_home(data_home=None) -> str:
    """Return the path of the benchmark data directory.
    This folder is used by dataset loaders to avoid downloading the
    data several times.
    By default the data directory is set to a folder named 'eot_benchmark_data' in the
    user home folder.
    Alternatively, it can be set by the 'EOT_BENCHMARK_DATA' environment
    variable or programmatically by giving an explicit folder path. The '~'
    symbol is expanded to the user home folder.
    If the folder does not already exist, it is automatically created.
    Parameters
    ----------
    data_home : str, default=None
        The path to scikit-learn data directory. If `None`, the default path
        is `~/sklearn_learn_data`.
    Returns
    -------
    data_home: str
        The path to scikit-learn data directory.
    """
    if data_home is None:
        data_home = environ.get("EOT_BENCHMARK_DATA", join("~", "eot_benchmark_data"))
    data_home = expanduser(data_home)
    makedirs(data_home, exist_ok=True)
    return data_home
