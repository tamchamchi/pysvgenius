from pysvgenius.common.registry import registry
import os
from iopath.common.file_io import g_pathmgr


def now():
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d%H%M")[:-1]


def get_abs_path(rel_path):
    return os.path.join(registry.get_path("library_root"), rel_path)


def makedir(dir_path):
    """
    Create the directory if it does not exist.
    """
    is_success = False
    try:
        if not g_pathmgr.exists(dir_path):
            g_pathmgr.mkdirs(dir_path)
        is_success = True
    except BaseException:
        print(f"Error creating directory: {dir_path}")
    return is_success


