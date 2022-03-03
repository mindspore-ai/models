import pprint

from src.config import check_config
from src.log import log


def process_cfg(func):
    """
    process config decorator
    """

    def wrapper(cfg, *args, **kwargs):
        cfg = check_config(cfg)
        log.info("config: %s", pprint.pformat(cfg))
        return func(cfg, *args, **kwargs)

    return wrapper
