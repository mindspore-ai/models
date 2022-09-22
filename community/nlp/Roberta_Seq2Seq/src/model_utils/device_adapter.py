"""Device adapter for ModelArts"""

from src.model_utils.config import config

if config.enable_modelarts:
    from src.model_utils.moxing_adapter import get_device_id, get_device_num, get_rank_id, get_job_id
else:
    from src.model_utils.local_adapter import get_device_id, get_device_num, get_rank_id, get_job_id

__all__ = [
    "get_device_id", "get_device_num", "get_rank_id", "get_job_id"
]
