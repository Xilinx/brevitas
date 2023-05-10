import logging
from multiprocessing import Manager
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import GPUtil
from hydra.core.config_loader import ConfigLoader
from hydra.core.hydra_config import HydraConfig
from hydra.core.singleton import Singleton
from hydra.core.utils import configure_log
from hydra.core.utils import filter_overrides
from hydra.core.utils import JobReturn
from hydra.core.utils import run_job
from hydra.core.utils import setup_globals
from hydra.plugins.launcher import Launcher
from hydra.types import HydraContext
from hydra.types import TaskFunction
from joblib import delayed
from joblib import Parallel
from omegaconf import DictConfig
from omegaconf import open_dict

log = logging.getLogger(__name__)


class JoblibLauncher(Launcher):

    def __init__(self, **kwargs: Any) -> None:
        """Joblib Launcher
        Launches parallel jobs using Joblib.Parallel. For details, refer to:
        https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html
        This plugin is based on the idea and inital implementation of @emilemathieutmp:
        https://github.com/facebookresearch/hydra/issues/357
        """
        self.config: Optional[DictConfig] = None
        self.config_loader: Optional[ConfigLoader] = None
        self.task_function: Optional[TaskFunction] = None

        self.joblib = kwargs

    def setup(
        self,
        *args,
        hydra_context: HydraContext,
        task_function: TaskFunction,
        config: DictConfig,
    ) -> None:
        self.config = config
        self.task_function = task_function
        self.hydra_context = hydra_context

    def launch(self, job_overrides: Sequence[Sequence[str]],
               initial_job_idx: int) -> Sequence[JobReturn]:
        """
        :param job_overrides: a List of List<String>, where each inner list is the arguments for one job run.
        :param initial_job_idx: Initial job idx in batch.
        :return: an array of return values from run_job with indexes corresponding to the input list indexes.
        """
        setup_globals()
        assert self.config is not None
        assert self.hydra_context is not None
        assert self.task_function is not None

        configure_log(self.config.hydra.hydra_logging, self.config.hydra.verbose)
        sweep_dir = Path(str(self.config.hydra.sweep.dir))
        sweep_dir.mkdir(parents=True, exist_ok=True)

        # Joblib's backend is hard-coded to loky since the threading
        # backend is incompatible with Hydra
        joblib_cfg = self.joblib
        joblib_cfg["backend"] = "loky"
        log.info(
            "Joblib.Parallel({}) is launching {} jobs".format(
                ",".join([f"{k}={v}" for k, v in joblib_cfg.items()]),
                len(job_overrides),
            ))
        log.info("Launching jobs, sweep output dir : {}".format(sweep_dir))
        for idx, overrides in enumerate(job_overrides):
            log.info("\t#{} : {}".format(idx, " ".join(filter_overrides(overrides))))

        singleton_state = Singleton.get_state()

        # Alessandro: declare manager and tuple of GPU semaphores
        m = Manager()
        gpu_semaphores = tuple(
            m.BoundedSemaphore(joblib_cfg["n_jobs"]) for i in range(joblib_cfg["n_gpus"]))
        joblib_cfg.pop("n_gpus", None)
        runs = Parallel(**joblib_cfg)(
            delayed(execute_job)(
                initial_job_idx + idx,
                overrides,
                self.hydra_context,
                self.config,
                self.task_function,
                singleton_state,
                gpu_semaphores,  # Alessandro: pass gpu semaphores
            ) for idx,
            overrides in enumerate(job_overrides))

        assert isinstance(runs, List)
        for run in runs:
            assert isinstance(run, JobReturn)
        return runs


def execute_job(
        idx: int,
        overrides: Sequence[str],
        hydra_context: HydraContext,
        config: DictConfig,
        task_function: TaskFunction,
        singleton_state: Dict[Any, Any],
        gpu_semaphores,  # Alessandro: accept gpu semaphores
) -> JobReturn:
    """Calls `run_job` in parallel
    """
    setup_globals()
    Singleton.set_state(singleton_state)

    sweep_config = hydra_context.config_loader.load_sweep_config(config, list(overrides))
    with open_dict(sweep_config):
        sweep_config.hydra.job.id = "{}_{}".format(sweep_config.hydra.job.name, idx)
        sweep_config.hydra.job.num = idx
    HydraConfig.instance().set_config(sweep_config)

    # Alessandro: acquire lock on gpu and set it in the env
    gpu_id = idx % len(gpu_semaphores)
    GPUs = GPUtil.getGPUs()
    available_gpus = GPUtil.getAvailability(
        GPUs, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[], excludeUUID=[])
    if available_gpus[gpu_id] == 0:
        gpu_id = GPUtil.getAvailable(
            order='memory',
            limit=3,
            maxLoad=0.5,
            maxMemory=0.5,
            includeNan=False,
            excludeID=[],
            excludeUUID=[])[0]
    print(idx, gpu_id)

    gpu_semaphores[gpu_id].acquire()
    os.environ["GPU_ID"] = str(gpu_id)
    ret = run_job(
        config=sweep_config,
        task_function=task_function,
        hydra_context=hydra_context,
        job_dir_key="hydra.sweep.dir",
        job_subdir_key="hydra.sweep.subdir",
    )

    # Alessandro: release lock on gpu
    gpu_semaphores[gpu_id].release()
    return ret
