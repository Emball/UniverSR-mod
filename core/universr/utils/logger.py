from abc import ABC, abstractmethod

import wandb

class BaseLogger(ABC):
    """Logger abstract base class."""
    @abstractmethod
    def log(self, data: dict, step: int):
        pass

    @abstractmethod
    def finish(self):
        pass

class WandbLogger(BaseLogger):
    """Logs data to Weights & Biases."""
    def __init__(self, **kwargs):
        wandb.init(**kwargs)

    def log(self, data: dict, step: int):
        wandb.log(data, step=step)

    def finish(self):
        wandb.finish()

class DummyLogger(BaseLogger):
    """A logger that does nothing."""
    def __init__(self,config,**kwargs):
        self.run_name = config.get('wandb',{}).get('run_name', 'dummy')
        
    def log(self, data: dict, step: int):
        # Do nothing
        pass

    def finish(self):
        # Do nothing
        pass
    
def get_logger(config, wandb_flag=False):
    if wandb_flag:
        print("INFO: Initializing WandbLogger.")
        try:
            return WandbLogger(
                project=config.wandb.project_name,
                entity=config.wandb.entity,
                config=config.to_dict(),
                name=config.wandb.run_name,
                notes=config.wandb.get('notes', ''),
                save_code=True
            )
        except AttributeError as e:
            raise AttributeError(f"Missing wandb configuration in your config file: {e}")
    else:
        print("INFO: Initializing DummyLogger.")
        return DummyLogger(config=config)