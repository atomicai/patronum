import abc


class ILogger(abc.ABC):
    """
    Base class for tracking experiments.

    This class can be extended to implement custom logging backends like MLFlow, Tensorboard, or Sacred.
    """

    disable_logging = False

    def __init__(self, tracking_uri, **kwargs):
        """
        Most of the remote logging platforms provide url for your experiment
        """
        self.tracking_uri = tracking_uri

    @classmethod
    @abc.abstractmethod
    def init_experiment(cls, **kwargs):
        raise NotImplementedError()

    @classmethod
    @abc.abstractmethod
    def end_run(cls, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def log_metrics(self, metrics, step, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def log_artifacts(self, artifacts, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def log_params(self, params, **kwargs):
        raise NotImplementedError()
