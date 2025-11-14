from abc import ABC, abstractmethod

class BaseProcessingModule(ABC):
    def __init__(self, dataset_path, channel_names):
        self.DIR = dataset_path
        self.channel_names = channel_names
    
    @abstractmethod
    def action(self):
        pass
