import os
import sys

class MultiSpectralProcessor:
    def __init__(self, dataset_path, channel_names, *step_classes):
        self.dataset_path = dataset_path
        self.channel_names = channel_names
        self.steps = [step_class(dataset_path, channel_names) for step_class in step_classes]

    def process(self):
        result = None
        for step in self.steps:
            result = step.action()
        return result
