from abc import ABC, abstractmethod

class FeatureAttribution(ABC):

    def __init__(self, model):
        self.model = model
        
    @abstractmethod
    def explain_instance(self, query, query_label=None, nun=None, **kwargs):
        pass

