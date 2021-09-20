from abc import ABC, abstractmethod

class FeatureRelevance(ABC):

    def __init__(self, model, feature_names, **kwargs):
        self.model = model
        self.feature_names = feature_names
        
    @abstractmethod
    def explain_instance(self, query, query_label, **kwargs):
        pass

