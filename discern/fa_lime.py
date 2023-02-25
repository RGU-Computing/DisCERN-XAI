from discern.fa_base import FeatureAttribution
from sklearn.base import ClassifierMixin
from lime.lime_tabular import LimeTabularExplainer
import tensorflow as tf
import numpy as np

class FeatureAttributionLIME(FeatureAttribution):

    def __init__(self, model, feature_names, train_data, labels):
        super().__init__(model)
        self.feature_names = feature_names
        self.train_data = train_data
        self.labels = labels
        self.lime_explainer = LimeTabularExplainer(self.train_data,
                                                              feature_names=self.feature_names,
                                                              class_names=self.labels,
                                                              discretize_continuous=True)
    def explain_instance(self, query, query_label=None, nun=None):
        weights_map = self.lime_explainer.explain_instance(query,
                                                            self.model.predict_proba if isinstance(self.model, ClassifierMixin) else self.model.predict if isinstance(self.model, tf.keras.Model) else None,
                                                            num_features=len(self.feature_names),
                                                            top_labels=1).as_map()
        return weights_map[list(weights_map.keys())[0]]
