from discern.fa_base import FeatureAttribution
import shap
import pandas as pd
from sklearn.base import ClassifierMixin
import tensorflow as tf
import numpy as np

class FeatureAttributionSHAP(FeatureAttribution):

    def __init__(self, model, feature_names, train_data):
        super().__init__(model)
        self.feature_names = feature_names
        self.train_data = train_data
        if isinstance(self.model, ClassifierMixin):
            self.shap_explainer = shap.TreeExplainer(self.model)
        elif isinstance(self.model, tf.keras.Model):
            self.shap_explainer = shap.DeepExplainer(self.model, self.train_data)
            

    def explain_instance(self, query, query_label=None, nun=None):
        if isinstance(self.model, ClassifierMixin):
            i_exp = pd.DataFrame([query], columns=self.feature_names)
            shap_values = self.shap_explainer.shap_values(i_exp)
            return [(i,w) for i,w in enumerate(shap_values[int(query_label)][0])]
        elif isinstance(self.model, tf.keras.Model):
            shap_values = self.shap_explainer.shap_values(np.array([query]))
            print(shap_values)
            return [(i,w) for i,w in enumerate(shap_values[int(query_label)][0])]
        
    
