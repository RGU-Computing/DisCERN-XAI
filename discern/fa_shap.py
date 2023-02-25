from discern.fa_base import FeatureAttribution
import shap
import pandas as pd

class FeatureAttributionSHAP(FeatureAttribution):

    def __init__(self, model, feature_names):
        super().__init__(model)
        self.feature_names = feature_names
        self.shap_explainer = shap.TreeExplainer(self.model)

    def explain_instance(self, query, query_label=None, nun=None):
        i_exp = pd.DataFrame([query], columns=self.feature_names)
        shap_values = self.shap_explainer.shap_values(i_exp)
        return [(i,w) for i,w in enumerate(shap_values[int(query_label)][0])]