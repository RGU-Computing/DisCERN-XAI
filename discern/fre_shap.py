from discern.fre_base import FeatureRelevance
import shap
import pandas as pd

class FeatureRelevanceSHAP(FeatureRelevance):

    def __init__(self, model, feature_names, **kwargs):
        super().__init__(model, feature_names)
        self.shap_explainer = shap.TreeExplainer(self.model)

    def explain_instance(self, query, query_label, **kwargs):
        i_exp = pd.DataFrame([query], columns=self.feature_names)
        shap_values = self.shap_explainer.shap_values(i_exp)
        return [(i,w) for i,w in enumerate(shap_values[int(query_label)][0])]
        # pass