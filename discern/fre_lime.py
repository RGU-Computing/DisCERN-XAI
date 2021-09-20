from discern.fre_base import FeatureRelevance
from lime.lime_tabular import LimeTabularExplainer

class FeatureRelevanceLIME(FeatureRelevance):

    def __init__(self, model, feature_names, train_data, class_names, **kwargs):
        super().__init__(model, feature_names)
        self.train_data = train_data
        self.class_names = class_names
        self.lime_explainer = LimeTabularExplainer(self.train_data,
                                                              feature_names=self.feature_names,
                                                              class_names=self.class_names,
                                                              discretize_continuous=True)
    def explain_instance(self, query, query_label, **kwargs):
        weights_map = self.lime_explainer.explain_instance(query,
                                                            self.model.predict_proba,
                                                            num_features=len(self.feature_names),
                                                            top_labels=1).as_map()
        return weights_map[list(weights_map.keys())[0]]
