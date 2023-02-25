from discern.fa_lime import FeatureAttributionLIME
from discern.fa_shap import FeatureAttributionSHAP
from discern.fa_intg import FeatureAttributionIntG
from sklearn.base import ClassifierMixin
import copy
from discern import util
from discern.discern_base import DisCERN
import numpy as np
import tensorflow as tf
class DisCERNTabular(DisCERN):
    """
    DisCERN class for tabular data and sklearn/keras classifier
    """

    def __init__(self, model, attrib, threshold=0.0):
        """
        Init method

        :param model: a trained ML model; currently supports sklearn backend
        :param attrib: preferred Fature Attribution Explainer; currently supports LIME
        :param threshold: threshold to consider two feature values are different; default is 0.0

        """
        super().__init__(model, attrib, threshold)

    def _init_data(self, cat_feature_indices, immutable_feature_indices):
        """
        Init Data method
        
        :param cat_feature_indices: list of indices where feature is categorical; shape=(num_cat_features, )
        :param immutable_feature_indices: list of indices where feature is immutable; e.g. race, sex; shape=(num_immutable_features, )

        """
        self.cat_feature_indices = cat_feature_indices
        self.immutable_feature_indices = immutable_feature_indices

        self.init_rel()

    def init_rel(self):
        """
        Init Feature Attribution Explainer Method

        """
        if self.attrib == 'LIME':
           self.feature_attrib = FeatureAttributionLIME(self.model, self.feature_names, train_data=self.train_data, labels=self.labels)
        elif self.attrib == 'SHAP':
            self.feature_attrib = FeatureAttributionSHAP(self.model, self.feature_names)
        elif self.attrib == 'IntG':
            self.feature_attrib = FeatureAttributionIntG(self.model)
        else:
            raise ValueError("Invalid Attribution Explainer!")

    def find_cf(self, test_instance, test_label, cf_label='opposite'):
        """
        Find Counterfactual method

        :param test_instance: query as a list of feature values
        :param test_label: class label predicted by the blackbox for the query
        :param cf_label: class label of the counterfactual; opposite or class label

        :returns: a counterfactual data instance as a list of feature values
        :returns: the number of feature changes i.e. sparsity
        :returns: the amount of feature change i.e. proximity
        """
        norm_test_instance = np.array(test_instance)
        sparsity = 0.0
        proximity = 0.0
        cf_label = 0 if int(test_label) == 1 else 1 if cf_label == 'opposite' else cf_label
        nun_data, nun_label = util.nun(self.train_data, self.train_labels, norm_test_instance, test_label, cf_label)

        _weights = self.feature_attrib.explain_instance(norm_test_instance, query_label=test_label, nun=nun_data)
        
        _weights_sorted = sorted(_weights, key=lambda tup: -tup[1])
        indices = [i for i,w in _weights_sorted]
        print(', '.join([str(s) for s in nun_data]))
        print(', '.join([str(s) for s in norm_test_instance]))
        x_adapted = copy.copy(norm_test_instance)
        now_index = 0
        # print("test_class: "+str(test_label))
        while True:
            val_x = x_adapted[indices[now_index]]
            val_nun = nun_data[indices[now_index]]

            if indices[now_index] in self.immutable_feature_indices:
                None
            elif indices[now_index] in self.cat_feature_indices:
                if val_x == val_nun: 
                    None
                else:
                    x_adapted[indices[now_index]] = nun_data[indices[now_index]]
                    sparsity +=1
                    proximity += 1
            else:
                if abs(val_x - val_nun) <= self.threshold:
                    None
                else:
                    x_adapted[indices[now_index]] = nun_data[indices[now_index]]
                    sparsity +=1
                    proximity += abs(val_x - val_nun)
            if isinstance(self.model, ClassifierMixin):
                new_label = self.model.predict([x_adapted])[0]
            elif isinstance(self.model, tf.keras.Model):
                new_label = self.model.predict(np.array([x_adapted])).argmax(axis=-1)[0]
            print('new_label: ', new_label, 'nun_label: ', nun_label, 'test_label: ', test_label)
            now_index += 1
            if new_label != test_label and new_label == nun_label:
                proximity = proximity/sparsity
                return x_adapted, new_label, sparsity, proximity
            if now_index >= len(self.feature_names):
                raise Exception('Counterfactual not found.')

    def show_cf(self, test_instance, test_label, cf, cf_label, **kwargs):
        None