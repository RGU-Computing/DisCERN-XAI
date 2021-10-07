from discern.fre_lime import FeatureRelevanceLIME
from discern.fre_shap import FeatureRelevanceSHAP
import copy
from discern import util
from discern.discern_base import DisCERN
from sklearn.preprocessing import MinMaxScaler
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

class DisCERNTabular(DisCERN):
    """
    DisCERN class for tabular data and sklearn classifier
    """

    def __init__(self, model, rel, p, threshold=0.0):
        """
        Init method

        :param model: a trained ML model; currently supports sklearn backend
        :param rel: preferred Fature Relevance Explainer; currently supports LIME
        :param p: preferred pivot; either Q for Query or N for NUN; default is Query
        :param threshold: threshold to consider two feature values are different; default is 0.0

        """
        super().__init__(model, rel, p, threshold)

    def _init_data(self, train_data, train_labels, feature_names, class_names, cat_feature_indices, **kwargs):
        """
        Init Data method

        :param train_data: train dataset as a numpy array; shape=(num_instances, num_features)
        :param train_labels: list of train dataset labels; shape=(num_instances, )
        :param feature_names: list of feature names; shape=(num_features, )
        :param class_names: list of class_names; shape=(num_classes, )
        :param cat_feature_indices: list of indices where feature is categorical; shape=(num_cat_features, )

        """
        self.cat_feature_indices = cat_feature_indices
        self.scalar = MinMaxScaler()
        self.norm_train_data= self.scalar.fit_transform(self.train_data)

        # if all validations pass, we can proceed to initialising the feature relevance explainer
        self.init_rel()

    def init_rel(self):
        """
        Init Feature Relevance Explainer Method

        """
        if self.rel_ex == 'LIME':
           self.feature_rel = FeatureRelevanceLIME(self.model, self.feature_names, train_data=self.train_data, class_names=self.class_names)
        elif self.rel_ex == 'SHAP':
            self.feature_rel = FeatureRelevanceSHAP(self.model, self.feature_names)
        else:
            raise ValueError("Invalid Relevance Explainer!")

    def find_cf(self, test_instance, test_label, desired_class='opposite', **kwargs):
        """
        Find Counterfactual method

        :param test_instance: query as a list of feature values
        :param test_label: class label predicted by the blackbox for the query
        :param desired_class: class label of the counterfactual; opposite or class label

        :returns: a counterfactual data instance as a list of feature values
        :returns: the number of feature changes i.e. sparsity
        :returns: the amount of feature change i.e. proximity
        """
        norm_test_instance = self.scalar.transform([test_instance])[0]

        sparsity = 0.0
        proximity = 0.0

        nun_data, nun_label = util.nun(self.norm_train_data, self.train_labels, norm_test_instance, test_label, desired_class if desired_class == 'opposite' else self.class_names.index(desired_class))

        if self.pivot == 'Q':
            _weights = self.feature_rel.explain_instance(norm_test_instance, test_label)
        elif self.pivot == 'N':
            _weights = self.feature_rel.explain_instance(nun_data, nun_label)
        else:
            raise ValueError("Invalid Pivot! Please use Q for Query or N for NUN.")
        
        _weights_sorted = sorted(_weights, key=lambda tup: -tup[1])
        indices = [i for i,w in _weights_sorted]

        x_adapted = copy.copy(norm_test_instance)
        now_index = 0
        changes = 0
        amounts = 0
        # print("test_class: "+str(test_label))
        while True:
            val_x = x_adapted[indices[now_index]]
            val_nun = nun_data[indices[now_index]]
            # val_col = self.feature_names[indices[now_index]] # to use for causality

            if indices[now_index] in self.cat_feature_indices:
                if val_x == val_nun: 
                    None
                else:
                    x_adapted[indices[now_index]] = nun_data[indices[now_index]]
                    changes +=1
                    amounts += 1
            else:
                if abs(val_x - val_nun) <= self.threshold:
                    None
                else:
                    if self.feature_names[indices[now_index]]=="age" and nun_data[indices[now_index]] > x_adapted[indices[now_index]]:
                        None
                    elif self.feature_names[indices[now_index]] in [' Black', ' Other', ' White']:
                        None
                    elif self.feature_names[indices[now_index]]==' Self-emp-inc' and nun_data[indices[now_index]]==" Local-gov":
                        None
                    elif self.feature_names[indices[now_index]]==' Self-emp-inc' and nun_data[indices[now_index]]==" State-gov":
                        None
                    else:
                        x_adapted[indices[now_index]] = nun_data[indices[now_index]]
                        changes +=1
                        amounts += abs(val_x - val_nun)

            new_class = self.model.predict([x_adapted])[0]
            # print('new_class: '+str(new_class))
            now_index += 1
            if desired_class == 'opposite' and new_class != test_label:
                    sparsity = changes
                    proximity = amounts
                    break
            elif desired_class != 'opposite' and self.class_names[new_class] == desired_class:
                    sparsity = changes
                    proximity = amounts
                    break
            if now_index >= len(self.feature_names):
                break
        return x_adapted, sparsity, proximity

    def show_cf(self, test_instance, cf, **kwargs):

        PATH = "../discern/NLG_model/model.pt"
        if torch.cuda.is_available():
            dev = torch.device("cuda:0") 
        else:
            dev = torch.device("cpu")

        tokenizer = T5Tokenizer.from_pretrained('t5-base')
        model = T5ForConditionalGeneration.from_pretrained('t5-base', return_dict=True)
        model.eval()
        model.to(dev)
        model.load_state_dict(torch.load(PATH))

        l_test = []
        for i in test_instance:
            l_test.append(str(i))
            l_test.append(str(test_instance[i]))
        test_instance = '|'.join(l_test)
        input_ids = tokenizer.encode(test_instance, return_tensors="pt")
        input_ids=input_ids.to(dev)
        outputs = model.generate(input_ids,
            do_sample=True, 
            max_length=50, 
            top_k=50, 
            top_p=0.95)
        out_test = tokenizer.decode(outputs[0])

        l_cf = []
        for i in cf:
            l_cf.append(str(i))
            l_cf.append(str(cf[i]))
        cf = '|'.join(l_cf)
        input_ids = tokenizer.encode(cf, return_tensors="pt")
        input_ids=input_ids.to(dev)
        outputs = model.generate(input_ids,
            do_sample=True, 
            max_length=50, 
            top_k=50, 
            top_p=0.95)
        out_cf = tokenizer.decode(outputs[0])

        print(f"Instance: {out_test}")
        print(f"Counterfactual: {out_cf}")
        