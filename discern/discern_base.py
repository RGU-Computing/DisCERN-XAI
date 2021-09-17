from abc import ABC, abstractmethod

class DisCERN(ABC):
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
        self.model = model
        self.rel_ex = rel
        self.pivot = p
        self.threshold = threshold

    def init_data(self, train_data, train_labels, feature_names, class_names, **kwargs):
        """
        Init Data method

        :param train_data: train dataset as a numpy array; shape=(num_instances, num_features)
        :param train_labels: list of train dataset labels; shape=(num_instances, )
        :param feature_names: list of feature names; shape=(num_features, )
        :param class_names: list of class_names; shape=(num_classes, )

        """
        self.train_data = train_data
        self.train_labels = train_labels
        self.feature_names = feature_names
        self.class_names = class_names

        if len(self.train_data) == 0 or len(self.train_labels) == 0:
            raise ValueError("DisCERN requires train dataset!")
        if len(self.feature_names) == 0:
            raise ValueError("DisCERN requires feature names!")
        if len(self.class_names) == 0:
            raise ValueError("DisCERN requires class names!")
        if len(self.class_names) != len(set(self.train_labels)):
            raise ValueError("Mismatch between class names and number of classes!")
        if len(self.feature_names) != self.train_data.shape[1]:
            raise ValueError("Mismatch between feature names and training data shape!")

        self._init_data(train_data, train_labels, feature_names, class_names, **kwargs)

    @abstractmethod
    def _init_data(self, train_data, train_labels, feature_names, class_names, **kwargs):
        """
        Internal Init Data method

        :param train_data: train dataset as a numpy array; shape=(num_instances, num_features)
        :param train_labels: list of train dataset labels; shape=(num_instances, )
        :param feature_names: list of feature names; shape=(num_features, )
        :param class_names: list of class_names; shape=(num_classes, )

        """
        pass

    @abstractmethod
    def find_cf(self, test_instance, test_label, **kwargs):
        pass

    @abstractmethod
    def show_cf(self, test_instance, test_label, cf, cf_label, **kwargs):
        pass
    