from abc import ABC, abstractmethod

class DisCERN(ABC):
    """
    DisCERN class for tabular data and sklearn classifier
    """

    def __init__(self, model, attrib, threshold=0.0):
        """
        Init method

        :param model: a trained ML model; currently supports sklearn backend
        :param attrib: preferred Fature Attribution Explainer; currently supports LIME
        :param threshold: threshold to consider two feature values are different; default is 0.0

        """
        self.model = model
        self.attrib = attrib
        self.threshold = threshold

    def init_data(self, train_data, train_labels, feature_names, labels, **kwargs):
        """
        Init Data method

        :param train_data: train dataset as a numpy array; shape=(num_instances, num_features)
        :param train_labels: list of train dataset labels; shape=(num_instances, )
        :param feature_names: list of feature names; shape=(num_features, )
        :param labels: list of labels; shape=(num_classes, )

        """
        self.train_data = train_data
        self.train_labels = train_labels
        self.feature_names = feature_names
        self.labels = labels

        if len(self.train_data) == 0 or len(self.train_labels) == 0:
            raise ValueError("DisCERN requires a train dataset.")
        if len(self.feature_names) == 0:
            raise ValueError("DisCERN requires feature names.")
        if len(self.labels) == 0:
            raise ValueError("DisCERN requires class names.")
        if len(self.labels) != len(set(self.train_labels)):
            raise ValueError("Mismatch between class names and number of classes.")
        if len(self.feature_names) != self.train_data.shape[1]:
            raise ValueError("Mismatch between number of features and training data.")

        self._init_data(**kwargs)

    @abstractmethod
    def _init_data(self, **kwargs):
        pass

    @abstractmethod
    def find_cf(self, test_instance, test_label, cf_label='opposite', **kwargs):
        pass

    @abstractmethod
    def show_cf(self, test_instance, test_label, cf, cf_label, **kwargs):
        pass
    