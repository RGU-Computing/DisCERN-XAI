from discern.fa_base import FeatureAttribution
import numpy as np
import tensorflow as tf

class FeatureAttributionIntG(FeatureAttribution):

    def __init__(self, model):
        super().__init__(model)

    def interpolate_queries(self, baseline, query, alphas):
        alphas_x = alphas[:, tf.newaxis]
        baseline_x = tf.expand_dims(baseline, axis=0)
        input_x = tf.expand_dims(query, axis=0)
        delta = input_x - baseline_x
        interpolated_querys = baseline_x +  alphas_x * delta
        return interpolated_querys
    
    def compute_gradients(self, queries):
        with tf.GradientTape() as tape:
            tape.watch(queries)
            probs = self.model(queries)[:]
        return tape.gradient(probs, queries) 

    def integral_approximation(self, gradients):
        grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0)
        integrated_gradients = tf.math.reduce_mean(grads, axis=0)
        return integrated_gradients  

    def explain_instance(self, query, query_label=None, nun=None):
        baseline = np.array(nun, dtype='float32')
        alphas = tf.linspace(start=0.0, stop=1.0, num=101)
        query = np.array(query, dtype='float32')
        interpolated_queries = self.interpolate_queries(baseline, query, alphas)
        path_gradients = self.compute_gradients(interpolated_queries)
        ig = self.integral_approximation(gradients=path_gradients)
        integrated_gradients = (query - baseline) * ig

        ig_normalised = (integrated_gradients-min(integrated_gradients))/(max(integrated_gradients)-min(integrated_gradients))
        _weights = zip(range(len(ig_normalised)), ig_normalised)
        return _weights