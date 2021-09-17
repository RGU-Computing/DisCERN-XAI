from sklearn.metrics.pairwise import euclidean_distances
import heapq


def nun(data, labels, query, query_label, desired_class):
    sample_size = len(data)
    ecd = euclidean_distances([query], data)[0]
    top_indices = heapq.nsmallest(sample_size, range(len(ecd)), ecd.take)
    top_labels = [labels[j] for j in top_indices[:sample_size]]
    for i, lab in enumerate(top_labels):
        if desired_class == 'opposite' and query_label != lab:
            nun_index = i
            break
        elif desired_class != 'opposite' and lab == desired_class:
            nun_index = i
            break
    return data[top_indices[nun_index]], top_labels[nun_index]

