from sklearn.metrics.pairwise import euclidean_distances
import heapq


def nun(data, labels, query, query_label, sample_size):
    ecd = euclidean_distances([query], data)[0]
    top_indices = heapq.nsmallest(sample_size, range(len(ecd)), ecd.take)
    top_labels = [labels[j] for j in top_indices[:sample_size]]
    for i, lab in enumerate(top_labels):
        if query_label != lab:
            nun_index = i
    return data[top_indices[nun_index]], top_labels[nun_index]

