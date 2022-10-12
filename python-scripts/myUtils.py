import numpy as np


def jaccard_coefficient(cluster, truth):
    n = float(len(cluster))
    r = float(len(truth))
    pc = cluster_probability(cluster)
    pt = cluster_probability(truth)

    sc = list(set(cluster))  # name of the clusters
    st = list(set(truth))  # name of the clusters in ground truth

    # declare empty array to store p_ij
    p_matrix = [[0 for x in range(len(sc))] for y in range(len(st))]

    for i in range(len(truth)):
        p_matrix[st.index(truth[i])][sc.index(cluster[i])] += 1
    mutual_info = 0

    # calculate tp
    tp = 0
    for i in range(len(st)):
        for j in range(len(sc)):
            n_ij = (p_matrix[i][j])
            tp += (n_ij * (n_ij - 1)) / 2

    # calculate fn
    pt = cluster_probability(truth)
    temp = 0
    for i in range(len(pt)):
        mi = pt[i] * n
        temp += (mi * (mi - 1)) / 2
    fn = temp - tp

    # calculate fn
    pc = cluster_probability(cluster)
    total_positives = 0
    for i in range(len(pc)):
        mi = pc[i] * n
        total_positives += (mi * (mi - 1)) / 2
    fp = total_positives - tp

    return tp / (tp + fn + fp)

def cluster_probability(cluster):
	cluster_sum=[]
	n = float(len(cluster))
	cluster_names = list(set(cluster))
	for index,cluster_name in enumerate(cluster_names):
		cluster_sum.append(0)
		for i in cluster:
			if(cluster_name==i):
				cluster_sum[index]+=1
	cluster_prob = [i/n for i in cluster_sum]
	return cluster_prob

i=jaccard_coefficient([1,1,0,0],[1,1,0,0])
print(i)

