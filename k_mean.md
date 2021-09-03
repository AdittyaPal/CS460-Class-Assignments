# How do we determine a good 'k', the number of clusters in k-Mean?

In k-mean, the distane is calculated from each point to the centriods(randon guess for first iteration), and the point is assigned to the cluster containig the closest centroid. The averade feauture vector for each cluster becomes the new centroid and it is iterated.
The number of clusters k has to be minimized(so the clustering is fairly generalised), keeping the members of the cluster similar(variance within the cluster should be small too). As the number of clusters increases, variance decreases, When k=n, var=0. 

The variance (sum of squared distances from the centriod) can be plotted against the number of clusters, and a k, where the variance does not decrease rapidly should give a near optimal, where

S`var=\sum_{x\in cluster i} (x-c_i)^2`$
where i is the cluster id.
