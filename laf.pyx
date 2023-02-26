import numpy as np
cimport numpy as np
import random
import math
from sklearn.neighbors import KDTree
from datetime import datetime
import time

cdef extern from "k_centers.h":
    void k_centers_cy(int m, int n, int d,
                      double * X,
                      double * closest_dist_sq,
                      int * result)


cdef k_centers_np(m, n, d,
                  np.ndarray[double,  ndim=2, mode="c"] X,
                  np.ndarray[double, ndim=1, mode="c"] closest_dist_sq,
                  np.ndarray[np.int32_t, ndim=1, mode="c"] result):
    k_centers_cy(m, n, d,
                 <double *> np.PyArray_DATA(X),
                 <double *> np.PyArray_DATA(closest_dist_sq),
                 <int *> np.PyArray_DATA(result))

cdef extern from "DBSCAN.h":
    void DBSCAN_cy(int c, int n,
                   int * X_core,
                   int * neighbors,
                   int * num_neighbors,
                   int * result)


cdef DBSCAN_np(c, n,
               np.ndarray[np.int32_t,  ndim=1, mode="c"] X_core,
               np.ndarray[np.int32_t, ndim=1, mode="c"] neighbors,
               np.ndarray[np.int32_t, ndim=1, mode="c"] num_neighbors,
               np.ndarray[np.int32_t, ndim=1, mode="c"] result):
    DBSCAN_cy(c, n,
              <int *> np.PyArray_DATA(X_core),
              <int *> np.PyArray_DATA(neighbors),
              <int *> np.PyArray_DATA(num_neighbors),
              <int *> np.PyArray_DATA(result))


cdef extern from "cluster_remaining.h":
    void cluster_remaining_cy(int n,
                              int * closest_point,
                              int * result)

cdef cluster_remaining_np(n, 
                          np.ndarray[int, ndim=1, mode="c"] closest_point,
                          np.ndarray[np.int32_t, ndim=1, mode="c"] result):
    cluster_remaining_cy(n,
                         <int *> np.PyArray_DATA(closest_point),
                         <int *> np.PyArray_DATA(result))


# for the standard DBSCAN impl
cdef extern from "standard_dbscan_impl.h":
    void StandardDBSCAN(int c, int n,
                   int * X_core,
                   int * neighbors,
                   int * num_neighbors,
                   int * core_or_not_indicators, 
                   int * result)

cdef standard_DBSCAN_impl(c, n,
               np.ndarray[np.int32_t,  ndim=1, mode="c"] X_core,
               np.ndarray[np.int32_t, ndim=1, mode="c"] neighbors,
               np.ndarray[np.int32_t, ndim=1, mode="c"] num_neighbors,
               np.ndarray[np.int32_t, ndim=1, mode="c"] core_or_not_indicators,
               np.ndarray[np.int32_t, ndim=1, mode="c"] result):
    StandardDBSCAN(c, n,
              <int *> np.PyArray_DATA(X_core),
              <int *> np.PyArray_DATA(neighbors),
              <int *> np.PyArray_DATA(num_neighbors),
              <int *> np.PyArray_DATA(core_or_not_indicators),
              <int *> np.PyArray_DATA(result))


class DBSCANPP:
    """
    Parameters
    ----------
    
    p: The sample fraction, which determines m, the number of points to sample

    eps_density: Radius for determining core points; points that have greater than
                 minPts in its eps_density-radii ball will be considered core points

    eps_clustering: Radius for determining neighbors and edges in the density graph

    minPts: Number of neighbors required for a point to be labeled a core point. Works
            in conjunction with eps_density

    """

    def __init__(self, p, eps_density,
                 eps_clustering, minPts):
        self.p = p
        self.eps_density = eps_density
        self.eps_clustering = eps_clustering
        self.minPts = minPts

    def k_centers(self, m, n, d, X):
      """
      Return m points from X that are far away from each other

      Parameters
      ----------
      m: Number of points to sample
      n: Size of original dataset
      d: Dimensions of original dataset
      X: (m, d) dataset

      Returns
      ----------
      (m, ) list of indices
      """

      n, d = X.shape

      indices = np.empty(m, dtype=np.int32)
      closest_dist_sq = np.empty(n, dtype=np.float64)

      k_centers_np(m, n, d,
                   X,
                   closest_dist_sq,
                   indices)

      return indices

    def fit_predict(self, X, init="k-centers", cluster_outliers=True):
        """
        Implementation for the original DBSCAN++.

        Determines the clusters in three steps.
        First step is to sample points from X using either the
        k-centers greedy sampling technique or a uniform
        sample technique. The next step is to run DBSCAN on the
        sampled points using the k-NN densities. Finally, all the 
        remaining points are clustered to the closest cluster.

        Parameters
        ----------
        X: Data matrix. Each row should represent a datapoint.
        init: String. Either "k-centers" for the K-center greedy
              sampling technique or "uniform" for a uniform random
              sampling technique
        cluster_outliers: Boolean. Whether we should cluster the 
              remaining points

        Returns
        ----------
        (n, ) cluster labels
        """

        X = np.ascontiguousarray(X)
        n, d = X.shape
        m = int(self.p * math.pow(n, d/(d + 4.0)))
        
        # Find a random subset of m points 
        if init == "uniform":
          X_sampled_ind = np.random.choice(np.arange(n, dtype=np.int32), m, replace=False)
          X_sampled_ind = np.sort(X_sampled_ind)
        elif init == "k-centers":
          X_sampled_ind = self.k_centers(m, n, d, X)
        else:
          raise ValueError("Initialization technique %s is not defined." % init)
        
        X_sampled_pts = X[X_sampled_ind]
        
        # Find the core points
        X_tree = KDTree(X)
        radii, indices = X_tree.query(X_sampled_pts, k=self.minPts)
        X_core_ind = X_sampled_ind[radii[:,-1] <= self.eps_density]
        X_core_pts = X[X_core_ind]
        
        # Get the list of core neighbors for each core point
        core_pts_tree = KDTree(X_core_pts)

        neighbors = core_pts_tree.query_radius(X_core_pts, self.eps_clustering)
        neighbors_ct = np.array([len(neighbors_for_one_point) for neighbors_for_one_point in neighbors])
        neighbors = np.asarray(np.concatenate(neighbors), dtype=np.int32)
        num_neighbors = np.cumsum(neighbors_ct, dtype=np.int32)
        
        # Cluster the core points
        c = X_core_ind.shape[0]
        result = np.full(n, -1, dtype=np.int32)
        DBSCAN_np(c,
                  n,
                  X_core_ind,
                  neighbors,
                  num_neighbors,
                  result)
      
        # Find the closest core point to every data point
        dist_to_core_pt, closest_core_pt = core_pts_tree.query(X, k=1)
        closest = X_core_ind[closest_core_pt[:,0]]

        # Cluster the remaining points
        cluster_remaining_np(n, closest, result)
        
        # Set outliers
        if not cluster_outliers:
          result[dist_to_core_pt[:,0] > self.eps_density] = -1

        return result

    def insertNonCorePointIfNotExists(self, pred_non_core_point, pred_non_core_point_neighbors):
      if pred_non_core_point not in pred_non_core_point_neighbors: 
          pred_non_core_point_neighbors[pred_non_core_point] = set()

    def updateNonCorePointIfNerghbored(self, center_point, neighors, pred_non_core_point_neighbors):
        for neighbor in neighors:
            if neighbor in pred_non_core_point_neighbors:
                # if a predicted non-core point is among the neighbors of the given center point,
                # then add the current center point into the predicted non-core point's neighbor set.
                pred_non_core_point_neighbors[neighbor].add(center_point)
            
    def fit_predict_with_card_est_with_postproc(self, X, pred_num_neighbors, pred_core_minPts=None, init="k-centers", cluster_outliers=True):
        """
        Implementation for LAF-DBSCAN++.

        Parameters
        ----------
        X: Data matrix. Each row should represent a datapoint. Shape (n_samples, n_features).
        pred_num_neighbors: The predictions made by the cardinality estimator for each point in X. Shape (n_samples, ).   
        pred_core_minPts: Threshold for determining core or not based on the predicted cardinalities. 
                          This threshold defaults to be minPts(i.e., tau),
                          but usually we use error_factor*minPts as the value for this threshold. 
        init: String. Either "k-centers" for the K-center greedy
              sampling technique or "uniform" for a uniform random
              sampling technique
        cluster_outliers: Boolean. Whether we should cluster the 
              remaining points. We fixed it as False in our evaluation for LAF.

        Returns
        ----------
        (n_samples, ) cluster labels for each point in X
        """
        if pred_core_minPts is None:
          pred_core_minPts = self.minPts
        
        pred_non_core_point_neighbors = {}

        X = np.ascontiguousarray(X)
        n, d = X.shape
        m = int(self.p * math.pow(n, d/(d + 4.0)))

        # Find a random subset of m points 
        if init == "uniform":
          X_sampled_ind = np.random.choice(np.arange(n, dtype=np.int32), m, replace=False)
          X_sampled_ind = np.sort(X_sampled_ind)
        elif init == "k-centers":
          X_sampled_ind = self.k_centers(m, n, d, X)
        else:
          raise ValueError("Initialization technique %s is not defined." % init)
        
        X_sampled_pts = X[X_sampled_ind]

        # predict the core points by learned cardinality estimator
        pred_num_neighbors_for_sampled_pts = pred_num_neighbors[X_sampled_ind]
        filter_pred_core_pts = pred_num_neighbors_for_sampled_pts >= pred_core_minPts 
        filter_pred_non_core_pts = pred_num_neighbors_for_sampled_pts < pred_core_minPts 
        X_pred_core_pts = X_sampled_pts[filter_pred_core_pts] # The points predicted to be core 
        X_pred_core_ind = X_sampled_ind[filter_pred_core_pts]
        X_pred_non_core_ind = X_sampled_ind[filter_pred_non_core_pts]
        for idx in X_pred_non_core_ind:
            self.insertNonCorePointIfNotExists(idx, pred_non_core_point_neighbors)
        
        # Double check for the predicted core points
        X_tree = KDTree(X)
        radii, indices = X_tree.query(X_pred_core_pts, k=self.minPts)
        X_core_ind = X_pred_core_ind[radii[:,-1] <= self.eps_density]
        X_core_pts = X[X_core_ind] # Real core points out of the predicted core points 
        
        # Get the list of core neighbors for each core point
        core_pts_tree = KDTree(X_core_pts)
        neighbors = core_pts_tree.query_radius(X_core_pts, self.eps_clustering)
        for i in range(len(X_core_pts)):
            self.updateNonCorePointIfNerghbored(X_core_pts[i], X_core_ind[neighbors[i]], pred_non_core_point_neighbors)

        neighbors_ct = np.array([len(neighbors_for_one_point) for neighbors_for_one_point in neighbors])
        neighbors = np.asarray(np.concatenate(neighbors), dtype=np.int32)
        num_neighbors = np.cumsum(neighbors_ct, dtype=np.int32)
        
        # Cluster the core points
        c = X_core_ind.shape[0]
        result = np.full(n, -1, dtype=np.int32)
        DBSCAN_np(c,
                  n,
                  X_core_ind,
                  neighbors,
                  num_neighbors,
                  result)

        # Find the closest core point to every data point
        dist_to_core_pt, closest_core_pt = core_pts_tree.query(X, k=1)
        closest = X_core_ind[closest_core_pt[:,0]]

        # Cluster the remaining points
        cluster_remaining_np(n, closest, result)
        
        # Set outliers
        if not cluster_outliers:
          result[dist_to_core_pt[:,0] > self.eps_density] = -1

        # Postprocessing
        num_false_negative = 0
        for pred_non_core_point in pred_non_core_point_neighbors: 
            if len(pred_non_core_point_neighbors[pred_non_core_point]) >= self.minPts:
                num_false_negative += 1
                # merge the clusters adjacent to this point, 
                #  as those clusters are wrongly separated clusters by this false non-core point.
                dest_cluster_id = -2
                for neighbor in pred_non_core_point_neighbors[pred_non_core_point]:
                    if result[neighbor] < 0:
                        # the current neighbor is noise, not belonging to any cluster, so skip it
                        continue
                    
                    if dest_cluster_id == -2:
                        # set the first (non-noise) cluster ID as the destination cluster ID of the merging, 
                        # i.e., all the other adjacent clusters will be modified to this ID.
                        dest_cluster_id = result[neighbor]
                        continue

                    # change IDs of the other adjacent clusters to ID of the first cluster (i.e., dest_cluster_id)
                    assert dest_cluster_id >= 0
                    neighbor_cluster = result[neighbor]
                    reassigned_indices = result == neighbor_cluster
                    result[reassigned_indices] = dest_cluster_id
                
        return result


class DBSCAN:
    """
    Standard/Naive DBSCAN implementation based on the DBSCANPP codebase above.

    Parameters
    ----------
    eps: Radius for determining core points; points that have greater than
                 minPts in its eps-radii ball will be considered core points
    minPts: Number of neighbors required for a point to be labeled a core point. Works
            in conjunction with eps
    """

    def __init__(self, eps, minPts):
        self.eps = eps
        self.minPts = minPts

    def fit_predict(self, X, cluster_outliers=True):
        """
        The standard DBSCAN implementation ("standard_DBSCAN_impl") is called here. 
        See "standard_dbscan_impl.h" for more details. 

        Parameters
        ----------
        X: Data matrix. Each row should represent a datapoint. Shape (n_samples, n_features).
        cluster_outliers: Boolean. Whether we should cluster the 
              remaining points. We fixed it as False in our evaluation for LAF.

        Returns
        ----------
        (n_samples, ) cluster labels for each point in X
        """

        X = np.ascontiguousarray(X)
        n, d = X.shape
        X_ind = np.arange(n, dtype=np.int32) 
        
        # Find the core points
        X_tree = KDTree(X)
        radii, indices = X_tree.query(X, k=self.minPts)
        X_core_ind = X_ind[radii[:,-1] <= self.eps]
        X_core_pts = X[X_core_ind]

        # prepare the core_or_not_indicators
        core_or_not_indicators = np.full(n, -1, dtype=np.int32)
        for local_idx in range(len(X_core_ind)):
          global_idx = X_core_ind[local_idx]
          core_or_not_indicators[global_idx] = local_idx
        
        # Get the list of core neighbors for each core point
        neighbors = X_tree.query_radius(X_core_pts, self.eps)
        neighbors_ct = np.array([len(neighbors_for_one_point) for neighbors_for_one_point in neighbors])
        neighbors = np.asarray(np.concatenate(neighbors), dtype=np.int32)
        num_neighbors = np.cumsum(neighbors_ct, dtype=np.int32)
        
        # Cluster by original DBSCAN
        c = X_core_ind.shape[0]
        result = np.full(n, -1, dtype=np.int32)
        standard_DBSCAN_impl(c,
                  n,
                  X_core_ind,
                  neighbors,
                  num_neighbors,
                  core_or_not_indicators, 
                  result)

        return result
    
    def insertNonCorePointIfNotExists(self, pred_non_core_point, pred_non_core_point_neighbors):
      if pred_non_core_point not in pred_non_core_point_neighbors: 
          pred_non_core_point_neighbors[pred_non_core_point] = set()

    def updateNonCorePointIfNerghbored(self, center_point, neighbors, pred_non_core_point_neighbors):
        for neighbor in neighbors:
            if neighbor in pred_non_core_point_neighbors:
                # if a predicted non-core point is among the neighbors of the given center point,
                # then add the current center point into the predicted non-core point's neighbor set.
                pred_non_core_point_neighbors[neighbor].add(center_point)
            
    def fit_predict_with_card_est_with_postproc(self, X, pred_num_neighbors, pred_core_minPts=None, cluster_outliers=True):
        """
        Implementation for LAF-DBSCAN.

        Parameters
        ----------
        X: Data matrix. Each row should represent a datapoint. Shape (n_samples, n_features).
        pred_num_neighbors: The predictions made by the cardinality estimator for each point in X. Shape (n_samples, ).   
        pred_core_minPts: Threshold for determining core or not based on the predicted cardinalities. 
                          This threshold defaults to be minPts(i.e., tau),
                          but usually we use error_factor*minPts as the value for this threshold. 
        cluster_outliers: Boolean. Whether we should cluster the 
              remaining points. We fixed it as False in our evaluation for LAF.

        Returns
        ----------
        (n_samples, ) cluster labels for each point in X, 
        #predicted-core-points, 
        #real-core-points out of the predicted-core-points
        """

        if pred_core_minPts is None:
          pred_core_minPts = self.minPts
        
        pred_non_core_point_neighbors = {}
        
        X = np.ascontiguousarray(X)
        n, d = X.shape
        X_ind = np.arange(n, dtype=np.int32) 

        # predict the core points by learned cardinality estimator
        pred_num_neighbors_for_sampled_pts = pred_num_neighbors
        filter_pred_core_pts = pred_num_neighbors_for_sampled_pts >= pred_core_minPts 
        filter_pred_non_core_pts = pred_num_neighbors_for_sampled_pts < pred_core_minPts 
        X_pred_core_pts = X[filter_pred_core_pts] # The points predicted to be core 
        X_pred_core_ind = X_ind[filter_pred_core_pts]
        X_pred_non_core_ind = X_ind[filter_pred_non_core_pts]
        for idx in X_pred_non_core_ind:
            self.insertNonCorePointIfNotExists(idx, pred_non_core_point_neighbors)
        
        # Double check for the predicted core points
        X_tree = KDTree(X)
        radii, indices = X_tree.query(X_pred_core_pts, k=self.minPts)
        X_core_ind = X_pred_core_ind[radii[:,-1] <= self.eps]
        X_core_pts = X[X_core_ind] # Real core points out of the predicted-core-points
        
        # prepare the core_or_not_indicators
        core_or_not_indicators = np.full(n, -1, dtype=np.int32)
        for local_idx in range(len(X_core_ind)):
          global_idx = X_core_ind[local_idx]
          core_or_not_indicators[global_idx] = local_idx
        
        # Get the list of core neighbors for each core point
        neighbors = X_tree.query_radius(X_core_pts, self.eps)

        for i in range(len(X_core_pts)):
            self.updateNonCorePointIfNerghbored(X_core_ind[i], neighbors[i], pred_non_core_point_neighbors)

        neighbors_ct = np.array([len(neighbors_for_one_point) for neighbors_for_one_point in neighbors])
        neighbors = np.asarray(np.concatenate(neighbors), dtype=np.int32)
        num_neighbors = np.cumsum(neighbors_ct, dtype=np.int32)
        
        # Cluster by original DBSCAN
        c = X_core_ind.shape[0]
        result = np.full(n, -1, dtype=np.int32)
        standard_DBSCAN_impl(c,
                  n,
                  X_core_ind,
                  neighbors,
                  num_neighbors,
                  core_or_not_indicators, 
                  result)
      
        # Postprocessing
        num_false_negative = 0
        for pred_non_core_point in  pred_non_core_point_neighbors: 
            if len(pred_non_core_point_neighbors[pred_non_core_point]) >= self.minPts:
                num_false_negative += 1
                # merge the clusters adjacent to this point, 
                #  as those clusters are wrongly separated clusters by this false non-core point.
                dest_cluster_id = -2
                for neighbor in pred_non_core_point_neighbors[pred_non_core_point]:
                    if result[neighbor] < 0:
                        # the current neighbor is noise, not belonging to any cluster, so skip it
                        continue
                    
                    if dest_cluster_id == -2:
                        # set the first (non-noise) cluster ID as the destination cluster ID of the merging, 
                        # i.e., all the other adjacent clusters will be modified to this ID.
                        dest_cluster_id = result[neighbor]
                        continue

                    # change IDs of the other adjacent clusters to ID of the first cluster (i.e., dest_cluster_id)
                    assert dest_cluster_id >= 0
                    neighbor_cluster = result[neighbor]
                    reassigned_indices = result == neighbor_cluster
                    result[reassigned_indices] = dest_cluster_id
        return result, len(X_pred_core_pts), len(X_core_pts)
    