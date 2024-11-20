#!/usr/bin/env python
import utils
import numpy
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree

class PFH(object):
    def __init__(self, pointcloud, radius, num_neighbors):
        self.pc = np.asarray(pointcloud).T # (N, 3)
        self.r = radius
        self.nneighbors = num_neighbors
        self.tree = None

    def pca_normal(self, X):
        """
        X - (N, 3)
        """
        n = X.shape[0]
        mu = np.mean(X, axis=0)
        X = X - mu
        Q = (X.T @ X) / (n - 1)
        U, sigma, V_T = np.linalg.svd(Q)
        normal = V_T.T[:, -1]
        return normal

    def get_kNNs(self, point):
        """
        Get k-neighborhood defined by a sphere centered at point with self.radius in self.pc

        PARAMS:
        point - the point of interest

        RETURN:
        ndarray of k points
        """
        if self.tree is None:
            self.tree = KDTree(self.pc)
        
        idx_within_radius = self.tree.query_ball_point(point, r=self.r)
        if idx_within_radius:
            points_within_radius = self.pc[idx_within_radius[0]]
            tree_within_radius = KDTree(points_within_radius)
            distances, k_indices = tree_within_radius.query(point, k=min(self.nneighbors, len(points_within_radius)))
            neighborhood = np.asarray(points_within_radius)[k_indices]
            # print(f"{self.nneighbors}-neighborhood within radius {self.r}: {neighborhood}")
            return neighborhood.squeeze()
        else:
            print("No points within radius")
            return None
        
    def get_normals(self):
        """
        Get normals for all points in self.pc. Re-orient vectors outward (away from mean).

        RETURN:
        ndarray of normals
        """
        normals = []
        N = self.pc.shape[0]
        mean = np.mean(self.pc, axis=0)
        for i in range(N):
            neighbors = self.get_kNNs(self.pc[i].reshape(1,3))
            normal = self.pca_normal(neighbors)
            v = mean - self.pc[i]
            v /= np.linalg.norm(v)
            if np.dot(normal, v) > 0:
                normal *= -1
            normals.append(normal)
        return np.asarray(normals)
