#!/usr/bin/env python
import utils
import numpy
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import KDTree
from scipy.special import comb

class PFH(object):
    def __init__(self, pointcloud, radius, num_neighbors=8, div=2, num_features=4):
        self.pc = pointcloud # (N, 3)
        self.r = radius
        self.nneighbors = num_neighbors
        self.tree = None
        self.normals = self.get_normals()
        self.div = div
        self.nfeatures = num_features
    
    def get_point(self, idx):
        return self.pc[idx]
    
    def get_size(self):
        return self.pc.shape[0]

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

    def get_kNNs(self, idx):
        """
        Get k-neighborhood defined by a sphere centered at point with self.radius in self.pc

        PARAMS:
        idx - point index in self.pc

        RETURN:
        ndarray of k points
        """
        if self.tree is None:
            self.tree = KDTree(self.pc)

        point = self.pc[idx]
        idx_within_radius = self.tree.query_ball_point(point, r=self.r)
        # Exclude the point itself
        idx_within_radius = [i for i in idx_within_radius if i != idx]
    
        if idx_within_radius is not None:
            if len(idx_within_radius) > self.nneighbors:
                points_within_radius = self.pc[idx_within_radius]
                tree_within_radius = KDTree(points_within_radius)
                distances, k_indices = tree_within_radius.query(point, k=self.nneighbors)
                neighbor_indices = np.asarray(idx_within_radius)[k_indices]
            else:
                neighbor_indices = np.asarray(idx_within_radius)
            # print(f"{idx}: {self.nneighbors}-neighborhood within radius {self.r}: {neighbor_indices}")
            return np.asarray(neighbor_indices)
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
            neighbor_indices = self.get_kNNs(i)
            neighbors = self.pc[neighbor_indices]
            normal = self.pca_normal(neighbors)
            v = mean - self.pc[i]
            v /= np.linalg.norm(v)
            if np.dot(normal, v) > 0:
                normal *= -1
            normals.append(normal)
        return np.asarray(normals)
    
    def get_features(self, idx):
        """
        idx - point index in self.pc

        RETURN:
        ndarray of size (self.nneighbors, self.nfeatures).
        """
        neighbor_indices = self.get_kNNs(idx)
        n_combinations = comb((len(neighbor_indices) + 1), 2)
        combined_list = [idx] + list(neighbor_indices)
        points_idx = np.array(combined_list, dtype=int)
        points_idx_copy = points_idx.copy()
        features = []
        for i_idx in points_idx:
            p_i = self.pc[i_idx]
            n_i = self.normals[i_idx]
            points_idx_copy = points_idx_copy[1:]
            for j_idx in points_idx_copy:
                p_j = self.pc[j_idx]
                n_j = self.normals[j_idx]
                if np.arccos(np.dot(n_i, p_j - p_i)) <= np.arccos(np.dot(n_j, p_i - p_j)):
                    source_idx = i_idx
                    source_point = p_i
                    target_idx = j_idx
                    target_point = p_j
                else:
                    source_idx = j_idx
                    source_point = p_j
                    target_idx = i_idx
                    target_point = p_i
                d = np.linalg.norm(target_point - source_point)
                # Construct the Darboux frame
                u = self.normals[source_idx]
                v = np.cross((target_point - source_point), u)
                w = np.cross(u, v)
                # Construct the four features (including the distance)
                nt = self.normals[target_idx]
                f1 = np.dot(v, nt)
                f2 = d
                f3 = np.dot(u, (target_point - source_point) / d)
                f4 = np.arctan(np.dot(w, nt) / np.dot(u, nt))
                if self.nfeatures == 4:
                    features.append(np.array([f1, f2, f3, f4]))
                elif self.nfeatures == 3:
                    features.append(np.array([f1, f3, f4]))
        features = np.asarray(features)
        return features

    def get_feature_histogram(self, idx):
        """
        Implemented 2 div, inclduing d in the feature.
        Try 3 div without distance.
        """
        features = self.get_features(idx)
        r = 0
        if self.nfeatures == 4:
            r = np.median(features[:, 1])
        histogram = np.zeros(self.div ** self.nfeatures)
        s = self.get_threshold(r)
        for j in range(0, features.shape[0]):
            index = 0
            for i in range(0, self.nfeatures):
                index += self.step(s[i], features[j, i]) * (self.div ** (i))
            histogram[index] += 1
        histogram /= features.shape[0]
        return histogram

    def get_threshold(self, r=0):
        delta = 2. / self.div
        s1 = np.array([-1 + i * delta for i in range(1, self.div)])
        s3 = np.array([-1 + i * delta for i in range(1, self.div)])

        delta = np.pi / self.div
        s4 = np.array([-np.pi / 2 + i * delta for i in range(1, self.div)])

        if self.nfeatures == 3:
            s = np.array([s1, s3, s4])
        elif self.nfeatures == 4:
            s2 = np.array([r])
            s = np.array([s1, s2, s3, s4])
        return s
    
    def step(self, si, fi):
        if self.div == 2:
            result = 0 if fi < si[0] else 1
        elif self.div == 3:
            if fi < si[0]:
                result = 0
            elif fi >= si[0] and fi < si[1]:
                result = 1
            else:
                result = 2
        return result
    
    def get_all_histograms(self):
        N = self.pc.shape[0]
        histograms = np.zeros((N, self.div ** self.nfeatures))
        for i in range(N):
            histograms[i] = self.get_feature_histogram(i)
        return histograms
    
    def transform(self, R, t):
        self.pc = (R @ self.pc.T + t).T
        self.tree = None
        self.normals = self.get_normals()
        return self.pc

def get_correspondence(pfh_source, pfh_target):
    C = []
    histogram_source = pfh_source.get_all_histograms()
    histogram_target = pfh_target.get_all_histograms()
    for i in range(pfh_source.get_size()):
        p = pfh_source.get_point(i)
        histogram_p= histogram_source[i]
        epsilon = 1e-10
        numerator = (histogram_target - histogram_p) ** 2
        denominator = histogram_target + histogram_p + epsilon
        chi_squared_distances = np.sum(numerator / denominator, axis=1)
        min_index = np.argmin(chi_squared_distances)
        q = pfh_target.get_point(min_index)
        C.append([p, q])
    C = np.asarray(C) # (N, 2, 3)
    return C

def get_transform(C):
    # Step 0
    Cp = C[:, 0, :] # (N, 3)
    Cq = C[:, 1, :] # (N, 3)
    # Step 1
    p_bar = np.mean(Cp, axis=0)
    q_bar = np.mean(Cq, axis=0)
    X = Cp - p_bar
    Y = Cq - q_bar
    # Step 2
    S = X.T @ Y
    U, sigma, V_T = np.linalg.svd(S)
    # Step 3
    M = np.eye(3)
    M[2, 2] = np.linalg.det(V_T.T @ U.T)
    R = V_T.T @ M @ U.T
    t = q_bar.reshape(3, 1) - R @ p_bar.reshape(3, 1)
    return R, t

def get_error(C, R, t):
    Cp = C[:, 0, :] # (N, 3)
    Cq = C[:, 1, :] # (N, 3)
    errors = (((R @ Cp.T).T + t.reshape(1, 3)) - Cq) ** 2
    return np.sum(errors)
