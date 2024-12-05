import numpy as np
from scipy.spatial import cKDTree

# Generate a random point cloud
N = 1000  # Number of points
points = np.random.rand(N, 3)  # Random points in 3D space

# Build a KDTree for neighbor searches
tree = cKDTree(points)

# Parameters
k = 10  # Number of neighbors for normals and PFH computation
nbins = 5  # Number of bins for histograms

# Compute normals
normals = np.zeros((N, 3))

for i in range(N):
    # Find k nearest neighbors including the point itself
    distances, indices = tree.query(points[i], k=k)
    neighbors = points[indices]
    
    # Compute covariance matrix
    mean_neighbor = neighbors.mean(axis=0)
    cov = np.cov(neighbors - mean_neighbor, rowvar=False)
    
    # Compute eigenvectors and eigenvalues
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Normal vector corresponds to smallest eigenvalue
    normal = eigvecs[:, np.argmin(eigvals)]
    normals[i] = normal

# Initialize PFH descriptors
pfh_descriptors = np.zeros((N, nbins * 3))

# Compute PFH descriptors
for i in range(N):
    # Find k nearest neighbors excluding the point itself
    distances, indices = tree.query(points[i], k=k+1)
    indices = indices[1:]  # Exclude the point itself
    neighbors = points[indices]
    neighbor_normals = normals[indices]
    
    # Histograms for angular features
    hist_alpha = np.zeros(nbins)
    hist_phi = np.zeros(nbins)
    hist_theta = np.zeros(nbins)
    
    ni = normals[i]
    
    for j in range(len(indices)):
        pj = neighbors[j]
        nj = neighbor_normals[j]
        
        # Compute difference vector
        vij = pj - points[i]
        d = np.linalg.norm(vij)
        if d == 0:
            continue
        vij_norm = vij / d
        
        # Compute the Darboux frame at pi
        u = ni
        cross = np.cross(vij_norm, u)
        norm_cross = np.linalg.norm(cross)
        if norm_cross == 0:
            continue
        v = cross / norm_cross
        w = np.cross(u, v)
        
        # Compute angular features
        alpha = np.dot(v, nj)
        phi = np.dot(u, vij_norm)
        theta = np.arctan2(np.dot(w, nj), np.dot(u, nj))
        
        # Map the features to bins
        # Alpha and phi range from -1 to 1
        # Theta ranges from -π to π
        # Bin alpha
        alpha_bin = int(((alpha + 1) / 2) * nbins)
        if alpha_bin >= nbins:
            alpha_bin = nbins - 1
        hist_alpha[alpha_bin] += 1
        
        # Bin phi
        phi_bin = int(((phi + 1) / 2) * nbins)
        if phi_bin >= nbins:
            phi_bin = nbins - 1
        hist_phi[phi_bin] += 1
        
        # Bin theta
        theta_normalized = (theta + np.pi) / (2 * np.pi)  # Normalize to [0, 1]
        theta_bin = int(theta_normalized * nbins)
        if theta_bin >= nbins:
            theta_bin = nbins - 1
        hist_theta[theta_bin] += 1
    
    # Normalize histograms
    if hist_alpha.sum() > 0:
        hist_alpha /= hist_alpha.sum()
    if hist_phi.sum() > 0:
        hist_phi /= hist_phi.sum()
    if hist_theta.sum() > 0:
        hist_theta /= hist_theta.sum()
    
    # Concatenate histograms to form the PFH descriptor
    pfh_descriptor = np.concatenate([hist_alpha, hist_phi, hist_theta])
    pfh_descriptors[i] = pfh_descriptor

# PFH descriptors computed for each point
# For demonstration, print the first PFH descriptor
print("PFH descriptor for point 0:")
print(pfh_descriptors[0])
