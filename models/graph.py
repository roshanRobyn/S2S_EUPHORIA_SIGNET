import numpy as np

# --- 1. Graph Configuration ---
# We map the 109 nodes to their connections.
# Ranges based on your preprocessing:
# 0-32: Pose (Body)
# 33-53: Left Hand
# 54-74: Right Hand
# 75-108: Face

class Graph:
    def __init__(self, strategy='spatial'):
        self.num_node = 109
        self.edges = self.get_edges()
        self.center = 0 # Nose is the center
        
        # This creates the Adjacency Matrix (A)
        self.A = self.get_adjacency_matrix(strategy)

    def get_edges(self):
        edges = []
        
        # --- A. POSE CONNECTIONS (Standard MediaPipe) ---
        # (Simplified for brevity, connecting main joints)
        pose_connections = [
            (0,1), (1,2), (2,3), (3,7), (0,4), (4,5), (5,6), (6,8), # Face/Head
            (9,10), (11,12), (11,13), (13,15), (15,17), (12,14), (14,16), (16,18), # Arms
            (11,23), (12,24), (23,24), (23,25), (25,27), (24,26), (26,28) # Torso/Legs
        ]
        edges.extend(pose_connections)

        # --- B. HAND CONNECTIONS (21 points each) ---
        # Offset 33 for Left Hand, 54 for Right Hand
        for offset in [33, 54]:
            # Wrist to fingers
            hand_connections = [
                (0,1), (1,2), (2,3), (3,4),       # Thumb
                (0,5), (5,6), (6,7), (7,8),       # Index
                (0,9), (9,10), (10,11), (11,12),  # Middle
                (0,13), (13,14), (14,15), (15,16),# Ring
                (0,17), (17,18), (18,19), (19,20) # Pinky
            ]
            # Add offset to indices
            edges.extend([(i+offset, j+offset) for (i,j) in hand_connections])

        # --- C. FACE CONNECTIONS ---
        # Connect face points to the Nose (Index 0) to anchor them
        # Face starts at index 75
        face_start = 75
        num_face_points = 34 
        for i in range(num_face_points):
            edges.append((0, face_start + i)) # Connect Nose -> Face Point

        # --- D. BODY TO HANDS ---
        # Connect Left Wrist (Pose 15) to Left Hand Start (33)
        edges.append((15, 33))
        # Connect Right Wrist (Pose 16) to Right Hand Start (54)
        edges.append((16, 54))

        return edges

    def get_adjacency_matrix(self, strategy):
        self.valid_hop = 1
        self.dilation = 1
        self.max_hop = 1
        
        A = np.zeros((self.num_node, self.num_node))
        for i, j in self.edges:
            A[j, i] = 1
            A[i, j] = 1
            
        # Normalize the matrix
        DL = np.sum(A, 0)
        num_node = A.shape[0]
        Dn = np.zeros((num_node, num_node))
        for i in range(num_node):
            if DL[i] > 0:
                Dn[i, i] = DL[i] ** (-0.5)
        DAD = np.dot(np.dot(Dn, A), Dn)

        # Spatial Strategy (Split into 3 sub-graphs: Root, Closer, Further)
        if strategy == 'spatial':
            A = np.zeros((3, self.num_node, self.num_node))
            A[0] = np.eye(num_node) # Center (Self-loop)
            A[1] = DAD # Neighbors
            A[2] = DAD # (Usually distinct, but simplified here)
            return A
        else:
            return DAD
