import numpy as np
import scipy.sparse as sp
import torch
from sklearn.preprocessing import OneHotEncoder


import os
import pickle

def encode_onehot(labels):
    encoder = OneHotEncoder(sparse_output=False, dtype=np.int32)
    labels_onehot = encoder.fit_transform(labels.reshape(-1, 1))
    labels_idx = encoder.transform(labels.reshape(-1, 1)).argmax(axis=1)
    return labels_onehot, labels_idx, encoder.categories_[0], encoder


def load_data(path=r"data_gcn/cora/", graph_type="gcn", device="cpu"):
    print(f"Loading Cora dataset for {graph_type.upper()}...")

    idx_features_labels = np.genfromtxt(f"{path}cora.content", dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels_onehot, labels_idx, classes, encoder = encode_onehot(idx_features_labels[:, -1])

    # Training uses indices
    labels = torch.LongTensor(labels_idx).to(device)


    # Step 2: Graph structure (edges)
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(f"{path}cora.cites", dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)

    # Step 3: Depending on GNN type
    if graph_type == "gcn":
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)

        # make symmetric + add self-loops
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = normalize(adj + sp.eye(adj.shape[0]))

        graph_struct = sparse_mx_to_torch_sparse_tensor(adj)

    elif graph_type == "gat":
        # build edge index with self-loops
        src, tgt = edges[:, 0], edges[:, 1]
        src = np.concatenate([src, np.arange(labels.shape[0])])
        tgt = np.concatenate([tgt, np.arange(labels.shape[0])])
        edge_index = np.vstack((src, tgt))

        graph_struct = torch.tensor(edge_index, dtype=torch.long, device=device)

    else:
        raise ValueError("graph_type must be 'gcn' or 'gat'")

    # Step 4: Normalize features (same for both)
    features = normalize(features)
    features = torch.FloatTensor(np.array(features.todense())).to(device)
    labels = torch.LongTensor(labels).to(device)

    # Step 5: Splits (you can parameterize too)
    if graph_type == "gcn":
        idx_train = torch.LongTensor(range(140)).to(device)
        idx_val = torch.LongTensor(range(200, 500)).to(device)
        idx_test = torch.LongTensor(range(500, 1500)).to(device)
    else:  # GAT
        idx_train = torch.arange(0, 140, dtype=torch.long, device=device)
        idx_val = torch.arange(140, 640, dtype=torch.long, device=device)
        idx_test = torch.arange(1708, 2708, dtype=torch.long, device=device)

    return graph_struct, features, labels, idx_train, idx_val, idx_test



def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

# -------------------------------- GAT utils --------------------------------


def pickle_read(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def normalize_features_sparse(node_features_sparse):
    """Row-normalize sparse features"""
    assert sp.issparse(node_features_sparse), f"Expected sparse matrix, got {type(node_features_sparse)}"
    rowsum = np.array(node_features_sparse.sum(1))
    r_inv = np.power(rowsum, -1).squeeze()
    r_inv[np.isinf(r_inv)] = 1.
    diag_inv = sp.diags(r_inv)
    return diag_inv.dot(node_features_sparse)

def build_edge_index(adjacency_list_dict, num_nodes, add_self_loops=True):
    """Convert adjacency dict to edge index (2 x num_edges)"""
    src, tgt = [], []
    seen = set()
    for node, neighbors in adjacency_list_dict.items():
        for nbr in neighbors:
            if (node, nbr) not in seen:
                src.append(node)
                tgt.append(nbr)
                seen.add((node, nbr))
    if add_self_loops:
        src.extend(np.arange(num_nodes))
        tgt.extend(np.arange(num_nodes))
    return np.row_stack((src, tgt))

def load_data_gat(data_path="data_gat/cora", device="cpu"):
    """
    Load Cora dataset for GAT from pickled files.
    Returns: edge_index, features, labels, idx_train, idx_val, idx_test
    """
    # Load data
    features_sparse = pickle_read(os.path.join(data_path, "node_features.csr"))
    labels_np = pickle_read(os.path.join(data_path, "node_labels.npy"))
    adj_dict = pickle_read(os.path.join(data_path, "adjacency_list.dict"))
    
    # Normalize features
    features_sparse = normalize_features_sparse(features_sparse)
    features = torch.tensor(features_sparse.todense(), dtype=torch.float32, device=device)
    
    labels = torch.tensor(labels_np, dtype=torch.long, device=device)
    
    num_nodes = len(labels_np)
    edge_index = torch.tensor(build_edge_index(adj_dict, num_nodes), dtype=torch.long, device=device)
    
    # Train / val / test splits (same as your original GAT script)
    idx_train = torch.arange(0, 140, dtype=torch.long, device=device)
    idx_val = torch.arange(140, 640, dtype=torch.long, device=device)
    idx_test = torch.arange(1708, 2708, dtype=torch.long, device=device)
    
    return edge_index, features, labels, idx_train, idx_val, idx_test
