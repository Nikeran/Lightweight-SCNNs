import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse, add_self_loops
from itertools import combinations
from torchvision.models import resnet18


def build_simplicial_adjacency(features, threshold=0.8):
    B, F, H, W = features.shape
    N = H * W

    feats = features.view(B, F, N).permute(0, 2, 1)  # (B, N, F)

    # Similarity-based adjacency
    sim = F.cosine_similarity(feats.unsqueeze(2), feats.unsqueeze(1), dim=-1)
    adj = (sim > threshold).float()

    return adj

def normalize_incidence(edges_list, tris_list, N):
    """
    Compute normalized incidence matrices for batches:
      - A01_hat_list: list of (N x E_b) node→edge incidences
      - A12_hat_list: list of (E_b x T_b) edge→triangle incidences
    where E_b, T_b vary per batch sample.
    """
    A01_hat_list = []
    A12_hat_list = []

    for edge_idx, tri_idx in zip(edges_list, tris_list):
        # --- 1) Unique undirected edges (i<j) ---
        src, dst = edge_idx
        mask = src < dst
        src_u = src[mask]
        dst_u = dst[mask]
        E = src_u.size(0)

        # --- 2) Build binary incidence A01 (N x E) ---
        A01 = edge_idx.new_zeros((N, E), dtype=torch.float)
        idx = torch.arange(E, device=src_u.device)
        A01[src_u, idx] = 1.0
        A01[dst_u, idx] = 1.0

        # --- 3) Degree normalization for A01 ---
        deg0 = A01.sum(dim=1)               # node degrees
        deg1 = A01.sum(dim=0)               # edge sizes (=2)
        d0 = torch.pow(deg0.clamp(min=1), -0.5)
        d1 = torch.pow(deg1.clamp(min=1), -0.5)
        A01_hat = d0.view(-1,1) * A01 * d1.view(1,-1)

        # --- 4) Build A12 if triangles exist ---
        if tri_idx.numel() > 0:
            T = tri_idx.size(1)
            # Map each undirected edge (u,v) → its index in [0..E-1]
            edge_pairs = list(zip(src_u.tolist(), dst_u.tolist()))
            edge_map = {edge_pairs[i]: i for i in range(E)}

            A12 = A01.new_zeros((E, T))
            for t in range(T):
                nodes = tri_idx[:,t].tolist()
                for u, v in combinations(nodes, 2):
                    key = (u, v) if u < v else (v, u)
                    if key in edge_map:
                        A12[edge_map[key], t] = 1.0

            # Degree normalization for A12
            deg1_12 = A12.sum(dim=1)          # edges → count of incident triangles
            deg2    = A12.sum(dim=0)          # triangle sizes (=3)
            d1_12 = torch.pow(deg1_12.clamp(min=1), -0.5)
            d2    = torch.pow(deg2.clamp(min=1),    -0.5)
            A12_hat = d1_12.view(-1,1) * A12 * d2.view(1,-1)
        else:
            A12_hat = edge_idx.new_zeros((E, 0))

        A01_hat_list.append(A01_hat)
        A12_hat_list.append(A12_hat)

    return A01_hat_list, A12_hat_list



def build_simplicial_complex(adj):
    """
    Given a batch-wise adjacency matrix (B, N, N),
    build edge_index (0↔1), tri_index (1↔2), and node-tri incidence.
    """
    B, N, _ = adj.shape
    # Build edge list from adj threshold
    edge_list = []
    for b in range(B):
        edge_idx, _ = dense_to_sparse(adj[b])
        edge_list.append(edge_idx)  # shape (2, E)
    # Build triangles (cliques of size 3)
    tri_list = []
    for b in range(B):
        edges = set([(int(i), int(j)) for i,j in edge_list[b].t().tolist()])
        tris = []
        for i, j, k in combinations(range(N), 3):
            if (i,j) in edges and (j,k) in edges and (i,k) in edges:
                tris.append((i, j, k))
        if tris:
            tri_list.append(torch.tensor(tris, dtype=torch.long).t())  # (3, T)
        else:
            tri_list.append(torch.empty((3, 0), dtype=torch.long))
    return edge_list, tri_list

class SCNLayer(nn.Module):
    def __init__(self, in_ch, out_ch, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        # learnable feature transforms for 0,1,2-simplices
        self.Q0 = nn.Linear(in_ch, out_ch, bias=False)
        self.Q1 = nn.Linear(in_ch, out_ch, bias=False)
        self.Q2 = nn.Linear(in_ch, out_ch, bias=False)

    def forward(self, X0, X1, X2, A01_hat, A12_hat):
        """
        X0: (B, N, F), X1: (B, E, F), X2: (B, T, F)
        A01_hat: normalized incidence (B, N, E)
        A12_hat: normalized incidence (B, E, T)
        """
        # 0-simplex update:
        # term1: self via edges
        term1_0 = self.alpha * torch.einsum('bne,be,ben->bnf',
            A01_hat, torch.ones(A01_hat.size(1), device=X0.device),
            torch.einsum('ben,bnf->benf', A01_hat, self.Q0(X0)))
        # term2: from edges
        term2_0 = (1-self.alpha) * torch.einsum('bne,bef->bnf',
            A01_hat, self.Q1(X1))
        # term3: self via triangles
        A02_hat = torch.einsum('bne,bet->bnt', A01_hat, A12_hat)  # node-tri incidence
        term3_0 = self.alpha * torch.einsum('bnt,btf->bnf', A02_hat, self.Q0(X0))
        # term4: from triangles
        term4_0 = (1-self.alpha) * torch.einsum('bnt,btf->bnf', A02_hat, self.Q2(X2))
        X0_new = F.relu(term1_0 + term2_0 + term3_0 + term4_0)

        # 1-simplex update:
        # term1: self via triangles + via nodes
        term1_1 = self.alpha * (
            torch.einsum('bet,btf->bef', A12_hat, torch.einsum('ben,bnf->benf', A01_hat, self.Q1(X1))) +
            torch.einsum('ben,bnf->bef', A01_hat, torch.einsum('bne,bnf->bnf', A01_hat, self.Q1(X1)))
        )
        # term2: from nodes and triangles
        term2_1 = (1-self.alpha) * (
            torch.einsum('ben,bnf->bef', A01_hat, self.Q0(X0)) +
            torch.einsum('bet,btf->bef', A12_hat, self.Q2(X2))
        )
        X1_new = F.relu(term1_1 + term2_1)

        # 2-simplex update:
        # term1: self via edges
        term1_2 = self.alpha * torch.einsum('bet,bef->btf', A12_hat, torch.einsum('ben,bef->benf', A01_hat, self.Q2(X2)))
        # term2: from edges
        term2_2 = (1-self.alpha) * torch.einsum('bet,bef->btf', A12_hat, self.Q1(X1))
        X2_new = F.relu(term1_2 + term2_2)

        return X0_new, X1_new, X2_new

# Example integration into your model
class ExpandedSimplicialResNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        base = resnet18(pretrained=False)
        self.stem = nn.Sequential(base.conv1, base.bn1, base.relu,
                                  base.maxpool, base.layer1, base.layer2)
        self.layer3, self.layer4 = base.layer3, base.layer4
        self.scn = SCNLayer(in_ch=128, out_ch=128, alpha=0.7)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512+128, num_classes)

    def forward(self, x):
        B = x.size(0)
        x = self.stem(x)      # (B,128,H',W')
        B, C, H, W = x.shape
        # build complex
        feat = x.view(B, C, -1).permute(0,2,1)  # (B,N,F)
        adj = build_simplicial_adjacency(x)     # your threshold graph
        edges, tris = build_simplicial_complex(adj)
        # compute normalized incidence A01_hat, A12_hat (not shown)
        A01_hat, A12_hat = normalize_incidence(edges, tris, N=H*W)
        # initial features
        X0, X1, X2 = feat, feat.new_zeros(B, len(edges[0].t()), C), feat.new_zeros(B, len(tris[0].t()), C)
        # one SCN layer
        X0, X1, X2 = self.scn(X0, X1, X2, A01_hat, A12_hat)
        # decode node features back to grid
        simp_feat = X0.permute(0,2,1).view(B,128,H,W)
        # continue ResNet
        x2 = self.layer3(x)
        x2 = self.layer4(x2)
        out = torch.cat([self.pool(x2).view(B,-1), self.pool(simp_feat).view(B,-1)], dim=-1)
        return self.fc(out)
