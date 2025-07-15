import torch
import torch.nn as nn
import torch.nn.functional as F

class ProtoNet18(nn.Module):
    def __init__(self, backbone, embedding_dim=512, **backbone_kwargs):
        """
        backbone: a nn.Module class (e.g. ResNet18)
        embedding_dim: dimensionality of the output embeddings
        backbone_kwargs: any kwargs passed to backbone constructor (e.g. num_classes)
        """
        super(ProtoNet18, self).__init__()
        # instantiate backbone and remove its final FC
        self.encoder = backbone(**backbone_kwargs)
        # assume backbone has attribute `fc` returning (batch, embedding_dim)
        self.encoder.fc = nn.Identity()
        self.embedding_dim = embedding_dim

    def forward(self, support_imgs, support_labels, query_imgs):
        """
        support_imgs:  Tensor of shape [N_supp, C, H, W]
        support_labels: LongTensor of shape [N_supp] with labels 0..n_way-1
        query_imgs:    Tensor of shape [N_query, C, H, W]
        Returns:
          logits: Tensor of shape [N_query, n_way], where
                  logits[i, j] = -||f(query_i) - prototype_j||^2
        """
        # embed support and query
        supp_emb, _ = self.encoder(support_imgs)   # [N_supp, D]
        qry_emb, _  = self.encoder(query_imgs)     # [N_query, D]

        # find unique classes in support set
        classes = torch.unique(support_labels)
        n_way = len(classes)
        #print(f"Number of classes: {n_way}, Classes: {classes}")

        # compute prototypes: mean embedding per class
        prototypes = []
        for c in classes:
            mask = support_labels == c
            #print(mask)
            #print(supp_emb)
            #print(f"Class {c}: {mask.sum()} samples")
            #print(f"Support embeddings shape: {supp_emb[mask].shape}")
            # supp_emb[mask] is [n_samples, D] for this class
            proto = supp_emb[mask].mean(dim=0)   # [D]
            #print(f"Prototype for class {c}: {proto.shape}")
            prototypes.append(proto)
        prototypes = torch.stack(prototypes)   # [n_way, D]

        # compute squared Euclidean distances
        #   qry_emb:    [N_query, 1, D]
        #   prototypes: [1, n_way, D]
        dists = (qry_emb.unsqueeze(1) - prototypes.unsqueeze(0)).pow(2).sum(dim=2)  
        # [N_query, n_way]

        # logits are negative distances
        logits = -dists
        return logits


class ProtoNet50(nn.Module):
    def __init__(self, backbone, embedding_dim=2048, **backbone_kwargs):
        """
        backbone: a nn.Module class (e.g. ResNet50)
        embedding_dim: dimensionality of the output embeddings
        backbone_kwargs: any kwargs passed to backbone constructor (e.g. num_classes)
        """
        super(ProtoNet50, self).__init__()
        # instantiate backbone and remove its final FC
        self.encoder = backbone(**backbone_kwargs)
        self.encoder.fc = nn.Identity()
        self.embedding_dim = embedding_dim

    def forward(self, support_imgs, support_labels, query_imgs):
        return ProtoNet18(self.encoder, self.embedding_dim).forward(support_imgs, support_labels, query_imgs)