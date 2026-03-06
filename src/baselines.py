"""
Baseline Policies for Comparison with HGT-Scheduler.

Implements:
1. HomoHGTPolicy: HGTConv with merged edge types (ablation for edge-type heterogeneity)
   FT06: makespan 66.0 ± 2.6 (gap 20.0%); FT10: 1540.6 ± 163.0 (gap 65.7%)
2. GINPolicy: Graph Isomorphism Network (L2D-style homogeneous baseline)
   FT06: makespan 66.6 ± 8.8 (gap 21.1%); FT10: 1994.8 ± 361.5 (gap 114.5%)

HGT-Scheduler outperforms HomoHGT on FT06 (p=0.011), validating the
contribution of type-dependent attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv, GINConv, Linear, global_mean_pool
from torch_geometric.data import HeteroData, Batch
from typing import Tuple, Optional, Dict
import numpy as np


class GlobalAttentionPooling(nn.Module):
    """Global attention pooling (shared with Policy.py)."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.gate_nn = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, 1)
        )

    def forward(self, x: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        if batch is None:
            gate_values = self.gate_nn(x)
            attention_weights = F.softmax(gate_values, dim=0)
            return torch.sum(attention_weights * x, dim=0, keepdim=True)
        else:
            gate_values = self.gate_nn(x)
            num_graphs = int(batch.max().item()) + 1
            graph_embeddings = []
            for i in range(num_graphs):
                mask = (batch == i)
                gate_i = gate_values[mask]
                x_i = x[mask]
                attention_weights = F.softmax(gate_i, dim=0)
                graph_emb = torch.sum(attention_weights * x_i, dim=0, keepdim=True)
                graph_embeddings.append(graph_emb)
            return torch.cat(graph_embeddings, dim=0)


# =====================================================================
# 1. Homogeneous HGT (ablation: merged edge types)
# =====================================================================

class HomoHGTEncoder(nn.Module):
    """
    Homogeneous HGT Encoder.

    Same as HGTEncoder but merges 'precedes' and 'competes' edges into
    a single edge type. This is the ablation baseline that removes
    edge-type heterogeneity.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        metadata = (
            ['op'],
            [('op', 'edge', 'op')]   # single merged edge type
        )

        for _ in range(num_layers):
            conv = HGTConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                metadata=metadata,
                heads=num_heads,
            )
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_channels))

        self.output_proj = Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x_dict = {'op': self.input_proj(x)}
        edge_index_dict = {('op', 'edge', 'op'): edge_index}

        for conv, norm in zip(self.convs, self.norms):
            x_dict_out = conv(x_dict, edge_index_dict)
            x_dict['op'] = x_dict['op'] + self.dropout(x_dict_out['op'])
            x_dict['op'] = norm(x_dict['op'])
            x_dict['op'] = F.relu(x_dict['op'])

        return self.output_proj(x_dict['op'])


class HomoHGTPolicy(nn.Module):
    """
    Homogeneous HGT Policy (ablation baseline).

    Identical to HGTPolicy but treats all edges as the same type.
    Isolates the contribution of edge-type heterogeneity.
    """

    def __init__(
        self,
        node_feature_dim: int = 3,
        hidden_dim: int = 128,
        embedding_dim: int = 64,
        num_hgt_layers: int = 3,
        num_heads: int = 4,
        actor_hidden_dim: int = 128,
        critic_hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.encoder = HomoHGTEncoder(
            in_channels=node_feature_dim,
            hidden_channels=hidden_dim,
            out_channels=embedding_dim,
            num_layers=num_hgt_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.global_pool = GlobalAttentionPooling(embedding_dim)

        self.actor = nn.Sequential(
            nn.Linear(embedding_dim * 2, actor_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(actor_hidden_dim, actor_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(actor_hidden_dim, 1),
        )

        self.critic = nn.Sequential(
            nn.Linear(embedding_dim, critic_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(critic_hidden_dim, critic_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(critic_hidden_dim, 1),
        )

    def _merge_edges(self, data: HeteroData) -> torch.Tensor:
        """Merge precedes and competes edges into single edge_index."""
        precedes = data['op', 'precedes', 'op'].edge_index
        competes = data['op', 'competes', 'op'].edge_index
        return torch.cat([precedes, competes], dim=1)

    def forward(self, data: HeteroData, batch: Optional[torch.Tensor] = None):
        x = data['op'].x
        edge_index = self._merge_edges(data)

        if batch is None and hasattr(data['op'], 'batch') and data['op'].batch is not None:
            batch = data['op'].batch

        node_embeddings = self.encoder(x, edge_index)
        graph_embedding = self.global_pool(node_embeddings, batch)

        num_nodes = node_embeddings.shape[0]
        if batch is None:
            graph_embedding_expanded = graph_embedding.expand(num_nodes, -1)
        else:
            graph_embedding_expanded = graph_embedding[batch]

        combined = torch.cat([node_embeddings, graph_embedding_expanded], dim=-1)
        action_logits = self.actor(combined).squeeze(-1)

        if batch is None:
            value = self.critic(graph_embedding).squeeze(-1)
        else:
            value = self.critic(graph_embedding).squeeze(-1)

        return action_logits, value

    def get_action_and_value(
        self,
        data: HeteroData,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ):
        logits, value = self.forward(data)
        action_mask = data['op'].action_mask
        masked_logits = logits.clone()
        masked_logits[~action_mask] = -1e8

        dist = torch.distributions.Categorical(logits=masked_logits)

        if action is None:
            action = masked_logits.argmax() if deterministic else dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value

    def get_value(self, data: HeteroData) -> torch.Tensor:
        _, value = self.forward(data)
        return value

    def evaluate_actions(self, data: HeteroData, actions: torch.Tensor):
        logits, values = self.forward(data)
        action_mask = data['op'].action_mask

        if hasattr(data['op'], 'batch') and data['op'].batch is not None:
            node_batch = data['op'].batch
            num_graphs = int(node_batch.max().item()) + 1
            log_probs_list, entropies_list = [], []
            for i in range(num_graphs):
                node_mask = (node_batch == i)
                logits_i = logits[node_mask]
                mask_i = action_mask[node_mask]
                masked_logits_i = logits_i.clone()
                masked_logits_i[~mask_i] = -1e8
                dist_i = torch.distributions.Categorical(logits=masked_logits_i)
                log_probs_list.append(dist_i.log_prob(actions[i]))
                entropies_list.append(dist_i.entropy())
            return torch.stack(log_probs_list), values, torch.stack(entropies_list)
        else:
            masked_logits = logits.clone()
            masked_logits[~action_mask] = -1e8
            dist = torch.distributions.Categorical(logits=masked_logits)
            return dist.log_prob(actions), values, dist.entropy()


# =====================================================================
# 2. GIN Policy (L2D-style baseline)
# =====================================================================

class GINEncoder(nn.Module):
    """
    Graph Isomorphism Network Encoder (L2D-style).

    Uses GINConv layers with merged edge types, similar to the
    L2D baseline (Zhang et al., NeurIPS 2020).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels * 2),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_channels * 2),
                nn.Linear(hidden_channels * 2, hidden_channels),
            )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.norms.append(nn.LayerNorm(hidden_channels))

        self.output_proj = nn.Linear(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)

        for conv, norm in zip(self.convs, self.norms):
            x_new = conv(x, edge_index)
            x = x + self.dropout(x_new)
            x = norm(x)
            x = F.relu(x)

        return self.output_proj(x)


class GINPolicy(nn.Module):
    """
    GIN-based Actor-Critic Policy (L2D-style baseline).

    Uses Graph Isomorphism Network with merged edges,
    comparable to the L2D framework.
    """

    def __init__(
        self,
        node_feature_dim: int = 3,
        hidden_dim: int = 128,
        embedding_dim: int = 64,
        num_gin_layers: int = 3,
        actor_hidden_dim: int = 128,
        critic_hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.encoder = GINEncoder(
            in_channels=node_feature_dim,
            hidden_channels=hidden_dim,
            out_channels=embedding_dim,
            num_layers=num_gin_layers,
            dropout=dropout,
        )

        self.global_pool = GlobalAttentionPooling(embedding_dim)

        self.actor = nn.Sequential(
            nn.Linear(embedding_dim * 2, actor_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(actor_hidden_dim, actor_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(actor_hidden_dim, 1),
        )

        self.critic = nn.Sequential(
            nn.Linear(embedding_dim, critic_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(critic_hidden_dim, critic_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(critic_hidden_dim, 1),
        )

    def _merge_edges(self, data: HeteroData) -> torch.Tensor:
        """Merge precedes and competes edges into single edge_index."""
        precedes = data['op', 'precedes', 'op'].edge_index
        competes = data['op', 'competes', 'op'].edge_index
        return torch.cat([precedes, competes], dim=1)

    def forward(self, data: HeteroData, batch: Optional[torch.Tensor] = None):
        x = data['op'].x
        edge_index = self._merge_edges(data)

        if batch is None and hasattr(data['op'], 'batch') and data['op'].batch is not None:
            batch = data['op'].batch

        node_embeddings = self.encoder(x, edge_index)
        graph_embedding = self.global_pool(node_embeddings, batch)

        num_nodes = node_embeddings.shape[0]
        if batch is None:
            graph_embedding_expanded = graph_embedding.expand(num_nodes, -1)
        else:
            graph_embedding_expanded = graph_embedding[batch]

        combined = torch.cat([node_embeddings, graph_embedding_expanded], dim=-1)
        action_logits = self.actor(combined).squeeze(-1)

        if batch is None:
            value = self.critic(graph_embedding).squeeze(-1)
        else:
            value = self.critic(graph_embedding).squeeze(-1)

        return action_logits, value

    def get_action_and_value(
        self,
        data: HeteroData,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ):
        logits, value = self.forward(data)
        action_mask = data['op'].action_mask
        masked_logits = logits.clone()
        masked_logits[~action_mask] = -1e8

        dist = torch.distributions.Categorical(logits=masked_logits)

        if action is None:
            action = masked_logits.argmax() if deterministic else dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value

    def get_value(self, data: HeteroData) -> torch.Tensor:
        _, value = self.forward(data)
        return value

    def evaluate_actions(self, data: HeteroData, actions: torch.Tensor):
        logits, values = self.forward(data)
        action_mask = data['op'].action_mask

        if hasattr(data['op'], 'batch') and data['op'].batch is not None:
            node_batch = data['op'].batch
            num_graphs = int(node_batch.max().item()) + 1
            log_probs_list, entropies_list = [], []
            for i in range(num_graphs):
                node_mask = (node_batch == i)
                logits_i = logits[node_mask]
                mask_i = action_mask[node_mask]
                masked_logits_i = logits_i.clone()
                masked_logits_i[~mask_i] = -1e8
                dist_i = torch.distributions.Categorical(logits=masked_logits_i)
                log_probs_list.append(dist_i.log_prob(actions[i]))
                entropies_list.append(dist_i.entropy())
            return torch.stack(log_probs_list), values, torch.stack(entropies_list)
        else:
            masked_logits = logits.clone()
            masked_logits[~action_mask] = -1e8
            dist = torch.distributions.Categorical(logits=masked_logits)
            return dist.log_prob(actions), values, dist.entropy()
