# Graph Neural Networks (GNNs) 
Graph Neural Networks (GNNs) are a class of neural networks designed to handle graph-structured data, where nodes represent entities and edges represent relationships between them. The core idea behind GNNs is to leverage the connections between nodes to learn better representations for tasks like node classification, graph classification, or link prediction.

### GNN Architecture

GNNs work by propagating and aggregating information across the graph structure. Here’s a general outline of the key components of a GNN:

1. **Node Features**: Each node $ v $ in the graph $ G $ has an associated feature vector $ \mathbf{x}_v \in \mathbb{R}^d $, where $ d $ is the dimensionality of the feature space.

2. **Message Passing**: For each node $ v $, the messages $ \mathbf{m}_{uv} $ sent to it from neighboring nodes $ u $ are computed. The message function $ \mathbf{m}_{uv} $ is often defined as:
   $$
   \mathbf{m}_{uv} = \text{MSG}(\mathbf{x}_u, \mathbf{x}_v, \mathbf{e}_{uv})
   $$
   where $ \mathbf{e}_{uv} $ represents the edge features between nodes $ u $ and $ v $, if applicable. 

3. **Aggregation Function**: The aggregation function $ \text{AGG} $ combines the messages received from all neighboring nodes. Common aggregation functions include summation, averaging, or pooling. For example:
   $$
   \mathbf{a}_v = \text{AGG}\left(\{ \mathbf{m}_{uv} \mid u \in \mathcal{N}(v) \} \right)
   $$
   where $ \mathcal{N}(v) $ denotes the set of neighbors of node $ v $, and $ \mathbf{a}_v $ is the aggregated message for node $ v $.

4. **Update Function**: The update function $ \text{UPDATE} $ combines the aggregated message $ \mathbf{a}_v $ with the node’s current features $ \mathbf{x}_v $ to update its state. This is typically performed using a neural network, such as a feed-forward network or a more complex network:
   $$
   \mathbf{x}_v^{\text{new}} = \text{UPDATE}(\mathbf{x}_v, \mathbf{a}_v)
   $$

5. **Layers**: GNNs consist of multiple layers, where each layer performs the message passing and updating steps. The final node features are obtained after several layers of such operations. For $ K $ layers, the feature vector of node $ v $ at layer $ k $ is denoted as $ \mathbf{x}_v^{(k)} $, where:
   $$
   \mathbf{x}_v^{(k+1)} = \text{UPDATE}\left(\mathbf{x}_v^{(k)}, \text{AGG}\left(\{ \mathbf{m}_{uv}^{(k)} \mid u \in \mathcal{N}(v) \} \right)\right)
   $$

6. **Readout Function**: After several layers of message passing, a readout function is applied to extract meaningful information from the graph. For node classification, the readout function might output node-specific features $ \mathbf{z}_v $. For graph classification, the graph-level readout aggregates features from all nodes in the graph:
   $$
   \mathbf{z}_G = \text{READOUT}\left(\{ \mathbf{x}_v^{(K)} \mid v \in G \} \right)
   $$

### Intuition Behind GNNs

The intuition behind GNNs is that nodes in a graph are not isolated, and their relationships with other nodes carry valuable information. GNNs allow the network to exploit this relational structure by allowing nodes to exchange information with their neighbors.

1. **Locality of Information**: Each node’s representation depends on the local graph structure.

2. **Homophily**: Connected nodes are likely to have similar properties.

3. **Relational Learning**: GNNs learn over entities and relationships simultaneously.

4. **Inductive Learning**: GNNs can generalize to unseen nodes or graphs.

### GNN Variants

There are several variants of GNNs, each with different message-passing and aggregation strategies:

- **Graph Convolutional Networks (GCN)**: Simplifies the message-passing process by considering only direct neighbors. The message update function is:
  $$
  \mathbf{x}_v^{\text{new}} = \text{ReLU}\left(\mathbf{W} \cdot \frac{1}{|\mathcal{N}(v)|} \sum_{u \in \mathcal{N}(v)} \mathbf{x}_u\right)
  $$
  where $ \mathbf{W} $ is a learnable weight matrix.

- **Graph Attention Networks (GAT)**: Uses attention mechanisms to weigh the importance of neighboring nodes dynamically:
  $$
  \mathbf{m}_{uv} = \alpha_{uv} \cdot \mathbf{x}_u
  $$
  where $ \alpha_{uv} $ is an attention coefficient computed by a learned attention mechanism.

- **GraphSAGE**: Introduces sampling techniques to handle large graphs by considering a fixed number of neighbors:
  $$
  \mathbf{a}_v = \text{AGG}\left(\{ \text{Sample}(\mathbf{m}_{uv}) \mid u \in \mathcal{N}(v) \} \right)
  $$

- **Graph Isomorphism Network (GIN)**: Designed to capture higher-order graph structures:
  $$
  \mathbf{x}_v^{\text{new}} = \text{MLP}\left(\mathbf{x}_v + \text{AGG}\left(\{ \mathbf{x}_u \mid u \in \mathcal{N}(v) \} \right)\right)
  $$

### Applications of GNNs

GNNs are applied in various domains, including:
- **Social Networks**
- **Knowledge Graphs**
- **Molecular Chemistry**
- **Recommendation Systems**
- **Computer Vision**

GNNs’ ability to leverage both node features and graph structure has made them a powerful tool for tasks involving graph-based data.

### References:
- [A Gentle Introduction to Graph - distill.pub](https://distill.pub/2021/gnn-intro/)
- [Graph Neural Network and Some of GNN Applications - Neptune.ai](https://neptune.ai/blog/graph-neural-network-and-some-of-gnn-applications)
- [Best Graph Neural Network architectures: GCN, GAT, MPNN and more - theAISummer.com](https://theaisummer.com/gnn-architectures/)
- [Graph Convolutional Networks - arXiv Paper](https://arxiv.org/abs/1609.02907)
- [Graph Attention Networks - arXiv Paper](https://arxiv.org/abs/1710.10903)
- [GraphSAGE - Inductive Representation Learning on Large Graphs- Stanford Paper](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf)
- [Graph Isomorphism Network (GIN) - HOW POWERFUL ARE GRAPH NEURAL NETWORKS? - Arxiv Paper](https://arxiv.org/pdf/1810.00826)