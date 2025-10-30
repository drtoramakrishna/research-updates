# GNN Resources & Further Learning

## 📚 Essential Papers

### Foundation Papers
1. **Graph Convolutional Networks** (Kipf & Welling, 2017)
   - Title: "Semi-Supervised Classification with Graph Convolutional Networks"
   - ArXiv: https://arxiv.org/abs/1609.02907
   - Impact: Foundational work on spectral-to-spatial GCN

2. **Graph Attention Networks** (Veličković et al., 2018)
   - Title: "Graph Attention Networks"
   - ArXiv: https://arxiv.org/abs/1710.10903
   - Impact: Introduced attention mechanism to GNNs

3. **GraphSAGE** (Hamilton et al., 2017)
   - Title: "Inductive Representation Learning on Large Graphs"
   - ArXiv: https://arxiv.org/abs/1706.02216
   - Impact: Enabled inductive learning and mini-batch training

### Advanced Architectures
4. **Graph Isomorphism Networks** (Xu et al., 2019)
   - ArXiv: https://arxiv.org/abs/1810.00826
   - Focus: Expressive power and Weisfeiler-Lehman test

5. **Heterogeneous Graph Neural Networks** (Wang et al., 2019)
   - Title: "Heterogeneous Graph Neural Network"
   - ArXiv: https://arxiv.org/abs/1903.08019
   - Focus: Handling heterogeneous node/edge types

6. **Temporal Graph Networks** (Trivedi et al., 2019)
   - ArXiv: https://arxiv.org/abs/1906.04289
   - Focus: Dynamic and temporal graphs

7. **Explainability in GNNs** (Ying et al., 2019)
   - Title: "GNNExplainer: Generating Explanations for Graph Neural Networks"
   - ArXiv: https://arxiv.org/abs/1903.03894

### Scalability & Efficiency
8. **Scaling GNNs** (Huang et al., 2021)
   - Title: "Towards Deep Graph Neural Networks"
   - Focus: Overcoming depth limitations

9. **Simplified GCN** (Wu et al., 2019)
   - Title: "Simplifying Graph Convolutional Networks"
   - ArXiv: https://arxiv.org/abs/1902.07153

## 🔧 Software Libraries

### PyTorch Ecosystem
- **PyTorch Geometric (PyG)**: Most popular, extensive docs
  - Installation: `pip install torch-geometric`
  - Docs: https://pytorch-geometric.readthedocs.io/
  
- **DGL (Deep Graph Library)**: Alternative implementation
  - Installation: `pip install dgl`
  - Docs: https://docs.dgl.ai/

- **Spektral**: Keras/TensorFlow focused
  - Installation: `pip install spektral`
  - Docs: https://spektral.graphneural.network/

### Graph Processing
- **NetworkX**: Classical graph algorithms
- **igraph**: Fast graph analysis
- **Graph-tool**: Efficient graph manipulation
- **Snap.py**: Stanford Network Analysis Project

## 📊 Benchmark Datasets

### Citation Networks
- **Cora**: 2,708 papers, 5,429 edges, 1,433 features, 7 classes
- **Citeseer**: 3,312 papers, 4,732 edges, 3,703 features, 6 classes
- **Pubmed**: 19,717 papers, 44,338 edges, 500 features, 3 classes

### Social Networks
- **Facebook**: 4,039 nodes, 176,468 edges
- **Twitter**: 81,306 nodes, 1,342,310 edges
- **Reddit**: 232,965 posts, 114,648,286 edges

### OGB (Open Graph Benchmark) - Large Scale
- **OGB-ArXiv**: 169,343 papers, 1,166,243 citations
- **OGB-Products**: 2,449,029 products, 61,859,140 connections
- **OGB-MAG**: 1,939,743 papers, 21,111,007 relationships

### Molecular Graphs
- **ZINC**: 250K drug-like molecules
- **QM9**: 134K small organic molecules
- **MolNet**: Multiple molecular property prediction tasks

### Knowledge Graphs
- **Freebase**: General knowledge graph
- **YAGO**: Knowledge extracted from Wikipedia
- **DBpedia**: Structured data from Wikipedia

## 🎓 Online Courses & Tutorials

### Comprehensive Courses
- **Stanford CS224W**: "Machine Learning with Graphs"
  - Instructors: Jure Leskovec, Marinka Žitnik
  - Link: http://web.stanford.edu/class/cs224w/
  
- **MIT 6.S897**: "Machine Learning for Healthcare"
  - Includes graph neural network module
  
- **UC Berkeley CS294**: "Deep Learning"
  - Section on GNNs and structured learning

### Online Resources
- **PyTorch Geometric Tutorials**: Step-by-step implementation
- **DGL Tutorials**: Comprehensive examples
- **Graph Neural Network Primer**: https://gdpr-info.eu/
- **Distill.pub**: Visual explanations of GNN concepts

## 💻 Implementation Tips & Best Practices

### Data Preprocessing
```python
# Normalize features
features = (features - features.mean(dim=0)) / features.std(dim=0)

# Create train/val/test split
train_idx = torch.arange(0, int(0.6 * num_nodes))
val_idx = torch.arange(int(0.6 * num_nodes), int(0.8 * num_nodes))
test_idx = torch.arange(int(0.8 * num_nodes), num_nodes)

# Add self-loops
edge_index = torch.cat([edge_index, torch.arange(num_nodes).repeat(2, 1)], dim=1)
```

### Model Architecture Patterns
```python
# Residual connections for deep networks
class ResGNNLayer(nn.Module):
    def forward(self, x, edge_index):
        out = self.gnn(x, edge_index)
        return x + out  # Skip connection

# Batch normalization for stability
class BatchNormGNN(nn.Module):
    def __init__(self, hidden_dim):
        self.bn = nn.BatchNorm1d(hidden_dim)
    
    def forward(self, x, edge_index):
        x = self.gnn(x, edge_index)
        return self.bn(x)
```

### Training Strategies
```python
# Learning rate scheduling
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=20, verbose=True
)

# Gradient clipping to prevent explosion
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

# Early stopping
if val_acc > best_acc:
    best_acc = val_acc
    best_state = model.state_dict()
    patience = 0
else:
    patience += 1
    if patience > 50:
        model.load_state_dict(best_state)
        break
```

## 🚀 Recent Advances (2023-2024)

### Emerging Architectures
- **Graph Transformer Networks**: Combining transformers with GNNs
- **Equivariant GNNs**: Incorporating geometric symmetries
- **Physics-Informed GNNs**: Combining with physical constraints
- **Neural ODEs for Graphs**: Continuous-time graph dynamics

### Scalability Solutions
- **Distributed GNNs**: Training across multiple GPUs/machines
- **Graph Sampling Strategies**: Importance sampling, layer sampling
- **Pruning & Quantization**: Reducing model size
- **Approximate Computation**: Trade-off accuracy for speed

### Application Areas
- **Protein Structure Prediction**: AlphaFold integration
- **Drug Discovery**: Molecular property prediction
- **Recommendation Systems**: Collaborative filtering with GNNs
- **Traffic Prediction**: Spatio-temporal graphs
- **Program Synthesis**: Code understanding via ASTs

## 📝 Common Pitfalls to Avoid

❌ **Using too many layers**: Leads to over-smoothing
✅ **Stick to 2-4 layers** for most tasks

❌ **Forgetting to normalize features**: Causes training instability
✅ **Always normalize** before training

❌ **No regularization**: Leads to overfitting
✅ **Use dropout, L2 regularization, early stopping**

❌ **Training on full graph**: Memory issues
✅ **Use mini-batch sampling** (GraphSAGE, FastGCN)

❌ **Same random seed everywhere**: Lucky initialization
✅ **Run multiple seeds**, report mean ± std

❌ **No validation set**: Can't detect overfitting
✅ **Always use proper train/val/test split**

## 🔗 Community & Collaboration

### Conferences
- **ICLR**: International Conference on Learning Representations
- **NeurIPS**: Neural Information Processing Systems
- **ICML**: International Conference on Machine Learning
- **AAAI**: Association for the Advancement of AI
- **WWW**: Web Conference
- **KDD**: Knowledge Discovery and Data Mining

### Workshops & Seminars
- **Learning on Graphs and Geometry Workshop**
- **Geometric Deep Learning Workshop**
- **Graph Representation Learning Workshop**

### GitHub Communities
- PyTorch Geometric: https://github.com/pyg-team/pytorch_geometric
- DGL: https://github.com/dmlc/dgl
- Papers with Code: https://paperswithcode.com/task/node-classification

## 📖 Textbooks & Books

1. **"Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges"**
   - Authors: Bronstein, Bruna, Cohen, Veličković
   - Available: https://geometricdeeplearning.com/

2. **"Graph Representation Learning"**
   - Authors: Hamilton, Ying, Leskovec
   - Comprehensive overview of representation learning

3. **"Spectral Graph Theory"**
   - Authors: Chung
   - Mathematical foundations

## 🎯 Next Steps for Learning

1. **Beginner Level**
   - Implement GCN from scratch
   - Train on Cora dataset
   - Visualize embeddings

2. **Intermediate Level**
   - Try different architectures (GAT, GraphSAGE)
   - Experiment with hyperparameters
   - Implement custom layers

3. **Advanced Level**
   - Work with large-scale graphs (OGB)
   - Explore heterogeneous/temporal GNNs
   - Contribute to open-source projects
   - Read recent papers and implement them

4. **Research Level**
   - Identify limitations of current methods
   - Design novel architectures
   - Apply to new domains
   - Publish findings

---

**Last Updated**: October 2025  
**Maintained by**: GNN Learning Community
