# Spectral Clustering API Reference

## Overview

This project implements a spectral clustering algorithm called SCOPE (Spectral Clustering fOr Pattern dEtection) that performs hierarchical graph clustering using spectral bisection. The algorithm builds a binary tree structure by recursively applying spectral cuts to partition graphs into clusters.

## Table of Contents

### Core Algorithm Functions
- [SCOPE.py](#scope-functions)
  - [best_cut_finder](#best_cut_finder)
  - [eigen_decomposition](#eigen_decomposition)
  - [bicut_group](#bicut_group)
  - [treebuilder](#treebuilder)
  - [matrixtype](#matrixtype)
  - [sparse_score](#sparse_score)
  - [pilot class](#pilot-class)
  - [BiCutNode class](#bicutnode-class)

### Graph Generation and Utilities
- [graph_models.py](#graph-models-functions)
  - [generate_test_laplacian](#generate_test_laplacian)
  - [generate_multi_group_laplacian](#generate_multi_group_laplacian)
  - [generate_layers_groups_graph](#generate_layers_groups_graph)
  - [matrix_shuffle](#matrix_shuffle)
  - [visualize_laplacian_matrix](#visualize_laplacian_matrix)
  - [combine_three_figures](#combine_three_figures)
  - [grade_matrix](#grade_matrix)

### Testing and Benchmarking
- [tests.py](#testing-functions)
  - [test_bicut_accuracy](#test_bicut_accuracy)
  - [measure_time_and_memory](#measure_time_and_memory)
  - [test_duration_memory](#test_duration_memory)
  - [run_one_graph_test](#run_one_graph_test)
  - [parallal_choices_test](#parallal_choices_test)
  - [read_csv_to_arrays](#read_csv_to_arrays)
  - [plot_results](#plot_results)

---

## SCOPE Functions

### `best_cut_finder`

```python
best_cut_finder(adj, gpu: bool, sparse: bool) -> int
```

**Purpose**: Find the optimal cut index for symmetric adjacency matrices along a 1D ordering (e.g., Fiedler vector sort).

**Parameters**:
- `adj`: Matrix (dense or CSR; NumPy/SciPy on CPU or CuPy/cupyx on GPU)
- `gpu`: Use GPU (CuPy) kernels if True, else CPU (NumPy/SciPy)
- `sparse`: Treat/convert to CSR and use O(nnz) path if True; else dense O(n²)

**Returns**:
- `int`: Best cut index in [1, n-1]

**Algorithm**: Uses prefix sums to efficiently compute the optimal cut that minimizes the ratio of cross-edges to group sizes.

---

### `eigen_decomposition`

```python
eigen_decomposition(L, gpu: bool, sparse: bool, k: int = 2)
```

**Purpose**: Compute the Fiedler vector (second smallest eigenvector) of a Laplacian matrix for spectral clustering.

**Parameters**:
- `L`: Laplacian matrix (numpy array, scipy sparse, or cupy array)
- `gpu`: Use GPU computation if True
- `sparse`: Use sparse eigensolver if True
- `k`: Number of eigenvalues to compute (default: 2, for Fiedler vector)

**Returns**:
- `numpy.ndarray`: Fiedler vector (second smallest eigenvector)

**Algorithm**: Uses ARPACK (CPU) or CuPy's eigensolver (GPU) to find the smallest eigenvalues. Automatically increases k if convergence fails.

---

### `bicut_group`

```python
bicut_group(L, gpueigen: bool = False, gpucut: bool = False, sparse: bool = False)
```

**Purpose**: Perform spectral bisection on a graph using the Fiedler vector to find optimal cut.

**Parameters**:
- `L`: Laplacian matrix of the graph
- `gpueigen`: Use GPU for eigenvector computation
- `gpucut`: Use GPU for cut finding
- `sparse`: Use sparse matrix operations

**Returns**:
- `tuple`: (first_group, second_group) - Two lists of node indices

**Algorithm**:
1. Compute Fiedler vector using `eigen_decomposition`
2. Sort nodes by Fiedler vector values
3. Find optimal cut using `best_cut_finder`
4. Return two groups of nodes

---

### `treebuilder`

```python
treebuilder(L, thre: int = None, indices: list = None, parallel: bool = True, manager: pilot = None)
```

**Purpose**: Recursively build a binary tree structure by applying spectral bisection to create hierarchical clusters.

**Parameters**:
- `L`: Full graph Laplacian matrix
- `thre`: Threshold for stopping recursion (nodes ≤ thre become leaves)
- `indices`: List of indices to process (None means all vertices)
- `parallel`: Use parallel processing for tree construction
- `manager`: Pilot object for managing GPU/sparse decisions

**Returns**:
- `BiCutNode`: Root of the binary tree structure

**Algorithm**:
1. Apply spectral bisection to current subgraph
2. Create sub-Laplacian matrices for each partition
3. Recursively build left and right subtrees
4. Return binary tree structure

---

### `matrixtype`

```python
matrixtype(L, sparse: bool, gpu: bool)
```

**Purpose**: Convert matrix to one of four standard types based on GPU and sparse preferences.

**Parameters**:
- `L`: Input matrix (any supported type)
- `sparse`: Target sparse format if True
- `gpu`: Target GPU format if True

**Returns**:
- Matrix in target format:
  - `gpu=True, sparse=True` → `cupyx.scipy.sparse.csr_matrix`
  - `gpu=True, sparse=False` → `cupy.ndarray`
  - `gpu=False, sparse=True` → `scipy.sparse.csr_matrix`
  - `gpu=False, sparse=False` → `numpy.ndarray`

---

### `sparse_score`

```python
sparse_score(L) -> float
```

**Purpose**: Calculate the sparsity score of a Laplacian matrix (ratio of diagonal sum to total elements).

**Parameters**:
- `L`: Laplacian matrix

**Returns**:
- `float`: Sparsity score between 0 and 1

**Note**: Only for Laplacian matrices. Higher values indicate denser matrices.

---

### `pilot` class

**Purpose**: Manages GPU and sparse computation decisions based on matrix properties and thresholds.

#### Methods:

##### `__init__(self)`
Initialize pilot with default settings.

##### `set_spthre(self, thr: float = 0.25)`
Set threshold for sparse computation decision.

##### `set_gputhre(self, thred_eig: int = 10000, thred_cut: int = 1000)`
Set thresholds for GPU computation decisions.

##### `copy(self)`
Create a copy of the pilot object.

---

### `BiCutNode` class

**Purpose**: Represents a node in the binary tree structure created by spectral clustering.

#### Methods:

##### `__init__(self, indices, left=None, right=None, parent=None)`
Initialize node with vertex indices and child nodes.

##### `is_leaf(self) -> bool`
Check if node is a leaf (no children).

##### `get_order(self) -> list`
Get the ordering of vertices by traversing the tree in-order.

##### `print_fancy_tree(self, prefix="", is_last=True, is_root=True)`
Print tree structure with box-drawing characters.

---

## Graph Models Functions

### `generate_test_laplacian`

```python
generate_test_laplacian(size1, size2, prob1, prob2, prob_between)
```

**Purpose**: Generate a random Laplacian matrix for testing spectral clustering with two distinct groups.

**Parameters**:
- `size1`: Size of first group
- `size2`: Size of second group
- `prob1`: Probability of edge within first group
- `prob2`: Probability of edge within second group
- `prob_between`: Probability of edge between groups

**Returns**:
- `numpy.ndarray`: Laplacian matrix L = D - A

---

### `generate_multi_group_laplacian`

```python
generate_multi_group_laplacian(num_groups, group_size, prob_within, prob_between)
```

**Purpose**: Generate a Laplacian matrix for multiple groups with uniform group sizes.

**Parameters**:
- `num_groups`: Number of groups
- `group_size`: Number of nodes in each group
- `prob_within`: Probability of edge within each group (0 to 1)
- `prob_between`: Probability of edge between different groups (0 to 1)

**Returns**:
- `numpy.ndarray`: Laplacian matrix L = D - A

---

### `generate_layers_groups_graph`

```python
generate_layers_groups_graph(
    num_supergroups=2,
    num_subgroups_per_supergroup=2,
    nodes_per_subgroup=5,
    p_intra_subgroup=0.8,
    p_intra_supergroup=0.3,
    p_inter_supergroup=0.05,
    seed=None
)
```

**Purpose**: Generate a hierarchical graph with three levels: supergroups → subgroups → nodes.

**Parameters**:
- `num_supergroups`: Number of top-level groups
- `num_subgroups_per_supergroup`: Number of subgroups per supergroup
- `nodes_per_subgroup`: Number of nodes per subgroup
- `p_intra_subgroup`: Probability of edge within same subgroup
- `p_intra_supergroup`: Probability of edge within same supergroup (different subgroups)
- `p_inter_supergroup`: Probability of edge between different supergroups
- `seed`: Random seed for reproducibility

**Returns**:
- `numpy.ndarray`: Laplacian matrix of the hierarchical graph

---

### `matrix_shuffle`

```python
matrix_shuffle(matrix) -> numpy.ndarray
```

**Purpose**: Randomly permute rows and columns of a matrix to test clustering recovery.

**Parameters**:
- `matrix`: Input matrix

**Returns**:
- `numpy.ndarray`: Permuted matrix

---

### `visualize_laplacian_matrix`

```python
visualize_laplacian_matrix(laplacian_matrix, show=True)
```

**Purpose**: Visualize a Laplacian matrix as a black and white image.

**Parameters**:
- `laplacian_matrix`: Laplacian matrix to visualize
- `show`: If True, display the plot; if False, return figure object

**Returns**:
- `matplotlib.figure` or `None`: Figure object if show=False

---

### `combine_three_figures`

```python
combine_three_figures(fig1, fig2, fig3, titles=None)
```

**Purpose**: Combine three existing figures into one figure with subplots.

**Parameters**:
- `fig1, fig2, fig3`: Matplotlib figure objects from `visualize_laplacian_matrix`
- `titles`: Optional list of titles for each subplot

**Returns**:
- `matplotlib.figure`: Combined figure

---

### `grade_matrix`

```python
grade_matrix(matrix) -> int
```

**Purpose**: Calculate grading energy: sum of (i-j) for all i>j where matrix[i,j]=1.

**Parameters**:
- `matrix`: 2D numpy array containing 0s and 1s

**Returns**:
- `int`: Grading energy value

**Note**: Used to measure how well a matrix is ordered (lower values indicate better ordering).

---

## Testing Functions

### `test_bicut_accuracy`

```python
test_bicut_accuracy(size1, size2, prob1, prob2, prob_between, num_tests=10, gpu=False, sparse=False)
```

**Purpose**: Test the accuracy of spectral clustering on generated graphs.

**Parameters**:
- `size1, size2`: Group sizes
- `prob1, prob2, prob_between`: Edge probabilities
- `num_tests`: Number of test iterations
- `gpu, sparse`: Computation options

**Returns**:
- `float`: Accuracy rate (0 to 1)

---

### `measure_time_and_memory`

```python
measure_time_and_memory(func, *args, **kwargs)
```

**Purpose**: Measure execution time and peak memory usage of a function call.

**Parameters**:
- `func`: Function to measure
- `*args, **kwargs`: Arguments to pass to function

**Returns**:
- `tuple`: (result, time_in_seconds, memory_in_MB)

---

### `test_duration_memory`

```python
test_duration_memory(oL, thre=None, show=False, use_parallel=False)
```

**Purpose**: Test the complete pipeline: shuffle → cluster → reorder → measure performance.

**Parameters**:
- `oL`: Original Laplacian matrix
- `thre`: Threshold for tree building
- `show`: Show visualization if True
- `use_parallel`: Use parallel tree building

**Returns**:
- `tuple`: (duration, peak_memory, energy_ratio)

**Pipeline**:
1. Shuffle the matrix
2. Build clustering tree
3. Get node ordering
4. Reorder matrix
5. Calculate energy ratio (ordered/original)

---

### `run_one_graph_test`

```python
run_one_graph_test(sup, sub, node, p_intrasub, p_intrasup, p_intersup, thre=1, parallel=False)
```

**Purpose**: Generate one hierarchical Laplacian and evaluate clustering performance.

**Parameters**:
- `sup, sub, node`: Graph structure parameters
- `p_intrasub, p_intrasup, p_intersup`: Edge probabilities
- `thre`: Tree building threshold
- `parallel`: Use parallel processing

**Returns**:
- `dict`: Contains 'duration', 'memory', and 'ratio' fields

---

### `parallal_choices_test`

```python
parallal_choices_test(iter, sup, sub, node, p_intrasub, p_intrasup, p_intersup, csv_filename="parallel_test_results.csv")
```

**Purpose**: Run scalability tests by scaling different graph dimensions.

**Parameters**:
- `iter`: Number of iterations
- `sup, sub, node`: Base graph structure
- `p_intrasub, p_intrasup, p_intersup`: Edge probabilities
- `csv_filename`: Output CSV file

**Algorithm**: For each iteration i:
1. Scale supergroups: (i×sup, sub, node)
2. Scale subgroups: (sup, i×sub, node)
3. Scale nodes: (sup, sub, i×node)

**Output**: CSV with columns: sup, sub, node, matrix_type, total_nodes, duration, memory, ratio

---

### `read_csv_to_arrays`

```python
read_csv_to_arrays(csv_filename='parallel_test_results.csv')
```

**Purpose**: Read benchmark results CSV into structured numpy arrays.

**Parameters**:
- `csv_filename`: Path to CSV file

**Returns**:
- `dict`: Structured data with keys 'supergroups', 'subgroups', 'nodes_per_sub', each containing arrays for sup, sub, node, total_nodes, duration, memory, ratio

---

### `plot_results`

```python
plot_results(csv_filename='parallel_test_results.csv')
```

**Purpose**: Plot benchmark results showing performance vs. total nodes.

**Parameters**:
- `csv_filename`: Path to CSV file with results

**Output**: Three subplots showing duration, memory, and ratio vs. total nodes, with separate lines for each scaling type.

---

## Usage Examples

### Basic Spectral Clustering

```python
from SCOPE import bicut_group, treebuilder
from graph_models import generate_test_laplacian

# Generate test graph
L = generate_test_laplacian(50, 50, 0.8, 0.8, 0.1)

# Single bisection
group1, group2 = bicut_group(L)

# Build complete tree
tree = treebuilder(L, thre=5)
ordering = tree.get_order()
```

### Performance Testing

```python
from tests import test_duration_memory, parallal_choices_test

# Test single graph
duration, memory, ratio = test_duration_memory(L, show=True)

# Run scalability benchmark
parallal_choices_test(10, 2, 2, 5, 0.8, 0.3, 0.05)
```

### Visualization

```python
from graph_models import visualize_laplacian_matrix, combine_three_figures

# Visualize single matrix
fig = visualize_laplacian_matrix(L, show=False)

# Combine multiple visualizations
combine_three_figures(fig1, fig2, fig3, titles=['Original', 'Shuffled', 'Restored'])
```

---

## Dependencies

- **NumPy**: Core numerical computations
- **SciPy**: Sparse matrices and linear algebra
- **Matplotlib**: Visualization
- **CuPy** (optional): GPU acceleration
- **cupyx.scipy** (optional): GPU sparse matrices

## Performance Notes

- GPU acceleration is available when CuPy is installed
- Sparse matrices are used for large graphs to save memory
- Parallel tree building is available for multi-core systems
- The algorithm automatically chooses optimal computation paths based on matrix properties