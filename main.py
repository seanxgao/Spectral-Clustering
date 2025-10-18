from graph_models import generate_test_laplacian, generate_multi_group_laplacian, generate_layers_groups_graph, get_node_index, matrix_shuffle, visualize_laplacian_matrix, combine_three_figures, grade_matrix
from tests import *
from SCOPE import best_cut_finder, eigen_decomposition, bicut_group, treebuilder, matrixtype, sparse_score

L = generate_layers_groups_graph(1,250,20,0.8,0.3,0.05)
print(measure_time_and_memory(treebuilder, L, thre=10, parallel=False, manager=None))
