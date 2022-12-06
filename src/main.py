"""
Project details:
----------------

Paper title:
Node2vec: Scalable Feature Learning for Networks 

Group members:
A.Sudarshan (2019B4A70744P)
Rahul Balike (2019A3PS0189P)
Manpreet Singh Ahluwalia (2020A3PS0419P)

Teaching Assistant:
Sarthak Gupta (2019B4A70464P)
"""


import argparse
import numpy as np
import networkx as nx
import node2vec
from gensim.models import Word2Vec, KeyedVectors


def parse_args():
	"""Parses the node2vec arguments:
	input: Path of file containing data of input graph
	output: Path of file to store learnt embeddings
	dimensions: Dimensions of embedding space (default is 128)
	walk-length: Length of walk per source node
	num-walks: Number of walks per source node
	window-size: Context size for optimization (default is 10)
	iter: Number of epochs for SGD (default is 1)
	workers: Number of parallel workers (for training parallelization - speed up training)
	p: Return hyperparameter (default is 1)
	q: InOut hyperparameter (default is 1)
	weighted: Boolean specifying (un)weighted. Default is unweighted
	directed: Boolean specifying (un)directed. Default is undirected

	Returns:
		parser.parse_args: The arguments required to run node2vec
	"""
	parser = argparse.ArgumentParser(description="Run node2vec.")

	parser.add_argument('--input', nargs='?', default='graph/karate.edgelist',
	                    help='Input graph path')

	parser.add_argument('--output', nargs='?', default='emb/karate.emb',
	                    help='Embeddings path')

	parser.add_argument('--dimensions', type=int, default=128,
	                    help='Number of dimensions. Default is 128.')

	parser.add_argument('--walk-length', type=int, default=80,
	                    help='Length of walk per source. Default is 80.')

	parser.add_argument('--num-walks', type=int, default=10,
	                    help='Number of walks per source. Default is 10.')

	parser.add_argument('--window-size', type=int, default=10,
                    	help='Context size for optimization. Default is 10.')

	parser.add_argument('--iter', default=1, type=int,
                      help='Number of epochs in SGD. Default is 1.')

	parser.add_argument('--workers', type=int, default=8,
	                    help='Number of parallel workers. Default is 8.')

	parser.add_argument('--p', type=float, default=1,
	                    help='Return hyperparameter. Default is 1.')

	parser.add_argument('--q', type=float, default=1,
	                    help='Inout hyperparameter. Default is 1.')

	parser.add_argument('--weighted', dest='weighted', action='store_true',
	                    help='Boolean specifying (un)weighted. Default is unweighted.')
	parser.add_argument('--unweighted', dest='unweighted', action='store_false')
	parser.set_defaults(weighted=False)

	parser.add_argument('--directed', dest='directed', action='store_true',
	                    help='c.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')
	parser.set_defaults(directed=False)

	return parser.parse_args()

def read_graph():
	"""Reads the input network in networkx.

	Returns:
		G: Networkx graph having information of the input network
	"""
	# By default create a directed graph (DiGraph) by reading the edge list
	if args.weighted: # If graph is weighted
		G = nx.read_edgelist(args.input, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
	else: # If graph is unweighted
		G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
		# Assign edge weights to 1 (since unweighted)
		for edge in G.edges():
			G[edge[0]][edge[1]]['weight'] = 1

	# If network is undirected, we convert G to undirected graph
	if not args.directed:
		G = G.to_undirected()

	return G

def learn_embeddings(walks):
	"""Learn embeddings by optimizing the Skipgram objective using SGD.

	Args:
		walks (List): List of simulated node2vec walks

	Returns:
		model: Trained model ready for visualization
	"""
	# Convert list of lists to a list of strings
	# by representing each walk as a string of nodes rather than a list of traveresed nodes
	walks = [list(map(str, walk)) for walk in walks]
	# Instantiate the word2vec model
	model = Word2Vec(sentences=walks, vector_size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers, epochs=args.iter)
	# Load the KeyedVectors of trained model
	KeyedVectors.load_word2vec_format(args.output)
	
	return model

def visualise_embeddings(model):
	# Retrieve node embeddings and corresponding subjects
	node_ids = model.wv.index2word  # list of node IDs
	node_embeddings = (model.wv.vectors)
	# numpy.ndarray of size number of nodes times embeddings dimensionality
	# node_targets = node_subjects[[int(node_id) for node_id in node_ids]]

def main(args):
	"""Pipeline for representational learning for all nodes in a graph.

	Args:
		args (parse_args): Arguments for the embeddings as defined in parse_args function
	"""
	# Convert network data to a graph
	nx_G = read_graph()
	# Call node2vec
	G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
	# Compute transition probabilities
	G.preprocess_transition_probs()
	# Simulate node2vec walks
	walks = G.simulate_walks(args.num_walks, args.walk_length)
	learn_embeddings(walks)

if __name__ == "__main__":
	args = parse_args()
	main(args)
