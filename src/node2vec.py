import numpy as np
import networkx as nx
import random

class Graph():
	def __init__(self, nx_G, is_directed, p, q):
		"""
		Initializer function for this class

		Args:
			nx_G (networkx): Networkx graph of the network
			is_directed (bool): Flag variable to indicate if graph is directed or not
			p (float): Return parameter - controls likelihood of immediately revisiting a node
			q (float): In-Out paramter - differentiates between inward and outward nodes
		"""
		self.G = nx_G
		self.is_directed = is_directed
		self.p = p
		self.q = q

	def node2vec_walk(self, walk_length, start_node):
		"""
		Simulate a random walk starting from start node.

		Args:
			walk_length (int): Length of the walk we wish to simulate
			start_node (networkx): The node from where we start the walk

		Returns:
			walk(List): Walk represented as a list of nodes traversed
		"""
		# Initializations
		G = self.G
		alias_nodes = self.alias_nodes # self.alias_nodes defined in preprocess_transition_probs function
		alias_edges = self.alias_edges # self.alias edges defined in preprocess_transition_probs function
		walk = [start_node] # Initialize the walk with the start node

		# The walk
		while len(walk) < walk_length:
			# current node indicated by the last element of the walk
			cur = walk[-1]
			# Get the neighbors of current node
			cur_nbrs = sorted(G.neighbors(cur))
			# If current node has atleast one neighbor
			if len(cur_nbrs) > 0:
				# If only one node has been traversed so far
				if len(walk) == 1:
					# We traverse the next node selected by alias sampling
					# We use first order random walk as there is no previous node as we are just starting
					walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])]) 
				else:
					# We get the previous node in the walk
					prev = walk[-2]
					# Find the next node to be traveresed using alias sampling via second order random walk
					next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0], 
						alias_edges[(prev, cur)][1])]
					# Update the walk
					walk.append(next)
			else:
				break

		return walk

	def simulate_walks(self, num_walks, walk_length):
		"""Repeatedly simulate random walks from each node.
		Wrapper function to get multiple node2vec walks.

		Args:
			num_walks (int): Number of node2vec walks we wish to simulate
			walk_length (int): Length of each node2vec walk

		Returns:
			walks(List): List containing num_walks node2vec walks
		"""
		# Initializations
		G = self.G
		walks = []
		nodes = list(G.nodes()) # Get the nodes of the network
		
		print('Walk iteration:')

		# Get the required number of walks
		for walk_iter in range(num_walks):
			print(str(walk_iter+1)  + '/' + str(num_walks))
			# To randomize the picking of nodes
			random.shuffle(nodes)
			# Get node2vec walks with each node of the network as a start node
			for node in nodes:
				# Get the node2vec walk and append it to the list of walks
				walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

		return walks

	def get_alias_edge(self, src, dst):
		"""Get the normalized transition probability after traversing edge src-dst.
		To use same notation as the paper:
		src here denotes 't' in the paper
		dst here denotes 'v' in the paper
		Unnormalized transition probability = search bias * weight(edge to be traversed)

		Args:
			src (nx): Source vertex of the edge (tail of edge if directed graphs)
			dst (nx): Destination vertex of the edge (head of edge if directed graphs)

		Returns:
			_type_: _description_
		"""
		# Initializations
		G = self.G
		p = self.p
		q = self.q
		unnormalized_probs = []

		# For each neighbor of the dst (v) (dst_nbr here denotes 'x' in the paper)
		for dst_nbr in sorted(G.neighbors(dst)):
			# If we are revisiting the src (t) (i.e., dtx = 0)
			if dst_nbr == src:
				# Search bias is 1/p
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/p)
			# If dtx = 1 ,i.e., if v and t share a common node 
			elif G.has_edge(dst_nbr, src):
				# Search bias = 1
				unnormalized_probs.append(G[dst][dst_nbr]['weight'])
			# If dtx = 2 ,i.e., the node only has common edge with v
			else:
				# Search bias = 1/q
				unnormalized_probs.append(G[dst][dst_nbr]['weight']/q)
		
		# Find the normalization constant to normalize this unnormalized transition probability
		norm_const = sum(unnormalized_probs)
		# Normalized probabilities obtained
		normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

		# Determining Alias probability distribution for edges
		return alias_setup(normalized_probs)

	def preprocess_transition_probs(self):
		"""
		Preprocessing of transition probabilities for guiding the random walks.
		After this, as the walk goes on, based on the transitions, the probabilities will be recomputed.
		"""
		# Initializations
		G = self.G
		is_directed = self.is_directed
		alias_nodes = {}
		alias_edges = {}
		# triads = {}

		# Generating alias distribution for first order random walk alias_node
		for v in G.nodes():
			# Unnormalized probability of traversing edge v-x = edge weight of edge v-x
			unnormalized_probs = [G[v][nbr]['weight'] for nbr in sorted(G.neighbors(v))]
			# Find the normalization constant
			norm_const = sum(unnormalized_probs)
			# Normalize the probabilities
			normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
			# Generating Alias probability distribution for the nodes
			alias_nodes[v] = alias_setup(normalized_probs)

		# Generating alias distribution for second order random walk alias_edges
		# For directed graphs ,i.e., only one way edges (may be both ways if defined in graph)
		if is_directed:
			for edge in G.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
		# For undirected graphs ,i.e., for dual connection between all edges
		else:
			for edge in G.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1]) 				# Between node 0 and node 1 in the edge
				alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0]) # Between node 1 and node 0 in the edge

		self.alias_nodes = alias_nodes
		self.alias_edges = alias_edges

		return


def alias_setup(probs):
	"""Compute utility lists for non-uniform sampling from discrete distributions.
	Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
	for details

	Args:
		probs (List): List of normalized transition probabilities

	Returns:
		_type_: _description_
	"""
	K = len(probs)
	q = np.zeros(K)	
	J = np.zeros(K, dtype=np.int)

	smaller = []
	larger = []
	for kk, prob in enumerate(probs):
		q[kk] = K*prob	# Multplying all probabilities by number of probabilities
		if q[kk] < 1.0:
			smaller.append(kk) 	# Probabilities lesser than 1/K in probs
		else:
			larger.append(kk)	# Probabilities greater than or equal to 1/K in probs

	while len(smaller) > 0 and len(larger) > 0:
		small = smaller.pop() 	# returns topmost index appended to the list
		large = larger.pop()	

		J[small] = large		# Creating a list with a relation from small probability to large probability
								# Is used in the case probability chosen during draw is 'small' and we need a larger one
								# So we link the small index with its corresponding larger relation

		q[large] = q[large] + q[small] - 1.0	# Modifying probabilities in array q

		if q[large] < 1.0:
			smaller.append(large)
		else:
			larger.append(large)

	return J, q

def alias_draw(J, q):
	"""Draw sample from a non-uniform discrete distribution using alias sampling.

	Args:
		J (numpy array): _description_
		q (numpy array): _description_

	Returns:
		_type_: _description_
	"""
	K = len(J)

	kk = int(np.floor(np.random.rand()*K)) 	# Randomly choosing an index value
	if np.random.rand() < q[kk]:			# If random value is lesser than probability stored in q
		return kk
	else:									# If random value is greater/equal them use small-large relation J
		return J[kk]