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
from sklearn.manifold import TSNE

import node2vec
from gensim.models import Word2Vec, KeyedVectors
import matplotlib.pyplot as plt


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

    parser.add_argument('--input', nargs='?', default='moreno_lesmis/out.moreno_lesmis_lesmis',
                        help='Input graph path')

    parser.add_argument('--output', nargs='?', default='moreno_lesmis/lesmis.emb',
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
    parser.set_defaults(weighted=True)

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
    if args.weighted:  # If graph is weighted
        G = nx.read_edgelist(args.input, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
    else:  # If graph is unweighted
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
    model = Word2Vec(sentences=walks, vector_size=args.dimensions, window=args.window_size, min_count=0, sg=1,
                     workers=args.workers, epochs=args.iter)
    # Load the KeyedVectors of trained model
    KeyedVectors.load_word2vec_format(args.output)

    return model


def visualise_input_network(graph, m):
    """To visualise the input network

    Args:
        graph (Networkx graph): The graph to plot
        m (integer): Number of vertices in the graph
    """
    # positions for all nodes
    pos = nx.spring_layout(graph)
    # Set the parameters for displaying network
    nx.draw_networkx(graph, pos, label=None, node_size=400, node_color='#4b8bc8', font_size=8, font_color='k',
                     font_family='sans-serif', font_weight='normal', alpha=1, bbox=None, ax=None)
    # Draw the edges
    nx.draw_networkx_edges(graph, pos)
    # Draw the nodes
    nx.draw_networkx_nodes(graph, pos, nodelist=list(range(m, len(graph))), node_color='r', node_size=400, alpha=1)

    #Print degrees
    print("Node   Degree")
    for v in sorted(list(graph.nodes)):
        print(f"{v:4} {graph.degree(v):6}")

    # nx.draw_circular(graph, with_labels = True)

    # Display the drawing
    plt.show()


def visualise_embeddings(model):
    # Retrieve node embeddings and corresponding subjects
    # node_ids = model.wv.index2word  # list of node IDs
    # node_embeddings = (model.wv.vectors)

    def tsne_plot(model):
        "Creates and TSNE model and plots it"
        labels = []
        tokens = np.empty(shape=(0,0))

        for word in list(model.wv.index_to_key):
            print(np.array(model.wv[word]))
            np.append(tokens, np.array(model.wv[word]))
            labels.append(word)

        tsne_model = TSNE(perplexity=74,n_components=128, init='pca', n_iter=2500, random_state=23)
        new_values = tsne_model.fit_transform(tokens)

        x = []
        y = []
        for value in new_values:
            x.append(value[0])
            y.append(value[1])

        plt.figure(figsize=(16, 16))
        for i in range(len(x)):
            plt.scatter(x[i], y[i])
            plt.annotate(labels[i],
                         xy=(x[i], y[i]),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')
        plt.show()

    tsne_plot(model)


# numpy.ndarray of size number of nodes times embeddings dimensionality
# node_targets = node_subjects[[int(node_id) for node_id in node_ids]]

def main(args):
    """Pipeline for representational learning for all nodes in a graph.

    Args:
        args (parse_args): Arguments for the embeddings as defined in parse_args function
    """
    # Convert network data to a graph
    nx_G = read_graph()
    # Visualise the input network
    visualise_input_network(nx_G, len(nx_G))
    # Call node2vec
    G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
    # Compute transition probabilities
    G.preprocess_transition_probs()
    # Simulate node2vec walks
    walks = G.simulate_walks(args.num_walks, args.walk_length)
    model = learn_embeddings(walks)
    try:
        visualise_embeddings(model)
    except Exception as e:
        print(e)
        print(e.with_traceback())


if __name__ == "__main__":
    args = parse_args()
    main(args)
