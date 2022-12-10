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
from sklearn.cluster import KMeans
import argparse
import numpy as np
import networkx as nx
from sklearn.manifold import TSNE
import node2vec
from gensim.models import Word2Vec
import matplotlib.pyplot as plt


def parse_args():
    """
    Parses the node2vec arguments:
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
    # parser.add_argument('--input', nargs='?', default='moreno_lesmis/out1.moreno_lesmis_lesmis',
    #                     help='Input graph path')

    parser.add_argument('--output', nargs='?', default='moreno_lesmis/lesmis.emb',
                        help='Embeddings path')
    # parser.add_argument('--output', nargs='?', default='moreno_lesmis/lesmis1.emb',
    #                     help='Embeddings path')

    parser.add_argument('--dimensions', type=int, default=16,
                        help='Number of dimensions. Default is 16.')

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

    parser.add_argument('--q', type=float, default=0.5,
                        help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')

    parser.add_argument('--unweighted', dest='unweighted', action='store_false')

    parser.set_defaults(weighted=False)
    # parser.set_defaults(weighted=True)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='c.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)

    return parser.parse_args()


def read_graph():
    """
    Reads the input network in networkx.

    Returns:
        G: Networkx graph having information of the input network
    """
    # By default create a directed graph (DiGraph) by reading the edge list
    if args.weighted:  # If graph is weighted
        G = nx.read_edgelist(args.input, nodetype=int, data=(('weight', float),), create_using=nx.DiGraph())
        print("The input graph is weighted")
    else:  # If graph is unweighted
        G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
        # Assign edge weights to 1 (since unweighted)
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1
        print("The input graph is unweighted")

    # If network is undirected, we convert G to undirected graph
    if not args.directed:
        G = G.to_undirected()
        print("The input graph is undirected")
    else:
        print("The input graph is directed")

    return G


def learn_embeddings(walks):
    """
    Learn embeddings by optimizing the Skipgram objective using SGD.

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
    # Save the trained model
    model.wv.save_word2vec_format(args.output)
    
    return model


def visualise_input_network(graph, m):
    """
    To visualise the input network

    Args:
        graph (Networkx graph): The graph to plot
        m (integer): Number of vertices in the graph
    """
    # # positions for all nodes
    # pos = nx.spring_layout(graph)
    # # Set the parameters for displaying network
    # nx.draw_networkx(graph, pos, label=None, node_size=400, node_color='#4b8bc8', font_size=8, font_color='k',
    #                  font_family='sans-serif', font_weight='normal', alpha=1, bbox=None, ax=None)
    # # Draw the edges
    # nx.draw_networkx_edges(graph, pos)
    # # Draw the nodes
    # nx.draw_networkx_nodes(graph, pos, nodelist=list(range(m, len(graph))), node_color='r', node_size=400, alpha=1)

    #Print degrees
    print("Node   Degree")
    for v in sorted(list(graph.nodes)):
        print(f"{v:4} {graph.degree(v):6}")

    nx.draw_circular(graph, with_labels=True)

    # Display the drawing
    plt.show()

def visualise_output_embeddings(model):
    """
    Visualizes the trained model using the TSNE function
    For more info: 
    Outputs a graph that visualized the embeddings present in model

    Args:
        model (gensim): Trained model ready to be visualized
    """
    # Initializations
    labels = []
    tokens_list = []

    # Tokens are the vectors (coordinates) of the trained model
    for word in list(model.wv.index_to_key):
        # Word is the node label and model.wv[word] is the coordinates of the node
        # in the final embedding
        tokens_list.append(np.array(model.wv[word]))
        # Collecting the node labels
        labels.append(word)

    tokens = np.array(tokens_list)
    tsne_model = TSNE(n_components=2)
    new_values = tsne_model.fit_transform(tokens)

    # Split data into x and y for plotting
    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
    
    # K-means clustering
    data = list(zip(x, y))
    inertias = []
    colors = ['r','g','b','k','m']

    # Elbow method to determine number of clusters
    for i in range(len(new_values)):
        kmeans = KMeans(n_clusters=i+1)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    plt.plot(range(1,len(new_values)+1), inertias, marker='o')
    plt.title('Elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()

    kmeans = KMeans(n_clusters=5)
    kmeans.fit(data)

    # Setting up for plotting
    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i],c=colors[kmeans.labels_[i]])
        plt.annotate(labels[i],
                        xy=(x[i], y[i]),
                        xytext=(5, 2),
                        textcoords='offset points',
                        ha='right',
                        va='bottom')
    
    # Display the plot
    plt.show()

def main(args):
    """
    Pipeline for representational learning for all nodes in a graph.

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
        visualise_output_embeddings(model)
    except Exception as e:
        print(e)
        print(e.with_traceback())


if __name__ == "__main__":
    args = parse_args()
    main(args)