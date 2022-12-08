# node2vec

This repository provides builds further on the reference implementation of *node2vec* as described in the [paper](https://arxiv.org/abs/1607.00653):<br>
> node2vec: Scalable Feature Learning for Networks.<br>
> Aditya Grover and Jure Leskovec.<br>
> Knowledge Discovery and Data Mining, 2016.<br>
> <Insert paper link>

# Authors

| Name | Github ID |
| --- | ----------- |
| A Sudarshan |[ASudu](https://github.com/ASudu) |
| Rahul Balike | [RahulBalike](https://github.com/RahulBalike)|
| Manpreet Singh Ahluwalia | [Manpreet-2002](https://github.com/Manpreet-2002)|

The *node2vec* algorithm learns continuous representations for nodes in any (un)directed, (un)weighted graph. Please check the [project page](https://snap.stanford.edu/node2vec/) for more details. 

### Basic Usage

#### Example
We run on two datasets:
To run *node2vec* on Zachary's [karate club](http://konect.cc/networks/ucidata-zachary/) network, execute the following command from the project home directory:<br/>
	``python src/karate_club.py --input graph/karate.edgelist --output emb/karate.emd``

To run *node2vec* on Victor Hugo's novel ['Les Misérables'](http://konect.cc/networks/moreno_lesmis/) network, execute the following command from the project home directory:<br/>
	``python src/les_miserables.py --input moreno_lesmis/out.moreno_lesmis_lesmis --output moreno_lesmis/lesmis.emb``

#### Options
You can check out the other options available to use with *node2vec* using:<br/>
	``python src/main.py --help``

#### Input
The supported input format is an edgelist:

	node1_id_int node2_id_int <weight_float, optional>
		
The graph is assumed to be undirected and unweighted by default. These options can be changed by setting the appropriate flags.

#### Output
The output file has *n+1* lines for a graph with *n* vertices. 
The first line has the following format:

	num_of_nodes dim_of_representation

The next *n* lines are as follows:
	
	node_id dim1 dim2 ... dimd

where dim1, ... , dimd is the *d*-dimensional representation learned by *node2vec*.

#### Visulaizations
For each network, we first display the input network and then output the final embeddings given by *node2vec* after applying *tSNE*.
