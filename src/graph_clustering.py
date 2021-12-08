import math
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.cluster import clustering
import common
import gensim
from gensim.models import Word2Vec
from common import cleanUpWord

# create a graph that relates each word with how similar to other words each word in
# use graph clustering to create groups of similar words.
# a word is similar to another if they share neighbors and have the same suffix
# search for clusters that contain a known word and assume that the words in the cluster share the same probabilities 
# note running this is very slow

suffixes = ['ee', 'eer', 'er', 'ion', 'ism', 'ity', 'ment', 'ness', 'or', 'sion', 'ship', 'th', 'able', 'ible', 'al', 'ant', 'ary', 'ful', 'ic', 'ious', 'ous', 'ive', 'less', 'y', 'ed', 'en', 'er', 'ing', 'ize','ise', 'ly', 'ward', 'wise', 's', 'es']

#take in unlabeled ids and existing probabilities and extend the probabilites
def augment_emission_probs_with_custom_clustering(unlabeled_ids, emission_probs_file, graph_outfile, probs_outfile):
    unlabeled_ids.seek(0)
    emission_probs_file.seek(0)
    # graph_outfile.seek(0)
    probs_outfile.seek(0)
    print("augmenting emissions probs with custom clustering. Note this may take a while")
    all_words = []
    neighbors = {}
    neighbors_count = {}

    # calculate neighbors from the data
    # with open('data/unlabeled_ids.txt', newline='') as file:
    for line in unlabeled_ids:
        words = line.split()
        words = [cleanUpWord(w) for w in words]
        for (index, word) in enumerate(words):
            if word not in all_words:
                all_words.append(word)
            
            if word not in neighbors:
                neighbors[word] = {}
                neighbors_count[word] = 0

            if not index-1 < 0:
                neighbor_back = words[index-1]
                if neighbor_back not in neighbors[word]:
                    neighbors[word][neighbor_back] = 0
                neighbors[word][neighbor_back] += 1
                neighbors_count[word] += 1 
            
            if not index+2 > len(words):
                neighbor_front = words[index+1]
                if neighbor_front not in neighbors[word]:
                    neighbors[word][neighbor_front] = 0
                neighbors[word][neighbor_front] += 1
                neighbors_count[word] += 1

    # generate graph and weights from the counts of neighbors
    #weight = #number of common neighbors / #total number of neighbors
    graph = nx.Graph()
    for (index,word1) in enumerate(all_words):
        for word2 in all_words:
            if word1 == word2:
                continue
            if graph.has_edge(word1, word2):
                continue
            
            common_neighbors_count = len(set(neighbors[word1]).intersection(set(neighbors[word2])))
            total_neighbors_count = len(set(neighbors[word1]).union(set(neighbors[word2])))

            if total_neighbors_count > 0:
                w = common_neighbors_count / total_neighbors_count
                for suffix in suffixes:
                    if word1.endswith(suffix) and word2.endswith(suffix):
                        w = w * 2
                        break

                if total_neighbors_count > 4:
                    if w >= 0.25:
                        # print(word1, word2, w)
                        graph.add_edge(word1, word2, weight=w)

    probs = {}

    #load in existing probabilities
    # with open('model/emissionProbs.txt', 'r') as infile:
    for line in emission_probs_file:
        # pos, tag, prob
        
        pos = line.split(': ')[0].split(' ')[0]
        word = line.split(': ')[0].split(' ')[1]
        prob = float(line.split(': ')[1])

        if word not in probs:
            probs[word] = {}
        probs[word][pos] = prob

    # print("got probs from model")
    #use chinese whispers to cluster
    from chinese_whispers import chinese_whispers, aggregate_clusters
    chinese_whispers(graph, weighting='top', iterations=100)
    # print("got clusters")
    #save the graph to mess around with later
    nx.write_gexf(graph, graph_outfile)
    # print("saved graph")

    #for each cluster
    for label, cluster in sorted(aggregate_clusters(graph).items(), key=lambda e: len(e[1]), reverse=True):
        #for each word in cluster
        for matchword in cluster:
            # if we have the probabilites for the word we can apply the probabilities to all the other words in the cluster
            if matchword in probs:
                for outword in cluster:
                    if outword not in probs:
                        probs[outword]=probs[matchword]
                continue

    #output the probs
    for word in probs:
        for pos in probs[word]:
            probs_outfile.write(f"{pos} {word}: {probs[word][pos]}")