import math
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.cluster import clustering



all_words = []
neighbors = {}
neighbors_count = {}

suffixes = ['ee', 'eer', 'er', 'ion', 'ism', 'ity', 'ment', 'ness', 'or', 'sion', 'ship', 'th', 'able', 'ible', 'al', 'ant', 'ary', 'ful', 'ic', 'ious', 'ous', 'ive', 'less', 'y', 'ed', 'en', 'er', 'ing', 'ize','ise', 'ly', 'ward', 'wise']

# calculate neighbors from the data
with open('data/unlabeled_ids.txt', newline='') as file:
    for line in file:
        words = line.split()
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
            
print("got counts")


# generate graph and weights from the counts of neighbors
graph = nx.Graph()


for word1 in all_words:
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

print("got probs and created graph")
# pos=nx.spring_layout(graph)
# nx.draw(graph,pos, with_labels=True)
# labels = nx.get_edge_attributes(graph,'weight')
# nx.draw_networkx_edge_labels(graph,pos,edge_labels=labels)
# plt.show()

# load the emmision probs from disk into memory
# probs[word][pos] = prob
probs = {}

with open('model/emissionProbs.txt', 'r') as infile:
    for line in infile:
        # pos, tag, prob
        
        pos = line.split(': ')[0].split(' ')[0]
        word = line.split(': ')[0].split(' ')[1]
        prob = float(line.split(': ')[1])

        if word not in probs:
            probs[word] = {}
        probs[word][pos] = prob

    print("got probs from model")

    from chinese_whispers import chinese_whispers, aggregate_clusters
    chinese_whispers(graph, weighting='top', iterations=1000)
    print("got clusters")

with open('model/emissionProbs.txt', 'a') as outfile:
    for label, cluster in sorted(aggregate_clusters(graph).items(), key=lambda e: len(e[1]), reverse=True):
        print(label, cluster)
        for matchword in cluster:
            if matchword in probs:
                print(label, matchword)
                for outword in cluster:
                    if outword not in probs:
                        for tag in probs[matchword]:
                            outfile.write(f"{tag} {outword}: {probs[matchword][tag]}\n")
                        probs[outword]=probs[matchword]
                continue