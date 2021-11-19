import math
import networkx as nx
import matplotlib.pyplot as plt



all_words = []
neighbors = {}
neighbors_count = {}

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
        
        if total_neighbors_count > 3:
            w = common_neighbors_count / total_neighbors_count
            if w >= 0.25:
                print(word1, word2, w)
                graph.add_edge(word1, word2, weight=w)

# print(nx.clustering(graph,weight="weight"))
# subax1 = plt.subplot(121)



pos=nx.spring_layout(graph)
nx.draw(graph,pos, with_labels=True)
labels = nx.get_edge_attributes(graph,'weight')
nx.draw_networkx_edge_labels(graph,pos,edge_labels=labels)

# print(graph)
# nx.draw(graph, with_labels=True, label="weight", font_weight='bold')
plt.show()
# print(graph)