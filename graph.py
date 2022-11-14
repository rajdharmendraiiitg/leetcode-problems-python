from platform import node
import queue
import networkx as nx
#import matplotlib.pyplot as plt
import sys
import heapq

def visualize(graph):
    G = nx.Graph()
    G.add_edges_from(graph)
    nx.draw_networkx(G)
    plt.show()

def bfs(graph,visied,root):
    queue = [root]

    while queue:
        vertax = queue.pop(0)
        visied.add(vertax)
        for i in graph[vertax]:
            if i not in visied:
                queue.append(i)
    print(visied)
def dfs(graph,visited,root):
    if root not in visited:
        print(root)
        visited.add(root)
        for node in graph[root]:
            dfs(graph,visited,node)

## Dijkstra Shortest Path Algorithm
def dijkstra(graph,src,dest,visited):
    inf = sys.maxsize
    node_data = {'A':{'cost':inf,'pred':[]},
                'B':{'cost':inf,'pred':[]},
                'C':{'cost':inf,'pred':[]},
                'D':{'cost':inf,'pred':[]},
                'E':{'cost':inf,'pred':[]},
                'F':{'cost':inf,'pred':[]}           
    }
    node_data[src]['cost'] = 0
    temp = src
    #min_heap = []
    for i in range(len(node_data)):
        min_heap = []
        if temp in visited:
            visited.add(temp)
            #min_heap = []
            for neighbor in graph[temp]:
                if neighbor not in visited:
                    cost = node_data[temp]['cost'] + graph[temp][neighbor]
                    if cost < node_data[neighbor]['cost']:
                        node_data[neighbor]['cost'] = cost
                        node_data[neighbor]['pred'] = node_data[temp]['pred'] + list(temp)
                    #print(node_data[neighbor]['cost'],neighbor)
                    heapq.heappush(min_heap,(node_data[neighbor]['cost'],neighbor))
        heapq.heapify(min_heap)
        print(min_heap)
        temp = min_heap[0][1]
    print(node_data[dest]['cost'])
    print(node_data[dest]['pred']+list(dest))

## Network Delay Time - Dijkstra's algorithm
def networkDelayTime():
    
if __name__ == "__main__":
    visited = set()
    #graph = {0: [1, 2], 1: [2,3,0], 2: [3,1,0], 3: [1, 2]}
    '''graph= {
        'A':['C','E'],
        'B':['C','E','D'],
        'C':['D','A','B'],
        'D':['E'],
        'E':['A']
    }'''
    graph = {
        'A':{'B':2,'C':4},
        'B':{'A':2,'C':3,'D':8},
        'C':{'A':4,'B':3,'E':5,'D':2},
        'D':{'B':8,'C':2,'E':11,'F':22},
        'E':{'C':5,'D':11,'F':1},
        'F':{'D':11,'E':1}
    }
    source = 'A'
    destination = 'F'
    #bfs(graph,visited,0)
    #dfs(graph,visited,'A')
    #dijkstra(graph,source,destination,visited)
    #visualize(graph)