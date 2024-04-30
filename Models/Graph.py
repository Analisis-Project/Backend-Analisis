import pandas as pd
import numpy as np
from Models.Node import Node
from Models.Edge import Edge

class Graph:
    def __init__(self):
        self.__nodes = []
        self.__edges = {}

    # Getters
    def getNodes(self):
        return self.__nodes
    
    def getEdges(self):
        return self.__edges
    
    # Setters
    def setNodes(self, nodes):
        self.__nodes = nodes

    def setEdges(self, edges):
        self.__edges = edges

    # Methods
    def addNode(self, id, value, label, data, type, radius, coordenates):
        self.__nodes.append(Node(id, value, label, data, type, radius, coordenates))
        self.__edges[str(id)] = []

    def addEdge(self, type, source, target, weight, directed = True):
        if self.nodeExists(source) and self.nodeExists(target):
            self.__edges[str(source)].append(Edge(type, source, target, weight, directed))
        
        if(not directed):
            if target in self.__edges and source in self.__edges:
                self.__edges[str(target)].append(Edge(type, target, source, weight, directed))
    
    def nodeExists(self, id):
        for node in self.__nodes:
            if node.getId() == id:
                return True
        return False

    def createFromJson(self, json):
        for node in json['nodes']:
            self.addNode(node['id'], node['value'], node['label'], node['data'], node['type'], node['radius'], node['coordenates'])

        for edge in json['edges']:
            self.addEdge(edge['type'], edge['from'], edge['to'], edge['weight'])

    def randomGraph(self, num_nodes, complete = False, conex = True, pondered = False, directed = False):
        for i in range(1, num_nodes + 1):
            id = i
            value = np.random.randint(1, 100)
            label = str(i)
            data = {}
            type = ''
            radius = np.random.randint(1, 3)
            coordenates = {}
            coordenates['x'] = np.random.randint(1, 100)
            coordenates['y'] = np.random.randint(1, 100)

            self.addNode(id, value, label, data, type, radius, coordenates)

        if complete:
            for i in range(1, num_nodes):
                for j in range(i+1, num_nodes + 1):
                    if pondered:
                        weight = np.random.randint(1, 100)
                    else:
                        weight = 1

                    self.addEdge('', i, j, weight, directed)
        else:
            if conex:
                for i in range(1, num_nodes + 1):
                    if pondered:
                        weight = np.random.randint(1, 100)
                    else:
                        weight = 1
                    
                    to = np.random.randint(1, num_nodes + 1)

                    while to == i:
                        to = np.random.randint(1, num_nodes + 1)

                    self.addEdge('', i, to, weight, directed)
            else:
                num_splits = 0
                if num_nodes < 3:
                    return
                elif 3 <= num_nodes <= 5:
                    num_splits = np.random.randint(2, num_nodes)
                else:
                    num_splits = np.random.randint(2, int(num_nodes/2) + 1)

                numbers = np.arange(1, num_nodes + 1)
                np.random.shuffle(numbers)

                groups = np.array_split(numbers, num_splits)

                for group in groups:
                    if len(group) > 1:
                        for item in group:
                            if pondered:
                                weight = np.random.randint(1, 100)
                            else:
                                weight = 1

                            to = np.random.choice(group)

                            while to == item:
                                to = np.random.choice(group)

                            self.addEdge('', np.int32(item).item(), np.int32(to).item(), weight, directed)
            
    def toJson(self):
        json = {}
        data = []

        for node in self.__nodes:
            aux = {}

            aux['id'] = node.getId()
            aux['value'] = node.getValue()
            aux['label'] = node.getLabel()
            aux['data'] = node.getData()
            aux['type'] = node.getType()

            linked_to = []

            for edge in self.__edges[str(node.getId())]:
                aux2 = {}

                aux2['type'] = edge.getType()
                aux2['from'] = edge.getSource()
                aux2['nodeId'] = edge.getTarget()
                aux2['weight'] = edge.getWeight()

                linked_to.append(aux2)

            aux['linkedTo'] = linked_to
            aux['radius'] = node.getRadius()
            aux['coordenates'] = node.getCoordenates()

            data.append(aux)


        json["graph"] = [{"name": "G", "data": data,}]

        return json
            
    def matrix(self):
        nodes = self.__nodes
        edges = self.__edges
        
        nodes_labels = [node.getLabel() for node in nodes]
        adj_matrix = pd.DataFrame(0, index=nodes_labels, columns=nodes_labels)
        
        for src, edge_list in edges.items():
            for edge in edge_list:
                dest = edge.getTarget()
                weight = edge.getWeight()
                adj_matrix.loc[str(dest), str(src)] = weight
                
                if not edge.getDirected():
                    adj_matrix.loc[str(src), str(dest)] = weight
        return adj_matrix