import pandas as pd
import Node
import Edge

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

    def addEdge(self, type, source, target, weight, directed):
        if source in self.__edges and target in self.__edges:
            self.__edges[source].append(Edge(type, source, target, weight, directed))
        
        if(not directed):
            if target in self.__edges and source in self.__edges:
                self.__edges[target].append(Edge(type, target, source, weight, directed))

    def createFromJson(self, json):
        for node in json['nodes']:
            self.addNode(node['id'], node['value'], node['label'], node['data'], node['type'], node['radius'], node['coordenates'])

        for edge in json['edges']:
            self.addEdge(edge['type'], edge['source'], edge['target'], edge['weight'], edge['directed'])

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

            for edge in self.__edges[node.getId()]:
                aux2 = {}

                aux2['type'] = edge.getType()
                aux2['from'] = edge.getSource()
                aux2['to'] = edge.getTarget()
                aux2['weight'] = edge.getWeight()

                linked_to.append(aux2)

            aux['linked_to'] = linked_to
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
                adj_matrix.loc[src, dest] = weight
                
                if not edge.get('directed', False):
                    adj_matrix.loc[dest, src] = weight
        
        return adj_matrix