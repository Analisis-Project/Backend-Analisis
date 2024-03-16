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
            