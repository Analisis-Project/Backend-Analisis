class Edge:
    def __init__(self, type, source, target, weight, directed):
        self.__type = type
        self.__source = source
        self.__target = target
        self.__weight = weight
        self.__directed = directed

    # Getters
    def getType(self):
        return self.__type

    def getSource(self):
        return self.__source
    
    def getTarget(self):
        return self.__target
    
    def getWeight(self):
        return self.__weight
    
    def getDirected(self):
        return self.__directed
    
    # Setters
    def setType(self, type):
        self.__type = type

    def setSource(self, source):
        self.__source = source

    def setTarget(self, target):
        self.__target = target

    def setWeight(self, weight):
        self.__weight = weight

    def setDirected(self, directed):
        self.__directed = directed

        