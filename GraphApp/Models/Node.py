import Edge

class Node:
    def __init__(self, id, value, label, data, type, radius, coordenates):
        self.__id = id
        self.__value = value
        self.__label = label
        self.__data = data
        self.__type = type
        self.__radius = radius
        self.__coordenates = coordenates

    # Getters
    def getId(self):
        return self.__id
        
    def getValue(self):
        return self.__value
    
    def getLabel(self):
        return self.__label
    
    def getData(self):
        return self.__data
    
    def getType(self):
        return self.__type
    
    def getRadius(self):
        return self.__radius
    
    def getCoordenates(self):
        return self.__coordenates
    
    # Setters
    def setId(self, id):
        self.__id = id

    def setValue(self, value):
        self.__value = value

    def setLabel(self, label):
        self.__label = label

    def setData(self, data):
        self.__data = data

    def setType(self, type):
        self.__type = type

    def setRadius(self, radius):
        self.__radius = radius

    def setCoordenates(self, coordenates):
        self.__coordenates = coordenates