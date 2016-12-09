class Stump:
    def __init__(self, ind1, ind2, alpha):
        self.pixelIndex1 = ind1
        self.pixelIndex2 = ind2
        self.alpha = alpha

    def getStumpProperties(self):
        return self.pixelIndex1, self.pixelIndex2, self.alpha

