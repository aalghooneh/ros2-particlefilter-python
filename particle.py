
from mapUtilities import *
from utilities import *
from numpy import cos, sin
import numpy as np

class particle:

    def __init__(self, pose, weight):
        self.pose=pose
        self.weight=weight

    def setWeight(self, weight):
        self.weight = weight

    def getWeight(self):
        return self.weight

    def setPose(self, pose):
        self.pose = pose

    def getPose(self):
        return self.pose[0], self.pose[1], self.pose[2]

    def motion_model(self, v,w):
        self.pose[0]=v * cos(self.pose[2]) + self.pose[0]
        self.pose[1]=v * sin(self.pose[2]) + self.pose[1]
        self.pose[2]=w * 0.1 + self.pose[2]


    def calculateParticleWeight(self, scanOutput: LaserScan, mapManipulatorInstance: mapManipulator):
        
        T = self.__poseToTranslationMatrix()

        _, scanCartesianHomo = convertScanToCartesian(scanOutput)
        scanInMap = np.dot(T, scanCartesianHomo)
        
        likelihoodField = mapManipulatorInstance.getLikelihoodField()
        
        origin = mapManipulatorInstance.getOrigin()
        res    = mapManipulatorInstance.getResolution()
        cellPositions = position_2_cell(scanInMap[:,0:2], origin, res)

        weight = np.prod(likelihoodField[np.ix_(cellPositions[:,0], cellPositions[:,1])].reshape(-1,))

        
        self.setWeight(weight)

        return weight

    def __poseToTranslationMatrix(self):
        x, y, th = self.getPose()
        translation = np.array([[cos(th), -sin(th),x],
                                [sin(th), cos(th),y],
                                [0,0,1]])

        return translation