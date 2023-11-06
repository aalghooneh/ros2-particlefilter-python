
from mapUtilities import *
from utilities import *
from numpy import cos, sin
import numpy as np

class particle:

    def __init__(self, pose, weight):
        self.pose=pose
        self.weight=weight
        self.dt = 0.2



    def motion_model(self, dx,dy, dth, v, w):
        self.pose[0]=v * np.cos(self.pose[2]) * self.dt + self.pose[0]
        self.pose[1]=v * np.sin(self.pose[2]) * self.dt + self.pose[1]
        self.pose[2]=dth + self.pose[2]


    def calculateParticleWeight(self, scanOutput: LaserScan, mapManipulatorInstance: mapManipulator):
        
        T = self.__poseToTranslationMatrix()

        _, scanCartesianHomo = convertScanToCartesian(scanOutput)
        scanInMap = np.dot(T, scanCartesianHomo.T).T
        
        likelihoodField = mapManipulatorInstance.getLikelihoodField()
        
        origin = mapManipulatorInstance.getOrigin()
        res    = mapManipulatorInstance.getResolution()
        cellPositions = position_2_cell(scanInMap[:,0:2], origin, res)

        lm_x, lm_y = likelihoodField.shape

        cellPositions = cellPositions[
            np.logical_and(cellPositions[:,0] < lm_x , cellPositions[:,1] < lm_y), :
        ]

        weight = np.prod(likelihoodField[np.ix_(cellPositions[:,0], cellPositions[:,1])].reshape(-1,))
        weight+=0.001
        

        plt.plot(-mapManipulatorInstance.occ_points[:,0], mapManipulatorInstance.occ_points[:,1],'.')
        plt.plot(self.getPose()[0], self.getPose()[1], '*')
        plt.plot(scanInMap[:,0], scanInMap[:,1],'.')
        #plt.show()
        self.setWeight(weight)

        return weight
    
    def setWeight(self, weight):
        self.weight = weight

    def getWeight(self):
        return self.weight

    def setPose(self, pose):
        self.pose = pose

    def getPose(self):
        return self.pose[0], self.pose[1], self.pose[2]
    

    def __poseToTranslationMatrix(self):
        x, y, th = self.getPose()
        translation = np.array([[cos(th), -sin(th),x],
                                [sin(th), cos(th),y],
                                [0,0,1]])

        return translation