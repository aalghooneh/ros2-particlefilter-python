

from particle import particle
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from mapUtilities import mapManipulator
import message_filters
import numpy as np
from utilities import *

from rclpy.duration import Duration

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

class particleFilter(Node):

    def __init__(self, mapFilename="/home/dastan/final/maps/room.yaml", numParticles=100):
        
        super().__init__("particleFiltering")


        qos_profile_odom=QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE, depth=10)
        qos_profile_laserScanner = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                              durability=DurabilityPolicy.VOLATILE,
                                              depth=10)


        self.particleMarkerArrayPublisher =\
                self.create_publisher(MarkerArray, "/particles/markers", 100)
        

        self.odomSub = message_filters.Subscriber(self, Odometry, "/odom", qos_profile=qos_profile_laserScanner)
        self.laserScanSub = message_filters.Subscriber(self, LaserScan, "/scan", qos_profile=qos_profile_laserScanner)

        self.timeSynchronizer = message_filters.ApproximateTimeSynchronizer([self.odomSub, self.laserScanSub], queue_size=10, slop=0.1)
        self.timeSynchronizer.registerCallback(self.filterCallback)


        self.mapUtilities=mapManipulator(mapFilename)
        self.mapUtilities.make_likelihood_field()


        self.historyOdom = []
        self.particles=[]


        width = -3
        height = -1



        self.particles = [particle(particle_, 1/numParticles) for particle_ in\
                           np.random.uniform(low=[0, -2, -np.pi], high=[2.0, 0.0, np.pi], size=(numParticles, 3))]
        

        self.weights = [1/numParticles] * numParticles
    
    def visualizeParticles(self):
        
        particles=MarkerArray()
        
        for i, particle_ in enumerate(self.particles):
            
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()

            marker.id = i
            marker.ns = "particles"
            marker.lifetime = Duration(seconds=0.5).to_msg()

            marker.type = marker.ARROW
            marker.action = marker.ADD

            if particle_.getWeight() <= 0.25:
                weight = 0.25

            marker.scale.x=weight
            marker.scale.y = 0.1; marker.scale.z = 0.1

            marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
            marker.pose.position.x = particle_.getPose()[0]
            marker.pose.position.y = particle_.getPose()[1]
            marker.pose.orientation.w = np.cos(particle_.getPose()[2]/2)
            marker.pose.orientation.z = np.sin(particle_.getPose()[2]/2)


            particles.markers.append(marker)

        self.particleMarkerArrayPublisher.publish(particles)

    def normalizeWeights(self):
        
        sumWeight = np.sum(self.weights)
        self.weights = self.weights / sumWeight


        for particle in self.particles:
            particle.setWeight( particle.getWeight()/sumWeight)


    def resample(self):
        numParticles = len(self.particles)
        indices = np.random.choice(range(numParticles),  size=100, p=self.weights)

        self.particles = [self.particles[i] for i in indices]

    def filterCallback(self, odomMsg: Odometry, laserMsg: LaserScan):
        import time
        start_timer = time.time()


        self.historyOdom.append(odomMsg)

        if len(self.historyOdom) > 2:
            self.historyOdom.pop(0)

        else:
            return

        dx, dy, dth = calculate_displacement(self.historyOdom[0], self.historyOdom[1])
        w = odomMsg.twist.twist.angular.z
        v = odomMsg.twist.twist.linear.x

        for i, particle_ in enumerate(self.particles):
            particle_.motion_model(dx, dy, dth, v, w)
            particle_.calculateParticleWeight(laserMsg, self.mapUtilities)
            #print(f"particle at {particle_.getPose()} has weight of {particle_.getWeight()}")
            try:
                self.weights[i] = particle_.getWeight()
            except IndexError:
                print(f"tried {i} while {len(self.weights)}, and {len(self.particles)}")

        

        self.normalizeWeights()
        self.visualizeParticles()


        #self.resample()
        
        
        
        end_time = time.time()
        #print(f"the time took for the filter callback is {end_time-start_timer}")

import rclpy

if __name__=="__main__":
    
    rclpy.init()

    pf = particleFilter()

    rclpy.spin(pf)    
