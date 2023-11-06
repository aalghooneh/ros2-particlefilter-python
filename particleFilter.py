

from particle import particle
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from mapUtilities import mapManipulator
import message_filters
import numpy as np

from rclpy.duration import Duration

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

class particleFilter(Node):

    def __init__(self, mapFilename="map.yaml", numParticles=1000):
        
        super().__init__("particleFiltering")


        qos_profile_odom=QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE, depth=10)
        qos_profile_laserScanner = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                              durability=DurabilityPolicy.VOLATILE,
                                              depth=10)


        self.particleMarkerArrayPublisher =\
                self.create_publisher(MarkerArray, "/particles/markers", 10)
        

        self.odomSub = message_filters.Subscriber(self, Odometry, "/odom", qos_profile=qos_profile_laserScanner)
        self.laserScanSub = message_filters.Subscriber(self, LaserScan, "/scan", qos_profile=qos_profile_laserScanner)

        self.timeSynchronizer = message_filters.ApproximateTimeSynchronizer([self.odomSub, self.laserScanSub], queue_size=10, slop=0.1)
        self.timeSynchronizer.registerCallback(self.filterCallback)


        #self.mapUtilities=mapManipulator(mapFilename)

        self.particles=[]


        width = 10
        height = 10



        self.particles = [particle(particle_, 1/numParticles) for particle_ in\
                           np.random.uniform(low=[0, 0, -np.pi], high=[width, height, np.pi], size=(numParticles, 3))]
        


    
    def visualizeParticles(self):
        
        particles=MarkerArray()
        
        for i, particle_ in enumerate(self.particles):
            
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()

            marker.id = i
            marker.ns = "particles"
            marker.lifetime = Duration(seconds=0.1).to_msg()

            marker.type = marker.ARROW
            marker.action = marker.ADD

            marker.scale.x=1.0; marker.scale.y = 0.1; marker.scale.z = 0.1

            marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
            marker.pose.position.x = particle_.getPose()[0]
            marker.pose.position.y = particle_.getPose()[1]
            marker.pose.orientation.w = np.cos(particle_.getPose()[2]/2)
            marker.pose.orientation.z = np.sin(particle_.getPose()[2]/2)


            particles.markers.append(marker)

        self.particleMarkerArrayPublisher.publish(particles)

    def filterCallback(self, odomMsg: Odometry, laserMsg: LaserScan):
        

        for particle_ in self.particles:
            
            particle_.motion_model(odomMsg.twist.twist.linear.x, 
                                   odomMsg.twist.twist.angular.z)
        

        self.visualizeParticles()

import rclpy

if __name__=="__main__":
    
    rclpy.init()

    pf = particleFilter()

    rclpy.spin(pf)    
