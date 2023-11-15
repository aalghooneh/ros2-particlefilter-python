

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
from geometry_msgs.msg import Point, PoseWithCovarianceStamped
from std_msgs.msg import ColorRGBA
from nav_msgs.msg import OccupancyGrid




class particleFilter(Node):

    def __init__(self, mapFilename="/home/dastan/final/maps/room.yaml", numParticles=500):
        
        super().__init__("particleFiltering")


        qos_profile_odom=QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE, depth=10)
        qos_profile_laserScanner = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                                              durability=DurabilityPolicy.VOLATILE,
                                              depth=10)


        self.particleMarkerArrayPublisher =\
                self.create_publisher(MarkerArray, "/particles/markers", 10)
        

        self.odomSub = message_filters.Subscriber(self, Odometry, "/odom",
                                                   qos_profile=qos_profile_laserScanner)
        self.laserScanSub = message_filters.Subscriber(self, LaserScan, "/scan",
                                                        qos_profile=qos_profile_laserScanner)


        self.initialPoseSubsriber = self.create_subscription(PoseWithCovarianceStamped, "/initialpose", self.initialPose2Dcallback ,10)
        self.timeSynchronizer = message_filters.ApproximateTimeSynchronizer([self.odomSub, self.laserScanSub], queue_size=10, slop=0.1)
        self.timeSynchronizer.registerCallback(self.filterCallback)

        self.publisher = self.create_publisher(OccupancyGrid, '/likelihood_map', 10)
        
        self.mapUtilities=mapManipulator(mapFilename, laser_sig=0.1)
        self.mapUtilities.make_likelihood_field()


        self.historyOdom = []
        self.particles=[]

        self.numParticles = numParticles
        self.std_particle_x = 0.5
        self.std_particle_y = 0.5

        self.br = TransformBroadcaster(self)

        print("give me your first estimate on rviz, 2D Pose Estimator")

        self.initialized = False
    
    def initializeParticleFilter(self, x, y, th):
        
        numParticles = self.numParticles
        self.particlePoses=np.random.uniform(low=[x-self.std_particle_x, y - self.std_particle_y, th - 0.3],
                                             high=[x+self.std_particle_x, y + self.std_particle_y, th + 0.3], size=(numParticles, 3))
        
        self.particles = [particle(particle_, 1/numParticles) for particle_ in\
                           self.particlePoses]
        

        self.weights = [1/numParticles] * numParticles

        self.initialized = True

    def initialPose2Dcallback(self, marker: PoseWithCovarianceStamped):
        
        x, y, th = marker.pose.pose.position.x,\
                   marker.pose.pose.position.y, euler_from_quaternion(marker.pose.pose.orientation)
        

        self.initializeParticleFilter(x,y,th)

    
    def visualizeParticles(self, particles_, stamp):
        
        particles=MarkerArray()
        
        for i, particle_ in enumerate(particles_):
            
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = stamp

            marker.id = i
            marker.ns = "particles"
            marker.lifetime = Duration(seconds=1.5).to_msg()

            marker.type = marker.ARROW
            marker.action = marker.ADD

            weight = particle_.getWeight()
            if weight <= 0.10:
                weight = 0.1

            marker.scale.x=weight
            marker.scale.y = 0.05; marker.scale.z = 0.05

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
        particles_sorted = sorted(self.particles.copy(), key=lambda particle: particle.getWeight(), reverse=True)

        
        return particles_sorted[:50]

    def filterCallback(self, odomMsg: Odometry, laserMsg: LaserScan):
        import time
        start_timer = time.time()


        if not self.initialized:
            return


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
            
            
            self.weights[i] = particle_.getWeight()



        weighted_average_translation = np.average(self.particlePoses[:, :2], axis=0, weights=self.weights)

        # For the rotation (theta), you might use a mean of circular quantities if the angles are small and don't wrap around
        mean_angle = np.arctan2(np.sum(np.sin(self.particlePoses[:, 2])*self.weights), np.sum(np.cos(self.particlePoses[:, 2])*self.weights))

        weighted_average_pose = np.append(weighted_average_translation, mean_angle)
        stamp = odomMsg.header.stamp
        publishTransform(self.br, weighted_average_pose[0], weighted_average_pose[1], weighted_average_pose[2], stamp)

        self.publisher.publish(self.mapUtilities.to_message())
        self.normalizeWeights()

        print(self.particles[np.argmax(self.weights)].getPose())
        self.visualizeParticles(self.resample(), stamp)


        #self.resample()
        
        
        
        end_time = time.time()
        #print(f"the time took for the filter callback is {end_time-start_timer}")

import rclpy

if __name__=="__main__":
    
    rclpy.init()

    pf = particleFilter()

    rclpy.spin(pf)    
