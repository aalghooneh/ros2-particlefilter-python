

from particle import particle
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import message_filters

class particleFilter(Node):

    def __init__(self, mapFilename, numParticles):
        
        super().__init__("particleFiltering")


        qos_profile_=QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE, depth=10)
        
        self.odomSub = message_filters.Subscriber(Odometry, "/odom", qos_profile=qos_profile_)
        self.laserScanSub = message_filters.Subscriber(LaserScan, "/scan", 10)

        self.timeSynchronizer = message_filters.ApproximateTimeSynchronizer([self.odomSub, self.laserScanSub], queue_size=10, slop=0.1)
        self.timeSynchronizer.registerCallback(self.filterCallback)

    
        self.particles=[]


    

    def filterCallback(self, odomMsg: Odometry, laserMsg: LaserScan):
        pass
