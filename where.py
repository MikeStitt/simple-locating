#!/usr/bin/python
#
# See https://github.com/MikeStitt/simple-locating/blob/master/license.txt for license.

import math
import cv2
import numpy as np
import scipy as Sci
import scipy.linalg

pi = math.pi

debug_found = ''

#
# Convert from degrees to radians
#
def deg2rad( d ):
	return 2.0 * pi * d / 360.0

#
# Camera x (width) Field of View
#
camera_x_fov_deg = 43.5                         # degrees
camera_x_fov_rad = deg2rad( camera_x_fov_deg )  # radians

#
# Camera pixels in width (x) and height (y)
#
camera_x_pixels = 320.0                         # pixels
camera_y_pixels = 240.0                         # pixels

#
#  Camera focal length in pixels.
#  See http://en.wikipedia.org/wiki/Angle_of_view
#
# Use angle = 2 atan( d/(2f))
#
# This equation assumes d and angle is centered in the field of view.
# 
# Solve angle = 2 atan( d/(2f ) for f:
# f = d / ( 2*tan(angle/2))
#
camera_focal_len = camera_x_pixels / ( 2.0 * math.tan( camera_x_fov_rad / 2.0 ))   # pixels

#
# Use the 1/2 angle form of angle = 2 atan( d/(2f))
#  angle = atan( d/f )
#
# This equation assumes d and angle is from the center of the field of view.
#
def pixel2rad( pixel ):
	return math.atan( pixel / camera_focal_len )  # + is up or right
#
# Camera height above ground
#
camera_height = 52.0 # inches

camera_initial_pitch_error = 0.0  # radians, + is pitched up

#
#  Optical target dimensions
#
#target_width = 24.0     # inches
target_height = 18.0    # inches

#
# Order the constants so that left is least, low and top are in the middle, and right is highest
#
MID_LEFT = 0
LOW = 1
UNKNOWN = 2
MID_UNKNOWN = 3
TOP = 4
MID_RIGHT = 5

# Step 0d
#
# define a class to hold a table of where the targets are on the field
#
# Definition of field coordinate system.
#
# assumes we are shooting at the blue alliance target on the left edge of
# the field see:  http://frc-manual.usfirst.org/viewItem/55#2.1
#
# south increasing as we move to right, towards blue alliance station, 0 inches is at the blue backboards
# east increases as me move up towards, the red kinect station, 0 inches is at center of the top and bottom hoops
# up increases as we move off the ground, 0 inches is at the ground
#
# a heading of 0 is facing due north towards the targets, + radians is towards the right (east), -radians is 
# towards the left (wast) 
#

class target_position:
	def __init__(self,l,r,t,b,h):
		self.left_inches = l             # east coordinate of left edge
		self.right_inches = r            # east coordinate of right edge
		self.top_inches = t              # up coordinate of top edge
		self.bottom_inches = b           # up coordinate of bottom edge
		self.hoop_height = h             # up coordinate of hoop
		self.center_east = (l+r)/2.0     # east coordinate center
		self.center_up = (t+b)/2.0       # up coordinate

#
# Height of hoop above ground
#

LOW_HOOP_UP = 28.0   # inches
MID_HOOP_UP = 61.0   # inches
TOP_HOOP_UP = 98.0   # inches

#
# Center of middle hoop
#
MID_LEFT_HOOP_EAST  = -27.38  # inches
MID_RIGHT_HOOP_EAST = +27.38  # inches

#
# Target edges from center of hoop
#
TARGET_LEFT_DELTA   = -12.0 #inches
TARGET_RIGHT_DELTA  = +12.0 #inches
TARGET_TOP_DELTA    = +20.0 #inches
TARGET_BOTTOM_DELTA =  +2.0 #inches

#define a dictionary look up table for the targe locations

target_locs = { LOW:         target_position( 0.0+TARGET_LEFT_DELTA,                  # l = left edge  
				              0.0+TARGET_RIGHT_DELTA,                 # r = right edge 
				              LOW_HOOP_UP+TARGET_TOP_DELTA,           # t = top edge   
				              LOW_HOOP_UP+TARGET_BOTTOM_DELTA,        # b = bottom edge
				              LOW_HOOP_UP),                           # h = hoop height

		# Default an unknown middle level hoop to be the left hoop
		MID_UNKNOWN: target_position( MID_LEFT_HOOP_EAST+TARGET_LEFT_DELTA,   # l = left edge     
				              MID_LEFT_HOOP_EAST+TARGET_RIGHT_DELTA,  # r = right edge 
				              MID_HOOP_UP+TARGET_TOP_DELTA,	      # t = top edge   
				              MID_HOOP_UP+TARGET_BOTTOM_DELTA,	      # b = bottom edge
				              MID_HOOP_UP),			      # h = hoop height

		MID_LEFT:    target_position( MID_LEFT_HOOP_EAST+TARGET_LEFT_DELTA,   # l = left edge     
				              MID_LEFT_HOOP_EAST+TARGET_RIGHT_DELTA,  # r = right edge 
				              MID_HOOP_UP+TARGET_TOP_DELTA,	      # t = top edge   
				              MID_HOOP_UP+TARGET_BOTTOM_DELTA,	      # b = bottom edge
				              MID_HOOP_UP),			      # h = hoop height

		MID_RIGHT:   target_position( MID_RIGHT_HOOP_EAST+TARGET_LEFT_DELTA,  # l = left edge      
				              MID_RIGHT_HOOP_EAST+TARGET_RIGHT_DELTA, # r = right edge 
				              MID_HOOP_UP+TARGET_TOP_DELTA,	      # t = top edge   
				              MID_HOOP_UP+TARGET_BOTTOM_DELTA,	      # b = bottom edge
				              MID_HOOP_UP),			      # h = hoop height

		TOP:         target_position( 0.0+TARGET_LEFT_DELTA,                  # l = left edge  
				              0.0+TARGET_RIGHT_DELTA,		      # r = right edge 
				              TOP_HOOP_UP+TARGET_TOP_DELTA,	      # t = top edge   
				              TOP_HOOP_UP+TARGET_BOTTOM_DELTA,	      # b = bottom edge
				              TOP_HOOP_UP) }			      # h = hoop height

# state variables

camera_pitch_error = camera_initial_pitch_error

class target:
#
# Step 1:
# When we find a target record where we found the edges in pixels:
#
	def __init__(self,l,r,t,b):
		self.left_pixels = l
		self.right_pixels = r
		self.top_pixels = t
		self.bottom_pixels = b
		self.pos = UNKNOWN
#
# Step 2:
# Convert the pixel locations to angles from the center line of the camera:
#
	def est_initial_angles(self):
		self.left_rad   = pixel2rad( self.left_pixels     - camera_x_pixels / 2.0 )
		self.right_rad  = pixel2rad( self.right_pixels    - camera_x_pixels / 2.0 )
		self.top_rad    = pixel2rad( self.top_pixels      - camera_y_pixels / 2.0 )
		self.bottom_rad = pixel2rad( self.bottom_pixels   - camera_y_pixels / 2.0 )
#
# Step 3:
# Azimuth is left to right angle from the center line of the camera. +angles are to the right.
# Elevation is down to up angle from the center line of the camera. +angles are up.
#
# Estimate the Azimuth and Elevation from the camera to the center of the target.
#
		self.azimuth_rad   = (self.left_rad + self.right_rad) / 2.0          # + is right
		self.elevation_rad = (self.top_rad  + self.bottom_rad) / 2.0 - camera_pitch_error # + is up

#
# Step 4:
# Initial estimate of the distance to this target based upon the vertical degrees this target takes in the 
# field of view.
# 
#                         c
#  top 1
#      |\
#      | \
#      |  2 bottom
#      |/  
#      3 ^ 
#        pitch_error
#
#     angle 123 is 90 deg
#     seg 13 is target height, and plumb
#     seg 12 is apparent target parallel to image plane
#     point 2 is apparent position of bottom in image plane
#     seg12 = apparent_height = target_height * cos( camera_pitch_error )
#
		self.dist_est_1 = target_height * math.cos( camera_pitch_error ) / ( math.tan(self.top_rad) - math.tan(self.bottom_rad) )

#
# Step 5:
# Initial estimate of the height of the center of this target above ground based upon the 
# distance to the target, the angle of the target, and the camera height
#

		self.height_est_1 = self.dist_est_1 * math.tan(self.elevation_rad) + camera_height
#
# Step 6:
# Classify the target as a low, middle or top target based upon it's height above ground.
#
		if ( self.height_est_1 < 56.0 ):
			self.level = LOW
		elif ( self.height_est_1 < 90.5 ):
			self.level = MID_UNKNOWN
		else:
			self.level = TOP
#
# Step 8:
# Given the minimum azimuth (most left target), and maximum azimuth (most right target)
# if we have identified more than one target, classify the middle level targets as the 
# left or the right middle.
#
	def classify_pos( self, min_az, max_az ):
		if self.level == MID_UNKNOWN:
			if min_az == max_az:
				self.pos = MID_UNKNOWN
			elif self.azimuth_rad == min_az:
				self.pos = MID_LEFT
			elif self.azimuth_rad == max_az:
				self.pos = MID_RIGHT
			else:
				self.pos = MID_UNKNOWN # should not reach this line, becaue if we
				                       # found a mid and another target, the mid
				                       # should be min_az or max_az
		else:
			self.pos = self.level


#Step 11
# Given 3 camera angles to 3 veritcal lines along the wall of backboards, estimate the
# camera heading (azimuth), east position, and south position
#
# See https://github.com/MikeStitt/simple-locating-docs/blob/master/mathToFindLocationFromAnglesTo3PointsOnALine.pdf?raw=true
#
#
def estimate_pos_3_sep_hrz_angles( left_rad, mid_rad, right_rad, left_pos, mid_pos, right_pos ):
	a0 = mid_rad - left_rad
	a1 = right_rad - mid_rad

	b0 = mid_pos - left_pos
	b1 = right_pos - mid_pos

 	A = math.atan2( -b0 , ( (b1/math.tan(a1)) - (b1+b0)/math.tan(a1+a0)))

	ak = pi/2-A
	alpha_k = pi/2-ak
	alpha_1 = pi/2-ak-a1
	alpha_2 = pi/2-ak-a1-a0

	k = b1 * math.tan(alpha_1) / (math.tan(alpha_k)-math.tan(alpha_1))

	d = k * math.tan(alpha_k)

	#      (           azimuth,        east, south )
	return (-(right_rad+ak-pi), right_pos+k,     d )

#
#  Given a list of found rectangles,
#  invoke steps 2 through 12 on the rectangles
#
def where( rectangles ):
	global debug_found

# Invoke steps 2 through 6 for each target found
#
	for r in rectangles:
		r.est_initial_angles()

# Step 7
# Find the center target azimuth that is most left and most right
#
	min_azimuth = +pi   # start at +180 which is out of view to right
	max_azimuth = -pi   # start at -180 which is out of view to left

	for r in rectangles:
		min_azimuth = min( min_azimuth, r.azimuth_rad )
		max_azimuth = max( max_azimuth, r.azimuth_rad )

# Invoke step 8 for each target
#
	for r in rectangles:
		r.classify_pos( min_azimuth, max_azimuth )
 
# For debugging purposes identify the rectangles we found
#
	ml = '--'
	mr = '--'
	bt = '--'
	tp = '--'
	mu = '--'
	leftmost = MID_RIGHT+1
	rightmost = MID_LEFT-1

	for r in rectangles:
		if r.pos == LOW:
			bt = 'BT'
		elif r.pos == MID_UNKNOWN:
			mu = 'MU'
		elif r.pos == MID_LEFT:
			ml = 'ML'
		elif r.pos == MID_RIGHT:
			mr = 'MR'
		elif r.pos == TOP:
			tp = 'TP'

	debug_found = '{0:s}{1:s}{2:s}{3:s}{4:s}'.format( ml, bt, tp, mr, mu )


# Step 9
# Identify the left most and right most target.

	for r in rectangles:
		if r.pos < leftmost :
			leftmost = r.pos
			left = r

		if r.pos > rightmost :
			rightmost = r.pos
			right=r
# Step 10
# If we have found two different targets, and they are not just the top and bottom targes.
# Then perform step 11 on two sets of 3 angles angles two the 3 targets.
#
	if (leftmost != MID_RIGHT+1) and (leftmost != rightmost) and not((leftmost==LOW) and (rightmost==TOP)) :
		# take two estimates of position with 3 angles

		# Perform step 11 using the using both vertical edges of the far left target
		# and the right edge of the far right target.
		#
		# See definition of field coordinate system.
		# az1 and az2 are estimates of the camera heading in azimuth in radians
		# east1 and east2 are estimates of the camera east position
		# south1 and south2 are estimates of the camera south position
		#
		az1, east1, south1 = estimate_pos_3_sep_hrz_angles( left.left_rad,
								    left.right_rad,
								    right.right_rad,
								    target_locs[left.pos].left_inches,
								    target_locs[left.pos].right_inches,
								    target_locs[right.pos].right_inches)

		# Perform step 11 using the using the left vertical edge of the far left target
		# and the both vertical edges of the far right target.
		az2, east2, south2 = estimate_pos_3_sep_hrz_angles( left.left_rad,
								    right.left_rad,
								    right.right_rad,
								    target_locs[left.pos].left_inches,
								    target_locs[right.pos].left_inches,
								    target_locs[right.pos].right_inches)
		#
		# Step 12.
		# Average the two passes of passes for an estimate of the camera position and heading
		#
		return ( (az1+az2)/2.0, (east1+east2)/2.0, (south1+south2)/2.0 )
	else:
		return( -1000*pi, -1000, -1000 )

#
# Step 13.
# For a target rectangle calculate the azimuth offset from the center of the
# center of the backboard to the center of the hoop.
#

def target_backboard_az_and_az_offset( target, east, south ):
	#
	# Hoop Center is 15 inches south of center of the backboard.
	#
	target_east = target_locs[target.pos].center_east
	target_south = -15.0                                     # inches

	backboard_center_az = math.atan2(target_east-east,-south)
	target_center_az = math.atan2(target_east-east,target_south-south)

	az_offset = target_center_az - backboard_center_az

	return backboard_center_az, az_offset

#
# Step 14.
# For a target rectangle calculate the range along the floor from the center
# of the camera to the center of the hoop.
#
def target_range( target, east, south ):
	target_east = target_locs[target.pos].center_east
	target_south = -15.0

	return math.sqrt( math.pow(target_east-east,2) + math.pow(target_south-south,2) )
