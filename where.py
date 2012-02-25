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

def deg2rad( d ):
	return 2.0 * pi * d / 360.0

camera_x_fov_deg = 43.5

camera_x_fov_rad = deg2rad( camera_x_fov_deg )

camera_x_pixels = 320.0
camera_y_pixels = 240.0

camera_focal_len = camera_x_pixels / ( 2.0 * math.tan( camera_x_fov_rad / 2.0 ))

camera_height = 52.0 # inches

camera_initial_pitch_error = 0.0  # radians, + is pitched up

target_width = 24
target_height = 18

#
# Order the constants so that left is least, low and top are in the middle, and right is highest
#
MID_LEFT = 0
LOW = 1
UNKNOWN = 2
MID_UNKNOWN = 3
TOP = 4
MID_RIGHT = 5

target_name = { UNKNOWN: 'UN', LOW: 'BT', MID_UNKNOWN: 'MU', MID_LEFT: 'ML', MID_RIGHT: 'MR', TOP: 'TP' } 

# define a class to hold a table of where the targets are on the field
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
	def __init__(self,l,r,t,b,h,s):
		self.left_inches = l
		self.right_inches = r
		self.top_inches = t
		self.bottom_inches = b
		self.hoop_height = h
		self.center_east = (l+r)/2.0
		self.center_up = (t+b)/2.0
		self.south = s

LOW_HOOP_UP = 28.0
MID_HOOP_UP = 61.0
TOP_HOOP_UP = 98.0

MID_LEFT_HOOP_EAST  = -27.38
MID_RIGHT_HOOP_EAST = +27.38

TARGET_LEFT_DELTA   = -12.0
TARGET_RIGHT_DELTA  = +12.0
TARGET_TOP_DELTA    = +20.0
TARGET_BOTTOM_DELTA =  +2.0

#define a dictionary look up table for the targe locations

target_locs = { LOW:         target_position( 0.0+TARGET_LEFT_DELTA,
				              0.0+TARGET_RIGHT_DELTA,
				              LOW_HOOP_UP+TARGET_TOP_DELTA,
				              LOW_HOOP_UP+TARGET_BOTTOM_DELTA,
				              LOW_HOOP_UP,
				              0.0),
		#TBD should we assume a unknown mid hoop is the left hoop????
		MID_UNKNOWN: target_position( MID_LEFT_HOOP_EAST+TARGET_LEFT_DELTA,      
				              MID_LEFT_HOOP_EAST+TARGET_RIGHT_DELTA,
				              MID_HOOP_UP+TARGET_TOP_DELTA,
				              MID_HOOP_UP+TARGET_BOTTOM_DELTA,
				              MID_HOOP_UP,
				              0.0 ),
		MID_LEFT:    target_position( MID_LEFT_HOOP_EAST+TARGET_LEFT_DELTA,      
				              MID_LEFT_HOOP_EAST+TARGET_RIGHT_DELTA,
				              MID_HOOP_UP+TARGET_TOP_DELTA,
				              MID_HOOP_UP+TARGET_BOTTOM_DELTA,
				              MID_HOOP_UP,
				              0.0 ),
		MID_RIGHT:   target_position( MID_RIGHT_HOOP_EAST+TARGET_LEFT_DELTA,      
				              MID_RIGHT_HOOP_EAST+TARGET_RIGHT_DELTA,
				              MID_HOOP_UP+TARGET_TOP_DELTA,
				              MID_HOOP_UP+TARGET_BOTTOM_DELTA,
				              MID_HOOP_UP,
				              0.0 ),
		TOP:         target_position( 0.0+TARGET_LEFT_DELTA,
				              0.0+TARGET_RIGHT_DELTA,
				              TOP_HOOP_UP+TARGET_TOP_DELTA,
				              TOP_HOOP_UP+TARGET_BOTTOM_DELTA,
				              TOP_HOOP_UP,
				              0.0 ) }
# state variables

camera_pitch_error = camera_initial_pitch_error


def pixel2rad( pixel ):
	return math.atan( pixel / camera_focal_len )  # + is up or right


class target:
	def __init__(self,l,r,t,b):
		self.left_pixels = l
		self.right_pixels = r
		self.top_pixels = t
		self.bottom_pixels = b
		self.pos = UNKNOWN

	def est_initial_angles(self):
		self.left_rad   = pixel2rad( self.left_pixels     - camera_x_pixels / 2.0 )
		self.right_rad  = pixel2rad( self.right_pixels    - camera_x_pixels / 2.0 )
		self.top_rad    = pixel2rad( self.top_pixels      - camera_y_pixels / 2.0 )
		self.bottom_rad = pixel2rad( self.bottom_pixels   - camera_y_pixels / 2.0 )

		self.azimuth_rad   = (self.left_rad + self.right_rad) / 2.0          # + is right
		self.elevation_rad = (self.top_rad  + self.bottom_rad) / 2.0 - camera_pitch_error # + is up

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

		self.height_est_1 = self.dist_est_1 * math.tan(self.elevation_rad) + camera_height

		if ( self.height_est_1 < 56.0 ):
			self.level = LOW
		elif ( self.height_est_1 < 90.5 ):
			self.level = MID_UNKNOWN
		else:
			self.level = TOP

	def classify_pos( self, min_az, max_az ):
		if self.level == MID_UNKNOWN:
			if min_az == max_az:
				self.pos = MID_UNKNOWN
			elif self.azimuth_rad == min_az:
				self.pos = MID_LEFT
			elif self.azimuth_rad == max_az:
				self.pos = MID_RIGHT
			else:
				self.pos = MID_UNKOWN  # should not reach this line, becaue if we
				                       # found a mid and another target, the mid
				                       # should be min_az or max_az
		else:
			self.pos = self.level


def estimate_pos_3_sep_hrz_angles( left_rad, mid_rad, right_rad, left_pos, mid_pos, right_pos ):
#
# See https://github.com/MikeStitt/simple-locating-docs/blob/master/mathToFindLocationFromAnglesTo3PointsOnALine.pdf?raw=true
#
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

	return (-(right_rad+ak-pi), right_pos+k,    d )

def where( rectangles, method ):
	global debug_found

	min_azimuth = +pi   # start at +180 which is out of view to right
	max_azimuth = -pi   # start at -180 which is out of view to left

	for r in rectangles:
		r.est_initial_angles()
		min_azimuth = min( min_azimuth, r.azimuth_rad )
		max_azimuth = max( max_azimuth, r.azimuth_rad )

	for r in rectangles:
		r.classify_pos( min_azimuth, max_azimuth )
 
	ml = '--'
	mr = '--'
	bt = '--'
	tp = '--'
	mu = '--'
	leftmost = MID_RIGHT+1
	rightmost = MID_LEFT-1

	for index,r in enumerate(rectangles):
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

		if r.pos < leftmost :
			leftmost = r.pos
			left = r

		if r.pos > rightmost :
			rightmost = r.pos
			right=r

	debug_found = '{0:s}{1:s}{2:s}{3:s}{4:s}'.format( ml, bt, tp, mr, mu )

	if (leftmost != MID_RIGHT+1) and (leftmost != rightmost) and not((leftmost==LOW) and (rightmost==TOP)) :
		# take two estimates of position with 3 angles
		az1, east1, south1 = estimate_pos_3_sep_hrz_angles( left.left_rad,
								    left.right_rad,
								    right.right_rad,
								    target_locs[leftmost].left_inches,
								    target_locs[leftmost].right_inches,
								    target_locs[rightmost].right_inches)

		az2, east2, south2 = estimate_pos_3_sep_hrz_angles( left.left_rad,
								    right.left_rad,
								    right.right_rad,
								    target_locs[leftmost].left_inches,
								    target_locs[rightmost].left_inches,
								    target_locs[rightmost].right_inches)
		return ( (az1+az2)/2.0, (east1+east2)/2.0, (south1+south2)/2.0 )
	else:
		return( -1000*pi, -1000, -1000 )

def target_backboard_az_and_az_offset( target, east, south ):
	target_east = target_locs[target.pos].center_east
	target_south = -15.0

	backboard_center_az = math.atan2(target_east-east,-south)
	target_center_az = math.atan2(target_east-east,target_south-south)

	az_offset = target_center_az - backboard_center_az

	return backboard_center_az, az_offset

def target_range( target, east, south ):
	target_east = target_locs[target.pos].center_east
	target_south = -15.0

	return math.sqrt( math.pow(target_east-east,2) + math.pow(target_south-south,2) )
