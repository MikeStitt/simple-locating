#!/usr/bin/python
#
# See https://github.com/MikeStitt/simple-locating/blob/master/license.txt for license.

import math
import cv2
import numpy as np
import scipy as Sci
import scipy.linalg

pi = math.pi

debug_label = ''
debug_pos_err = ''
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

MID_LEFT_HOOP_EAST = -27.38
MID_RIGHT_HOOP_EAST = +27.38

TARGET_LEFT_DELTA = -12.0
TARGET_RIGHT_DELTA = +12.0
TARGET_TOP_DELTA = +20.0
TARGET_BOTTOM_DELTA = +2.0

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


#BEGIN:debug code to create a test image and a rotation...
#                                 x(+is E)    y(+ is Up) z(+ is N)
test_locs = {
#	      'test1' : np.array([10.,0.,0.]),
#	      'test2' : np.array([0.,10.,0.]),
#	      'test3' : np.array([0.,0.,10.]),
#	      'test4' : np.array([0.,0.,0.]),
#	      'test5' : np.array([10.,10.,20.0]),
#	      'test6' : np.array([47.8751,0.,0.]),
#	      'test7' : np.array([0.,47.8751,0.]),
#	      'test8' : np.array([-47.8751,0.,0.])}
	      'ml-ul' : np.array([-27.38-12.0,         61.0+20.0, 0]),
	      'ml-ll' : np.array([-27.38-12.0,         61.0+ 2.0, 0]),
	      'ml-ur' : np.array([-27.38+12.0,         61.0+20.0, 0]),
	      'ml-lr' : np.array([-27.38+12.0,         61.0+ 2.0, 0]),
	      'mr-ul' : np.array([+27.38-12.0,         61.0+20.0, 0]),
	      'mr-ll' : np.array([+27.38-12.0,         61.0+ 2.0, 0]),
	      'mr-ur' : np.array([+27.38+12.0,         61.0+20.0, 0]),
	      'mr-lr' : np.array([+27.38+12.0,         61.0+ 2.0, 0]),
	      'bt-ul' : np.array([      -12.0,         28.0+20.0, 0]),
	      'bt-ll' : np.array([      -12.0,         28.0+ 2.0, 0]),
	      'bt-ur' : np.array([      +12.0,         28.0+20.0, 0]),
	      'bt-lr' : np.array([      +12.0,         28.0+ 2.0, 0]),
	      'tp-ul' : np.array([      -12.0,         98.0+20.0, 0]),
	      'tp-ll' : np.array([      -12.0,         98.0+ 2.0, 0]),
	      'tp-ur' : np.array([      +12.0,         98.0+20.0, 0]),
	      'tp-lr' : np.array([      +12.0,         98.0+ 2.0, 0]) }

#
# See http://en.wikipedia.org/wiki/Euler%E2%80%93Rodrigues_parameters
#
def rotation_matrix(axis,theta):
    axis = axis/np.sqrt(np.dot(axis,axis))
    a = np.cos(theta/2)
    b,c,d = -axis*np.sin(theta/2)
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])

v = np.array([3,5,4])
axis = np.array([0,0,1])
theta = pi

#focal length = d / ( 2 * tan ( angle_of_view / 2 ) )
fl = 320.0 / ( 2.0 * math.tan( math.radians(43.5)/2.0 ) )

cameraMatrix = np.array([ np.array([fl,      0, 160]),
			  np.array([0     , fl, 120]),
			  np.array([0     ,      0,   1]) ])

distCoeff = np.float64([0,0,0,0])

def get_sides( ul, ll, ur, lr ):
	# use ceil and floor to shorten boxes at partial pixels...
	#           left,                       right,                          top,                        bottom
#	return [ min(ul[0],ll[0]), max(ur[0],lr[0]), max(ul[1],ur[1]), min(ll[1],lr[1]) ] 
	return [ float(math.ceil(min(ul[0],ll[0]))), float(math.floor(max(ur[0],lr[0]))), float(math.floor(max(ul[1],ur[1]))), float(math.ceil(min(ll[1],lr[1]))) ] 

def construct_test_image( az_rot, pitch_rot, pos_x, pos_y, pos_z ):
	projected = {}
	rectangles = []
	y_axis = np.array([0,1,0])
	az_rot_matrix = rotation_matrix(y_axis,az_rot)
	x_axis = np.array([1,0,0])
	el_rot_matrix = rotation_matrix(x_axis,pitch_rot)

	sum_rot_matrix = np.dot(el_rot_matrix,az_rot_matrix)
	
	for k, a in test_locs.iteritems():
		p = cv2.projectPoints(np.array([a + [-pos_x,-pos_y,-pos_z]]), sum_rot_matrix, np.float64([0,0,0]), cameraMatrix, distCoeff)[0][0][0]
#		print debug_label, k, 'x=', p[0], 'y=', p[1]
		if ( 0 <= p[0] < 319 ) and ( 0 <= p[1] < 239 ):
			projected[k] = p

	if ('ml-ul' in projected) and ('ml-ll' in projected) and ('ml-ur' in projected) and ('ml-lr' in projected):
		rectangles.append( get_sides( projected['ml-ul'], projected['ml-ll'], projected['ml-ur'], projected['ml-lr'] ) )

	if ('mr-ul' in projected) and ('mr-ll' in projected) and ('mr-ur' in projected) and ('mr-lr' in projected):
		rectangles.append( get_sides( projected['mr-ul'], projected['mr-ll'], projected['mr-ur'], projected['mr-lr'] ) )

 	if ('bt-ul' in projected) and ('bt-ll' in projected) and ('bt-ur' in projected) and ('bt-lr' in projected):
		rectangles.append( get_sides( projected['bt-ul'], projected['bt-ll'], projected['bt-ur'], projected['bt-lr'] ) )

 	if ('tp-ul' in projected) and ('tp-ll' in projected) and ('tp-ur' in projected) and ('tp-lr' in projected):
		rectangles.append( get_sides( projected['tp-ul'], projected['tp-ll'], projected['tp-ur'], projected['tp-lr'] ) )

	return rectangles

#END:debug code to create 

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
	#print '{0:s} {1:d} {2:d}'.format(debug_found,leftmost, rightmost)

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

def test_cases():
	global debug_label
	global debug_pos_err

	rms_avg_a_err = 0.0
	rms_avg_r_err = 0.0
	rms_clc_a_err = 0.0
	rms_clc_r_err = 0.0
	cnt = 0

	for south in range (60, 241, 10):
		for x in range (-60, +61, 10):
			for t in (MID_LEFT, LOW, MID_RIGHT):
				east = x
				az = math.degrees(math.atan2( target_locs[t].center_east-east, south+15.0 ))
				debug_label = 'az={0:6.1f} e={1:6.1f} s={2:6.1f}'.format(az, east, south) 
#				print debug_label
#                             Rotate Right, Tilt Up, Shift Right, Shift Up, Shift Forward
				constructed_rectangles = construct_test_image( math.radians(float(az)), 0.0, float(east), 54.0, float(-south)  )
				targets = []				   
				for r in constructed_rectangles:
					targets.append( target( r[0], r[1], r[2], r[3] ) )
				calc_az, calc_east, calc_south = where( targets, 0 )

				avg_az, avg_east, avg_south = where( targets, 1 )

				if calc_south != -1000 :
					debug_pos_err = 'az-err={0:6.1f} e-err={1:6.1f} s-err={2:6.1f} az-avg={3:6.1f} e-avg={4:6.1f} s-avg={5:6.1f}'.format(az-math.degrees(calc_az), calc_east-east, calc_south - south, az-math.degrees(avg_az), avg_east-east, avg_south-south )
					for r in targets:
                                           if r.pos == t :
						actual_target_az, az_offset    = target_backboard_az_and_az_offset( r, east, south )
						calc_target_az, calc_az_offset = target_backboard_az_and_az_offset( r, calc_east, calc_south )
						avg_target_az, avg_az_offset = target_backboard_az_and_az_offset( r, avg_east, avg_south )
						actual_target_range = target_range( r, east, south)
						calc_target_range = target_range( r, calc_east, calc_south )
						avg_target_range = target_range( r, avg_east, avg_south )
						cnt = cnt + 1
						rms_clc_a_err = rms_clc_a_err + math.pow( calc_az_offset-az_offset,2 )
						rms_clc_r_err = rms_clc_a_err + math.pow( calc_target_range-actual_target_range,2 )
						rms_avg_a_err = rms_clc_a_err + math.pow( avg_az_offset-az_offset,2 )
						rms_avg_r_err = rms_clc_a_err + math.pow( avg_target_range-actual_target_range,2 )

						if math.fabs(calc_target_range-actual_target_range) >= math.fabs(avg_target_range-actual_target_range):
							better = '+'
						else:
							better = '.'
						print '{0:s} {1:s} {2:s} {3:s} az-err={4:6.1f} r-err={5:6.1f} az-avg-err={6:6.1f} r-avg-err={7:6.1f} {8:1s}'.format(debug_label, debug_pos_err, debug_found, target_name[r.pos], math.degrees(calc_az_offset-az_offset), calc_target_range - actual_target_range, math.degrees(avg_az_offset-az_offset), avg_target_range - actual_target_range, better )
				else:
					debug_pos_err = '---------------------------------------'
	print 'rms_clc_r_err={0:10.7f} rms_avg_r_err={1:10.7f} rms_clc_a_err={2:10.7f} rms_avg_a_err={3:10.7f}'.format( math.sqrt(rms_clc_r_err/cnt), math.sqrt(rms_avg_r_err/cnt), math.degrees(math.sqrt(rms_clc_a_err/cnt)), math.degrees(math.sqrt(rms_avg_a_err/cnt)) )

test_cases()
