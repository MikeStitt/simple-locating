#!/usr/bin/python
#
# See https://github.com/MikeStitt/simple-locating/blob/master/license.txt for license.

import math
import cv2
import numpy as np
import scipy as Sci
import scipy.linalg
import where

pi = math.pi

debug_label = ''
debug_pos_err = ''

target_name = { where.UNKNOWN: 'UN', where.LOW: 'BT', where.MID_UNKNOWN: 'MU', where.MID_LEFT: 'ML', where.MID_RIGHT: 'MR', where.TOP: 'TP' } 


#                                 x(+is E)    y(+ is Up) z(+ is N)
test_locs = {
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
	#
	return [ float(math.ceil (min(ul[0],ll[0]))),    # left 
		 float(math.floor(max(ur[0],lr[0]))),    # right
		 float(math.floor(max(ul[1],ur[1]))),    # top
		 float(math.ceil (min(ll[1],lr[1]))) ]   # bottom

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

def test_cases():
	global debug_label
	global debug_pos_err

	rms_clc_a_err = 0.0
	rms_clc_r_err = 0.0
	cnt = 0

	for south in range (60, 241, 10):
		for east in range (-60, +61, 10):
			for t in (where.MID_LEFT, where.LOW, where.MID_RIGHT):
				az = math.degrees(math.atan2( where.target_locs[t].center_east-east, south+15.0 ))
				debug_label = 'az={0:6.1f}(deg) e={1:6.1f}(in) s={2:6.1f}(in)'.format(az, east, south) 
				#
                                # Step 0g
                                #
				# Project the image on to the camera, identify the complete targets in the field of view
                                #
				constructed_rectangles = construct_test_image( math.radians(float(az)), # Rotate Right - (Azimuth)   - radians 
									       0.0,                     # Tilt Up      - (Elevation) - radians
									       float(east),             # Shift Right  - (East)      - inches
									       54.0,                    # Shift Up     - (Up)        - inches
									       float(-south)  )         # Shift Forward- (North)     - inches
				# Start with an empty list of targets
				targets = []			
				#
				# Perform Step 1 on all the target rectangles in the field of view
				#
				for r in constructed_rectangles:
					targets.append( where.target( r[0], r[1], r[2], r[3] ) )

				# Perform Steps 2 through 12 on the target set of rectangles in the field of view
				#
				calc_az, calc_east, calc_south = where.where( targets )

				# calc_south = -1000 if we did not find two targets in the field of view
				#
				# if we found at least 2 targets in the camera field of view
				if calc_south != -1000 :
					debug_pos_err = 'heading-err={0:6.1f}(deg) east-err={1:6.1f}(in) south-err={2:6.1f}(in)'.format(
						az-math.degrees(calc_az), calc_east-east, calc_south - south)
					#
					# Find the target we were aiming at in this test case
					for r in targets:
                                           if r.pos == t :
						   #
						   # Perform step 13
						   # Calculate the azimuth offset from the center of the backboard to the
						   # center of the hoop
						   calc_target_az, calc_az_offset = where.target_backboard_az_and_az_offset( 
							   r, calc_east, calc_south )
						   #
						   # Perform step 14
						   # Calculate the range from the camera to the center of the hoop
						   calc_target_range              = where.target_range( r, calc_east, calc_south )

						   #
						   # Calculate the actual (ideal) azimuth offset and range assuming
						   # that we had no errors calculating where we were at and calculating
						   # our heading
						   #
						   actual_target_az, az_offset    = where.target_backboard_az_and_az_offset( 
							   r, east, south )
						   actual_target_range            = where.target_range( r, east, south)

						   #
						   # Accumulate Root Mean Square (RMS) Heading and Range for this test run
						   #
						   cnt = cnt + 1						   
						   rms_clc_a_err = rms_clc_a_err + math.pow( calc_az_offset-az_offset,2 )
						   rms_clc_r_err = rms_clc_a_err + math.pow( calc_target_range-actual_target_range,2 )

						   print '{0:s} {1:s} in-view:{2:s} target:{3:s} az-err-to-hoop={4:4.1f}(deg) range-err-to-hoop={5:4.1f}(in)'.format(
							debug_label, debug_pos_err, where.debug_found, target_name[r.pos], 
							math.degrees(calc_az_offset-az_offset), calc_target_range - actual_target_range)
				else:
					debug_pos_err = '---------------------------------------'

	#
	# Print the RMS errors
	#
	print 'rms_clc_r_err={0:10.7f} rms_clc_a_err={1:10.7f}'.format( math.sqrt(rms_clc_r_err/cnt), math.degrees(math.sqrt(rms_clc_a_err/cnt)) ) 

#
# Run the test cases
#
test_cases()
