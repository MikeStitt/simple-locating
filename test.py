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

	rms_avg_a_err = 0.0
	rms_avg_r_err = 0.0
	rms_clc_a_err = 0.0
	rms_clc_r_err = 0.0
	cnt = 0

	for south in range (60, 241, 10):
		for x in range (-60, +61, 10):
			for t in (where.MID_LEFT, where.LOW, where.MID_RIGHT):
				east = x
				az = math.degrees(math.atan2( where.target_locs[t].center_east-east, south+15.0 ))
				debug_label = 'az={0:6.1f} e={1:6.1f} s={2:6.1f}'.format(az, east, south) 
#                                                                              Rotate Right, Tilt Up, Shift Right, Shift Up, Shift Forward
				constructed_rectangles = construct_test_image( math.radians(float(az)), 0.0, float(east), 54.0, float(-south)  )
				targets = []				   
				for r in constructed_rectangles:
					targets.append( where.target( r[0], r[1], r[2], r[3] ) )
				calc_az, calc_east, calc_south = where.where( targets )

				avg_az, avg_east, avg_south = where.where( targets )

				if calc_south != -1000 :
					debug_pos_err = 'az-err={0:6.1f} e-err={1:6.1f} s-err={2:6.1f} az-avg={3:6.1f} e-avg={4:6.1f} s-avg={5:6.1f}'.format(az-math.degrees(calc_az), calc_east-east, calc_south - south, az-math.degrees(avg_az), avg_east-east, avg_south-south )
					for r in targets:
                                           if r.pos == t :
						actual_target_az, az_offset    = where.target_backboard_az_and_az_offset( r, east, south )
						calc_target_az, calc_az_offset = where.target_backboard_az_and_az_offset( r, calc_east, calc_south )
						avg_target_az, avg_az_offset   = where.target_backboard_az_and_az_offset( r, avg_east, avg_south )
						actual_target_range            = where.target_range( r, east, south)
						calc_target_range              = where.target_range( r, calc_east, calc_south )
						avg_target_range               = where.target_range( r, avg_east, avg_south )
						cnt = cnt + 1
						rms_clc_a_err = rms_clc_a_err + math.pow( calc_az_offset-az_offset,2 )
						rms_clc_r_err = rms_clc_a_err + math.pow( calc_target_range-actual_target_range,2 )
						rms_avg_a_err = rms_clc_a_err + math.pow( avg_az_offset-az_offset,2 )
						rms_avg_r_err = rms_clc_a_err + math.pow( avg_target_range-actual_target_range,2 )

						if math.fabs(calc_target_range-actual_target_range) >= math.fabs(avg_target_range-actual_target_range):
							better = '+'
						else:
							better = '.'
						print '{0:s} {1:s} {2:s} {3:s} az-err={4:6.1f} r-err={5:6.1f} az-avg-err={6:6.1f} r-avg-err={7:6.1f} {8:1s}'.format(debug_label, debug_pos_err, where.debug_found, target_name[r.pos], math.degrees(calc_az_offset-az_offset), calc_target_range - actual_target_range, math.degrees(avg_az_offset-az_offset), avg_target_range - actual_target_range, better )
				else:
					debug_pos_err = '---------------------------------------'
	print 'rms_clc_r_err={0:10.7f} rms_avg_r_err={1:10.7f} rms_clc_a_err={2:10.7f} rms_avg_a_err={3:10.7f}'.format( math.sqrt(rms_clc_r_err/cnt), math.sqrt(rms_avg_r_err/cnt), math.degrees(math.sqrt(rms_clc_a_err/cnt)), math.degrees(math.sqrt(rms_avg_a_err/cnt)) )

test_cases()
