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

        f_csv = open( 'output.csv', 'w' )
        printedHeader = 0

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
				constructed_rectangles = construct_test_image( 
                                    math.radians(float(az)), # Rotate Right - (Azimuth)   - radians 
                                    0.0,                     # Tilt Up      - (Elevation) - radians
				    float(east),             # Shift Right  - (East)      - inches
				    54.0,                    # Shift Up     - (Up)        - inches
                                    float(-south)  )         # Shift Forward- (North)     - inches
                                #
				# Start with an empty list of targets
				targets = []			
                                # Start with a empty csv and header list
                                header = []
                                csv = []
                                header.append( "south (in)" )
                                csv.append( south )
                                header.append( "east (in)" )
                                csv.append( east )
                                header.append( "az  (deg)" )
                                csv.append( az )
 				#
				# Perform Step 1 on all the target rectangles in the field of view
				#
				for r in constructed_rectangles:
                                        #       edges:               left, right,  top, bottom     : in pixels
					targets.append( where.target( r[0], r[1], r[2],   r[3] ) )
                                        header.append( "left (pix)" )
                                        csv.append( r[0] )
                                        header.append( "rght (pix)" )
                                        csv.append( r[1] )
                                        header.append( "top  (pix)" )
                                        csv.append( r[2] )
                                        header.append( "bot  (pix)" )
                                        csv.append( r[3] )

                                for i in range( len(constructed_rectangles), 4 ):
                                        header.append( "left (pix)" )
                                        csv.append( -1 )
                                        header.append( "rght (pix)" )
                                        csv.append( -1 )
                                        header.append( "top  (pix)" )
                                        csv.append( -1 )
                                        header.append( "bot  (pix)" )
                                        csv.append( -1 )

				# Perform Steps 2 through 12 on the target set of rectangles in the field of view
				#
				calc_az, calc_east, calc_south = where.where( targets )

				for tgt in targets:
                                        header.append( "left (rad)" )
                                        csv.append( tgt.left_rad )
                                        header.append( "rght (rad)" )
                                        csv.append( tgt.right_rad )
                                        header.append( "top  (rad)" )
                                        csv.append( tgt.top_rad )
                                        header.append( "bot  (rad)" )
                                        csv.append( tgt.bottom_rad )
                                        header.append( "ctr-az-rad" )
                                        csv.append( tgt.azimuth_rad )
                                        header.append( "ctr-el-rad" )
                                        csv.append( tgt.elevation_rad )
                                        header.append( "d_est_1-in" )
                                        csv.append( tgt.dist_est_1 )
                                        header.append( "h_est_1-in" )
                                        csv.append( tgt.height_est_1 )
                                        header.append( "level     " )
                                        csv.append( target_name[tgt.level] )
                                        header.append( "position  " )
                                        csv.append( target_name[tgt.pos] )

                                for i in range( len(targets), 4 ):
                                        header.append( "left (rad)" )
                                        csv.append( "NaN" )
                                        header.append( "rght (rad)" )
                                        csv.append( "NaN" )
                                        header.append( "top  (rad)" )
                                        csv.append( "NaN" )
                                        header.append( "bot  (rad)" )
                                        csv.append( "NaN" )
                                        header.append( "ctr-az-rad" )
                                        csv.append( "NaN" )
                                        header.append( "ctr-el-rad" )
                                        csv.append( "NaN" )
                                        header.append( "d_est_1-in" )
                                        csv.append( "NaN" )
                                        header.append( "h_est_1-in" )
                                        csv.append( "NaN" )
                                        header.append( "level     " )
                                        csv.append( target_name[where.UNKNOWN] )
                                        header.append( "position  " )
                                        csv.append( target_name[where.UNKNOWN] )

                                header.append( "leftmost  " )
                                csv.append( where.leftmost )
                                header.append( "rightmost " )
                                csv.append( where.rightmost )
                                header.append( "south1    " )
                                csv.append( where.south1 )
                                header.append( "east1     " )
                                csv.append( where.east1 )
                                header.append( "az1       " )
                                csv.append( where.az1 )
                                header.append( "south2    " )
                                csv.append( where.south2 )
                                header.append( "east2" )
                                csv.append( where.east2 )
                                header.append( "az2       " )
                                csv.append( where.az2 )
                                header.append( "south     " )
                                csv.append( calc_south )
                                header.append( "east      " )
                                csv.append( calc_east )
                                header.append( "az        " )
                                csv.append( calc_az )


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
                                                   header.append( "az-off-rad" )
                                                   csv.append( calc_az_offset )

						   #
						   # Perform step 14
						   # Calculate the range from the camera to the center of the hoop
						   calc_target_range              = where.target_range( r, calc_east, calc_south )
                                                   header.append( "range (in)" )
                                                   csv.append( calc_target_range )

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
                                                   if printedHeader == 0:
                                                       comma = ''
                                                       for i in header:
                                                           f_csv.write( '{0:s}{1:>15s}'.format( comma, i ) )
                                                           comma = ','
                                                       printedHeader = 1
                                                       f_csv.write( "\n" )
                                                   comma = ''
                                                   for i in csv:
                                                       if type(i) == str:
                                                           f_csv.write( '{0:s}{1:>15s}'.format( comma, i ) )
                                                       if type(i) == int:
                                                           f_csv.write( '{0:s}{1:>15d}'.format( comma, i ) )
                                                       if type(i) == float:
                                                           f_csv.write( '{0:s}{1:>15f}'.format( comma, i ) )
                                                       comma = ','
                                                   f_csv.write( "\n" )

				else:
					debug_pos_err = '---------------------------------------'

	#
	# Print the RMS errors
	#
	print 'rms_clc_r_err={0:10.7f} rms_clc_a_err={1:10.7f}'.format( math.sqrt(rms_clc_r_err/cnt), math.degrees(math.sqrt(rms_clc_a_err/cnt)) ) 
        f_csv.close()

#
# Run the test cases
#
test_cases()
