import numpy as np
from scipy.spatial.transform import Rotation as R
from math import *
import geopy.distance

# camera info
fx = 809.0155 
fy = 807.7006 
cx = 948.0682 
cy = 523.6103 

# ground altitude
ground_altitude = 14.90

# camera_install angle
camera_install_angle_psi_deg = 0
camera_install_angle_theta_deg = -106
camera_install_angle_phi_deg = 0

def unit_vector(vector):
    magnitude = np.linalg.norm(vector)
    unit_vector = vector / magnitude
    return unit_vector

def Cal_GNSS_Position(input_info):
    '''
    input_info:  [GNSS_latitude, GNSS_longitude, flying_altitude, target_x, target_y, fav_roll(phi), fav_pitch(theta), fav_yaw(psi)]
    result:  [target_latitude, target_longitude]
    pg: [x, y, z]
    '''
    latitude = input_info[0]
    longitude = input_info[1]

    flying_height = input_info[2] - ground_altitude
    
    psi_radi = input_info[7]
    theta_radi = input_info[6]
    phi_radi = input_info[5]

    camera_psi_deg = camera_install_angle_psi_deg
    camera_theta_deg = camera_install_angle_theta_deg
    camera_phi_deg = camera_install_angle_phi_deg

    psi_radi_m = radians(camera_psi_deg)
    theta_radi_m = radians(camera_theta_deg)
    phi_radi_m = radians(camera_phi_deg)

    # the intrinsic matrix of the camera
    camera_K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # Matrix transformation
    trans_mtx = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    targt_pixel = np.array([input_info[3], input_info[4], 1])

    # R_g_b = (R.from_euler("zyx", [psi_radi, theta_radi, phi_radi])).as_matrix()
    # R_b_m = (R.from_euler("zyx", [psi_radi_m, theta_radi_m, phi_radi_m])).as_matrix()
    
    R_g_b = np.array([[cos(theta_radi)*cos(psi_radi), sin(phi_radi)*sin(theta_radi)*cos(psi_radi) - cos(phi_radi)*sin(psi_radi), cos(phi_radi)*sin(theta_radi)*cos(psi_radi)+sin(phi_radi)*sin(psi_radi)], 
                      [cos(theta_radi)*sin(psi_radi), sin(phi_radi)*sin(theta_radi)*sin(psi_radi) + cos(phi_radi)*cos(psi_radi), cos(phi_radi)*sin(theta_radi)*sin(psi_radi)-sin(phi_radi)*cos(psi_radi)], 
                      [-sin(theta_radi), sin(phi_radi)*cos(theta_radi), cos(phi_radi)*cos(theta_radi)]])
    
    
    R_b_m = np.array([[cos(theta_radi_m)*cos(psi_radi_m), sin(phi_radi_m)*sin(theta_radi_m)*cos(psi_radi_m) - cos(phi_radi_m)*sin(psi_radi_m), cos(phi_radi_m)*sin(theta_radi_m)*cos(psi_radi_m)+sin(phi_radi_m)*sin(psi_radi_m)], 
                      [cos(theta_radi_m)*sin(psi_radi_m), sin(phi_radi_m)*sin(theta_radi_m)*sin(psi_radi_m) + cos(phi_radi_m)*cos(psi_radi_m), cos(phi_radi_m)*sin(theta_radi_m)*sin(psi_radi_m)-sin(phi_radi_m)*cos(psi_radi_m)], 
                      [-sin(theta_radi_m), sin(phi_radi_m)*cos(theta_radi_m), cos(phi_radi_m)*cos(theta_radi_m)]])
    
    Pg_n = np.dot(R_g_b, np.dot(R_b_m, np.dot(trans_mtx, np.dot(np.linalg.inv(camera_K), targt_pixel))))
    Pe = unit_vector(Pg_n)
    Z_axis = np.array([0, 0, 1])
    data_z = np.sqrt(np.sum(Z_axis*Z_axis))
    data_Pe = np.sqrt(np.sum(Pe*Pe))
    cos_Pe_z = np.sum(Pe*Z_axis) / (data_Pe*data_z)
    L = flying_height / cos_Pe_z
    Pg = L*Pe

    GNSS_pos = geopy.distance.distance(kilometers=Pe[0]/1000).destination((latitude, longitude), bearing=0)  #bearing=0，North
    GNSS_pos = geopy.distance.distance(kilometers=Pe[1]/1000).destination((GNSS_pos[0], GNSS_pos[1]), bearing=90)  #bearing=90，East
    result = (GNSS_pos[0], GNSS_pos[1])
    return Pg, result

if __name__ == "__main__":
    # input_info:  [GNSS_latitude, GNSS_longitude, flying_altitude, target_x, target_y, fav_roll(phi), fav_pitch(theta), fav_yaw(psi)]
    input_info = [30, 120, 45, 960,540,0,0.1,0.2]
    Pg, result = Cal_GNSS_Position(input_info)
    print(Pg)
    print(result)