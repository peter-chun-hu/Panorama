import pickle
import sys
import time
import numpy as np
from transforms3d.euler import euler2mat, mat2euler, axangle2euler, euler2quat, quat2euler
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import statistics
import os
import math

class Quaternion:
    def __init__(self, s, v = None):
        if v is None:
            self.vector = np.array(s)
            self.s = np.array(self.vector[0])
            self.v = np.array([self.vector[1], self.vector[2], self.vector[3]])
        else:
            self.s = np.array(s)
            self.v = np.array(v)
            self.vector = np.array([s, v[0], v[1], v[2]])        
    def add(self, p):
        return Quaternion(self.s + p.s, self.v + p.v)
    def multiply(self, p):
        return Quaternion(self.s.dot(p.s) - self.v.dot(p.v), self.s.dot(p.v) + p.s.dot(self.v) + np.cross(self.v, p.v))
    def conjugate(self):
        return Quaternion(self.s, -self.v)
    def norm(self):
        return self.s.dot(self.s) + self.v.dot(self.v)
    def inverse(self):
        norm = self.norm()
        temp = self.conjugate()
        return Quaternion(temp.s / norm, np.divide(temp.v, norm))

def tic():
    return time.time()
def toc(tstart, nm=""):
    print('%s took: %s sec.\n' % (nm,(time.time() - tstart)))

def read_data(fname):
    d = []
    with open(fname, 'rb') as f:
        if sys.version_info[0] < 3:
            d = pickle.load(f)
        else:
            d = pickle.load(f, encoding='latin1')  # need for python 3
    return d

def read_dataset(dataset):
    cfile = "../testset/cam/cam" + dataset + ".p"
    ifile = "../testset/imu/imuRaw" + dataset + ".p"
    vfile = "../testset/vicon/viconRot" + dataset + ".p"

    ts = tic()
    # camd = []
    camd = read_data(cfile)
    imud = read_data(ifile)
    vicd = []
    # vicd = read_data(vfile)
    toc(ts,"Data import")
    return [imud, vicd, camd]

def draw_vic(vicd):
    x_table = []
    y_table = []
    z_table = []
    for i in range(1000):
        R = np.array(vicd['rots'][:,:,i])
        x, y, z = mat2euler(R, 'sxyz')
        x_table.append(x)
        y_table.append(y)
        z_table.append(z)
    plt.plot(range(len(x_table)), x_table, 'r--', range(len(y_table)), y_table, 'bs', range(len(z_table)), z_table, 'g^')
    plt.show()

def calculate_bias(imud, num):
    Ax_table = []
    Ay_table = []
    Az_table = []
    Wx_table = []
    Wy_table = []
    Wz_table = []
    for i in range(num):
        Ax_table.append(imud['vals'][0,i])
        Ay_table.append(imud['vals'][1,i])
        Az_table.append(imud['vals'][2,i])
        Wz_table.append(imud['vals'][3,i])
        Wx_table.append(imud['vals'][4,i])
        Wy_table.append(imud['vals'][5,i])
    return [sum(Ax_table)/float(len(Ax_table)), sum(Ay_table)/float(len(Ay_table)), sum(Az_table)/float(len(Az_table)) - 300*1023/3300, sum(Wx_table)/float(len(Wx_table)), sum(Wy_table)/float(len(Wy_table)), sum(Wz_table)/float(len(Wz_table))]

def calculate_acceleration(Ax_bias, Ay_bias, Az_bias, imud):
    acceleration = []
    scale = 3300 / 1023 / 300
    for i in range(len(imud['vals'][0])):
        acceleration_x = -(imud['vals'][0,i] - Ax_bias) * scale
        acceleration_y = -(imud['vals'][1,i] - Ay_bias) * scale
        acceleration_z = (imud['vals'][2,i] - Az_bias) * scale
        acceleration.append([acceleration_x, acceleration_y, acceleration_z])
    return np.array(acceleration)

def calculate_angle_velocity(Wx_bias, Wy_bias, Wz_bias, imud):
    angle_velocity = []
    delta_time = []
    scale = 3300 / 1023 / 3.33 * np.pi / 180
    for i in range(len(imud['vals'][0])-1):
        angle_velocity_z = (imud['vals'][3,i] - Wz_bias) * scale
        angle_velocity_x = (imud['vals'][4,i] - Wx_bias) * scale
        angle_velocity_y = (imud['vals'][5,i] - Wy_bias) * scale
        angle_velocity.append([angle_velocity_x, angle_velocity_y, angle_velocity_z])
        delta_time.append(imud['ts'][0][i+1] - imud['ts'][0][i])
    return np.array(angle_velocity), np.array(delta_time)

def calculate_imu_angle(angle_velocity, delta_time, imud):
    imu_angle = np.zeros((1,3))
    imu_time = []
    imu_time.append(imud['ts'][0][0])
    for i in range(len(angle_velocity)):
        angle_x = imu_angle[i,0] + angle_velocity[i][0] * delta_time[i]
        angle_y = imu_angle[i,1] + angle_velocity[i][1] * delta_time[i]
        angle_z = imu_angle[i,2] + angle_velocity[i][2] * delta_time[i]
        imu_angle = np.vstack((imu_angle, [angle_x, angle_y, angle_z]))
        imu_time.append(imud['ts'][0][i+1])
    return imu_angle, imu_time

def calculate_imu_angle_quaternion(angle_velocity, delta_time, imud):
    quaternion = Quaternion([1,0,0,0])
    imu_angle = np.array(quat2euler(quaternion.vector))
    imu_time = []
    imu_time.append(imud['ts'][0][0])
    for i in range(len(angle_velocity)):
        temp = angle_velocity[i][:] * delta_time[i]
        quaternion = quaternion.multiply(Quaternion(euler2quat(temp[0], temp[1], temp[2])))

        imu_angle = np.vstack((imu_angle, quat2euler(quaternion.vector)))
        imu_time.append(imud['ts'][0][i+1])
    return imu_angle, imu_time

def calculate_vicon_angle(vicd):
    vicon_angle = []
    vicon_time = []
    for i in range(1,len(vicd['rots'][0,0])):
        x_deg, y_deg, z_deg = mat2euler(vicd['rots'][:, :, i])
        vicon_angle.append([x_deg, y_deg, z_deg])
        vicon_time.append(vicd['ts'][0][i])
    return np.array(vicon_angle), vicon_time

def draw(imu_angle, vicon_angle, acceleration, imu_time, vicon_time, filtered_angle):
    plt.figure(1)
    plt.subplot(321)
    line1, = plt.plot(vicon_time, vicon_angle[:,0], label="vicon data")
    line2, = plt.plot(imu_time, filtered_angle[:,0], label="UKF")
    line3, = plt.plot(imu_time, imu_angle[:,0], label="imu data")
    plt.legend(handles=[line1, line2, line3], loc=2)

    plt.subplot(323)
    line6, = plt.plot(vicon_time, vicon_angle[:,1], label="vicon data")
    line7, = plt.plot(imu_time, filtered_angle[:,1], label="UKF")
    line8, = plt.plot(imu_time, imu_angle[:,1], label="imu data")
    plt.legend(handles=[line6, line7, line8], loc=2)

    plt.subplot(325)
    line11, = plt.plot(vicon_time, vicon_angle[:,2], label="vicon data")
    line12, = plt.plot(imu_time, filtered_angle[:,2], label="UKF")
    line13, = plt.plot(imu_time, imu_angle[:,2], label="imu data")
    plt.legend(handles=[line11, line12, line13], loc=2)

    plt.show()
    return

def draw_no_vicon(imu_angle, acceleration, imu_time, filtered_angle):
    plt.figure(1)
    plt.subplot(321)
    line2, = plt.plot(imu_time, filtered_angle[:,0], label="UKF")
    line3, = plt.plot(imu_time, imu_angle[:,0], label="imu data")
    plt.legend(handles=[line2, line3], loc=2)

    plt.subplot(323)
    line7, = plt.plot(imu_time, filtered_angle[:,1], label="UKF")
    line8, = plt.plot(imu_time, imu_angle[:,1], label="imu data")
    plt.legend(handles=[line7, line8], loc=2)

    plt.subplot(325)
    line12, = plt.plot(imu_time, filtered_angle[:,2], label="UKF")
    line13, = plt.plot(imu_time, imu_angle[:,2], label="imu data")
    plt.legend(handles=[line12, line13], loc=2)

    plt.show()
    return

def calculate_avg_quaternions(weights, quaternions, initial_guess):
    q_e = quaternions[:]
    q_bar = initial_guess
    e_v = np.zeros((len(quaternions),3))
    for i in range(10000):
        e_sum = np.zeros(3)
        for j in range(len(quaternions)):
            q_e[j] = q_bar.inverse().multiply(quaternions[j])
            e_v[j] = np.array(quat2euler(q_e[j].vector))
            e_sum = e_sum + weights[j] * e_v[j][:]
        temp = euler2quat(e_sum[0], e_sum[1], e_sum[2])
        q_bar = q_bar.multiply(Quaternion(temp))
        if(np.linalg.norm(e_sum) < 0.00001):
            break;
    return q_bar, e_v

def UKF(acceleration, angle_velocity, delta_time):
    n = 3
    filtered_angle = np.zeros((1,3))
    P_r = 1 * np.identity(3)
    P = 0.01 * np.identity(3)
    Q = 0.000001 * np.identity(3)
    q = Quaternion([1, 0, 0, 0])
    angle_velocity_quat = []
    angle_velocity_quat.append(q)
    quaternion_avg = Quaternion([1, 0, 0, 0])
    E = []
    q_i = []
    for i in range(7):
        E.append(np.array([0, 0, 0]))
        q_i.append(Quaternion([1, 0, 0, 0]))
    alpha = np.array([0, 1/(2*n), 1/(2*n), 1/(2*n), 1/(2*n), 1/(2*n), 1/(2*n)])
    for index in range(len(angle_velocity)):
        print(index)

        E[0] = np.array([0,0,0])
        L = np.linalg.cholesky(n*(P+Q))
        for i in range(3):
            E[i+1] = np.array(L[:,i]) 
        for i in range(3):
            E[i+4] = np.array(-L[:,i])
        temp = angle_velocity[index][:]*delta_time[index]
        imu_quaternion = Quaternion(euler2quat(temp[0], temp[1], temp[2]))
        for i in range(7):
            q_i[i] = q.multiply(Quaternion(euler2quat(E[i][0], E[i][1], E[i][2]))).multiply(imu_quaternion)
        quaternion_avg, e_v = calculate_avg_quaternions(alpha, q_i, q_i[0])
        P = 0.0 * np.dot(e_v[0][np.newaxis].T, e_v[0][np.newaxis])
        for i in range(1,7):
            P = P + np.dot(e_v[i][np.newaxis].T, e_v[i][np.newaxis])/ (2*n)

        Z = []
        for i in range(7):
            temp = q_i[i].conjugate().multiply(Quaternion([0,0,0,1])).multiply(q_i[i])
            Z.append(np.array([temp.v[0], temp.v[1], temp.v[2]]))
        Z_mean = 0.0 * Z[0]
        for i in range(1,7):
            Z_mean = Z_mean + Z[i] / (2*n)

        P_zz = 0.0*np.dot((Z[0][:] - Z_mean[:])[np.newaxis].T, (Z[0][:]- Z_mean[:])[np.newaxis])
        for i in range(1,7):
            P_zz = P_zz + np.dot((Z[i][:] - Z_mean[:])[np.newaxis].T, (Z[i][:] - Z_mean[:])[np.newaxis]) / (2*n)

        P_vv = P_zz + P_r

        P_xz = 0.0*np.dot(e_v[0][np.newaxis].T, (Z[0][:] - Z_mean[:])[np.newaxis])
        for i in range(1,7):
            P_xz = P_xz + np.dot(e_v[i][np.newaxis].T, (Z[i][:] - Z_mean[:])[np.newaxis]) / (2*n)

        K = np.dot(P_xz, np.linalg.inv(P_vv))

        temp = np.dot(K, (acceleration[index][:] - Z_mean[:])[np.newaxis].T)
        q = quaternion_avg.multiply(Quaternion(euler2quat(temp[0], temp[1], temp[2])))
        angle_velocity_quat.append(q)
        P = P - np.dot(K, np.dot(P_vv, K.T))
        filtered_angle = np.vstack((filtered_angle, np.array(quat2euler(q.vector))))
    return filtered_angle, angle_velocity_quat

def test():
    quaternions = []
    q1 = Quaternion(euler2quat(0, 0, 170/180*np.pi))
    q2 = Quaternion(euler2quat(0, 0, -101/180*np.pi))
    q3 = Quaternion(euler2quat(0, 0, 270/180*np.pi))
    quaternions.append(q1)
    quaternions.append(q2)
    quaternions.append(q3)
    alpha = [1/3, 1/3, 1/3]
    temp = calculate_avg_quaternions(alpha, quaternions, q1)
    print(quat2euler(temp.vector)[2]*180/np.pi)
    return

def sph2cart(az, el):
    rcos_theta = np.cos(el*np.pi/180)
    x = rcos_theta * np.cos(az*np.pi/180)
    y = rcos_theta * np.sin(az*np.pi/180)
    z = np.sin(el*np.pi/180)
    normalize_distance = np.sqrt(x**2+y**2+z**2)
    x = x / normalize_distance
    y = y / normalize_distance
    z = z / normalize_distance
    return x, y, z

def cart2sph(x, y, z):
    normalize_distance = np.sqrt(x**2+y**2+z**2)
    x = x / normalize_distance
    y = y / normalize_distance
    z = z / normalize_distance
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy) #phi
    az = np.arctan2(y, x) #theta
    return az, el, r

def cart2Cy(x, y, z):
    p = np.sqrt(x*x+y*y)
    theta = np.arctan2(y,x)
    z = z
    return p, theta, z

def sph2cy(az, el, r):
    z = r * np.sin(el)
    theta = az
    return z, theta

def getViconMatrix(vicd):
    vicon = []
    for i in range(len(vicd['rots'][0,0])):
        vicon.append(vicd['rots'][:, :, i])
    return vicon

def panorama_imu(filtered_angle, camd, imu_time):
    imu_rotation_matrix = []
    for i in range(len(filtered_angle)):
        imu_rotation_matrix.append(euler2mat(filtered_angle[i][0], filtered_angle[i][1], filtered_angle[i][2]))

    height = 1200
    width = 1600

    height_index = 0
    width_index = 0
    length = len(camd['cam'][0][0][0])
    panorama_img = np.zeros((height, width, 3))
    spherical = np.zeros((240, 320, 2))
    cartesian = np.zeros((240, 320, 3))
    image_coordinate = np.zeros((240, 320, 3))
    temp = Quaternion([1,0,0,0])
    index = 0
    for_loop = True
    for i in range(240):
        # spherical[i][0] = (22.5 - 45/239/2 - i * 45/239, -30+60/319/2)
        spherical[i][0] = (22.5 - i * 45/239, -30)
    for i in range(240):
        for j in range(1,320):
            spherical[i][j] = spherical[i][j-1] + (0,60/319)
    for i in range(240):
        for j in range(320):
            cartesian[i][j] = sph2cart(spherical[i][j][1], spherical[i][j][0])
    
    width_scale = width / 2 / np.pi
    height_scale = height / 2
    for t in range(0, len(camd['cam'][i][j][0])):
        print(str(t) + "/" + str(len(camd['cam'][i][j][0])))
        while(np.abs(camd['ts'][0][t] - imu_time[index]) > 0.01):
            index = index + 1
            if(index >= len(imu_time)):
                for_loop = False
                break;
        if for_loop == False:
            break
        for i in range(240):
            for j in range(320):
                # temp = angle_velocity_quat[index].multiply(cartesian_quat[i][j].multiply(q_inv))
                temp = np.dot(imu_rotation_matrix[index], cartesian[i][j][:])
                # az, el, r = cart2sph(temp[0], temp[1], temp[2])
                # z, theta = sph2cy(az, el, r)
                p, theta, z = cart2Cy(temp[0], temp[1], temp[2])
                width_index = int((theta + np.pi) * width_scale)
                height_index = int((z + 1) * height_scale)
                panorama_img[height_index][width_index][0] = int(camd['cam'][i][j][0][t])
                panorama_img[height_index][width_index][1] = int(camd['cam'][i][j][1][t])
                panorama_img[height_index][width_index][2] = int(camd['cam'][i][j][2][t])
        if(t % 50 == 0):
            plt.clf()
            temp1 = np.flipud(panorama_img)
            temp1 = temp1.astype(np.uint8)
            imgplot = plt.imshow(temp1)
            plt.draw()
            plt.pause(0.000001)
    return panorama_img

def main():
    bias_threshold = 200
    dataset="10"
    [imud, vicd, camd] = read_dataset(dataset)
    [Ax_bias, Ay_bias, Az_bias, Wx_bias, Wy_bias, Wz_bias] = calculate_bias(imud, bias_threshold)
    acceleration = calculate_acceleration(Ax_bias, Ay_bias, Az_bias, imud)
    angle_velocity, delta_time = calculate_angle_velocity(Wx_bias, Wy_bias, Wz_bias, imud)
    imu_angle, imu_time = calculate_imu_angle_quaternion(angle_velocity, delta_time, imud)
    # vicon_angle, vicon_time = calculate_vicon_angle(vicd)
    filtered_angle, angle_velocity_quat = UKF(acceleration, angle_velocity, delta_time)
    # draw(imu_angle, vicon_angle, acceleration, imu_time, vicon_time, filtered_angle)
    draw_no_vicon(imu_angle, acceleration, imu_time, filtered_angle)
    # vicon = getViconMatrix(vicd)
    # temp_vicon = vicon[1800:]

    filtered_quat = []
    # panorama_img = panorama_vicon(temp_vicon, camd)
    panorama_img = panorama_imu(filtered_angle, camd, imud['ts'][0])
    panorama_img = np.flipud(panorama_img)
    panorama_img = panorama_img.astype(np.uint8)
    imgplot = plt.imshow(panorama_img[:][:][:])
    print("finish")
    plt.show()

if __name__ == "__main__": main()
