import numpy as np  
from scipy.spatial.transform import Rotation as R  
def rvec2quat(tcp_pose):
    # 给定的机械臂 TCP 位姿  
    # 前三个元素为位置 [x, y, z]  
    # 后三个元素为旋转矢量 [rx, ry, rz]，以弧度为单位  

    # 提取位置部分  
    position = tcp_pose[0:3]  # [x, y, z]  

    # 提取旋转矢量部分  
    rotvec = tcp_pose[3:6]  # [rx, ry, rz]  

    # 将旋转矢量转换为旋转对象  
    rotation = R.from_rotvec(rotvec)  

    # 获取对应的四元数表示，格式为 [x, y, z, w]  
    quat = rotation.as_quat()  

    # 构建与 update_pose 兼容的当前位姿字典  
    # current_pose = {  
    #     'position': position,              # 位置列表  
    #     'orientation': quat.tolist()       # 四元数列表  
    # }  
    current_pose = np.concatenate((position, quat))
    return current_pose

def euler_to_quaternion(roll, pitch, yaw):  
    # 计算四元数分量  
    q_w = (np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) +  
            np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2))  
    
    q_x = (np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) -  
            np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2))  
    
    q_y = (np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) +  
            np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2))  
    
    q_z = (np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) -  
            np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2))  
    return np.array([q_x, q_y, q_z, q_w])  
def update_pose(current_pose, relative_pose):  
    """  
    根据当前位姿和相对位姿变换计算下一时刻的位姿。  

    参数：  
    - current_pose: 当前的位姿，包含位置和姿态（四元数）  
    - relative_pose: 相对位姿变换，包含位置增量和姿态增量（四元数）  

    返回：  
    - next_pose: 下一时刻的位姿，包含位置和姿态（四元数）  
    """  
    # 提取当前位姿  
    t_abs = current_pose['position']  
    q_abs = current_pose['orientation']  

    # 提取相对位姿  
    t_rel = relative_pose['delta_position']  
    q_rel = relative_pose['delta_orientation']  

    # 更新姿态（四元数相乘）  
    r_abs = R.from_quat(q_abs)  
    r_rel = R.from_quat(q_rel)  
    r_new = r_abs * r_rel  # 组合旋转  
    q_new = r_new.as_quat()  

    # 更新位置（应用当前姿态下的相对位移）  
    t_new = t_abs + r_abs.apply(t_rel)  

    # 返回更新后的位姿  
    next_pose = {  
        'position': t_new,  
        'orientation': q_new  
    }  
    return next_pose  
def quat2rec(current_pose):  
    # 提取位置  
    assert(len(current_pose)==7)
    x, y, z = current_pose[:3] 

    # 提取并归一化四元数  
    quaternion = current_pose[3:]  # [qx, qy, qz, qw]  
    quaternion = quaternion / np.linalg.norm(quaternion)  

    # 创建旋转对象  
    rotation = R.from_quat(quaternion)  

    # **方法二：转换为旋转向量**  
    rot_vec = rotation.as_rotvec()  # 得到旋转向量 [rx, ry, rz]  
    rx, ry, rz = rot_vec  

    # **组合位姿**  
    robot_pose = {  
        'x': x,  
        'y': y,  
        'z': z,  
        'rx': rx,  
        'ry': ry,  
        'rz': rz  
    }  
    robot_pose = np.array([  
        robot_pose['x'],  
        robot_pose['y'],  
        robot_pose['z'],  
        robot_pose['rx'],  
        robot_pose['ry'],  
        robot_pose['rz']  
    ])  
    return robot_pose  
def quantvec2quantdict(quantvec):
    return {
        'position': quantvec[:3],
        'orientation': quantvec[3:]
    }
def quantdict2quantvec(quantdict):
    return np.concatenate((quantdict['position'], quantdict['orientation']))


def update_pose_diffusionpolicy(operation,current_pose_6d):  
    '''
        pose x y z rx ry rz w
        action是一个长度为9的向量，前6个元素是力，后3个元素是位移和旋转(欧拉角)
    '''  
    assert(len(operation.shape)==1 and operation.shape[0]==6)

    t_rel = operation[:3] # 位移
    q_rel = euler_to_quaternion(operation[3],operation[4],operation[5]) # 旋转
    relative_pose =   {
            'delta_position': t_rel,  
            'delta_orientation': q_rel  
    }                     
    # next_pose  = update_pose(rvec2quat(current_pose_6d), relative_pose)  
    # current_pose_6d = quat2rec(next_pose)
    next_pose  = update_pose(quantvec2quantdict(current_pose_6d), relative_pose)  
    current_pose_6d = quantdict2quantvec(next_pose)
    return current_pose_6d
def update_pose_diffusionpolicy_inverse(operation,current_pose_6d):  
    '''
        pose x y z rx ry rz w
        action是一个长度为9的向量，前6个元素是力，后3个元素是位移和旋转(欧拉角)
    '''  
    assert(len(operation.shape)==1 and operation.shape[0]==6)

    q_rel = euler_to_quaternion(operation[3],operation[4],operation[5]) # 旋转 # x y z w
    q_rel = np.array([-q_rel[0],-q_rel[1],-q_rel[2],q_rel[3]])
    t_rel = -R.from_quat(q_rel).apply(operation[:3])
    relative_pose =   {
            'delta_position': t_rel,  
            'delta_orientation': q_rel  
    }                     
    # next_pose  = update_pose(rvec2quat(current_pose_6d), relative_pose)  
    # current_pose_6d = quat2rec(next_pose)
    next_pose  = update_pose(quantvec2quantdict(current_pose_6d), relative_pose)  
    current_pose_6d = quantdict2quantvec(next_pose)
    return current_pose_6d
def quat_to_euler(quat):  
    """将四元数转换为欧拉角（roll, pitch, yaw）"""  
    x, y, z, w = quat  
    # 计算欧拉角  
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))  
    pitch = np.arcsin(2*(w*y - z*x))  
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))  
    return roll, pitch, yaw  

def quatvec_to_eulervec(quatvec):  
    """将四元数转换为欧拉角（roll, pitch, yaw）"""  
    dx, dy, dz, x, y, z, w = quatvec
    # 计算欧拉角  
    roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))  
    pitch = np.arcsin(2*(w*y - z*x))  
    yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))  
    return np.array([dx,dy,dz, roll, pitch, yaw]  )

def compute_relative_pose_element(q_prev, t_prev, q_curr, t_curr):  
    """  
    计算基于四元数和位移的相对位姿变换。  

    参数：  
    - q_prev: 前一时刻的四元数（numpy数组，长度为4，格式为 [x, y, z, w]）  
    - t_prev: 前一时刻的位移向量（numpy数组，长度为3）  
    - q_curr: 当前时刻的四元数（numpy数组，长度为4，格式为 [x, y, z, w]）  
    - t_curr: 当前时刻的位移向量（numpy数组，长度为3）  

    返回：  
    - T: 4x4 齐次变换矩阵，表示从前一坐标系到当前坐标系的变换  
    """  
    # 规范化四元数  
    q_prev = q_prev / np.linalg.norm(q_prev)  
    q_curr = q_curr / np.linalg.norm(q_curr)  

    # 将四元数转换为旋转矩阵  
    R_prev = R.from_quat(q_prev).as_matrix()  
    R_curr = R.from_quat(q_curr).as_matrix()  

    # 计算相对旋转矩阵  
    R_rel = R_prev.T @ R_curr  

    # 计算相对位移  
    t_rel = R_prev.T @ (t_curr - t_prev)  
    # t_rel = t_curr - np.dot(R_rel, t_prev)  

    # 构造齐次变换矩阵  
    T = np.eye(4)  
    T[:3, :3] = R_rel  
    T[:3, 3] = t_rel  

    return T  

def matrix_to_rotvec(transform_mat):  
    translation = transform_mat[:3, 3]  

    # 2. 提取旋转矩阵  
    rotation_mat = transform_mat[:3, :3]  

    # 3. 创建 Rotation 对象并获取旋转矢量  
    r = R.from_matrix(rotation_mat)  
    rotvec = r.as_rotvec()  

    # 拼接结果 [x, y, z, rx, ry, rz]  
    result = np.concatenate([translation, rotvec])  
    return result  

def matrix_to_quat(transform_mat):  
    translation = transform_mat[:3, 3]  

    # 2. 提取旋转矩阵  
    rotation_mat = transform_mat[:3, :3]  

    # 3. 创建 Rotation 对象并获取旋转矢量  
    r = R.from_matrix(rotation_mat)  
    quat = r.as_quat()  

    # 拼接结果 [x, y, z, rx, ry, rz]  
    result = np.concatenate([translation, quat])  
    return result  
