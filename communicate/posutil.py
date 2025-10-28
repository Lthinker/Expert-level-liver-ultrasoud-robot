import numpy as np
import numpy as np  
from scipy.spatial.transform import Rotation as R  

class PosAgent:
    def __init__(self):
        self.pos = np.array([0,0,0])
        self.ori = np.array([0,0,0]) # 其实应该是9自由度位姿，多个ori1，ori2

    def get_pos(self):
        pos = np.concatenate([self.pos, self.ori])
        return pos
    
    def close(self):
        pass

def extend_along_orientation(position, orientation, extension_length):  

    # 计算旋转矩阵  
    rotation = R.from_euler('xyz', orientation)  
    rotation_matrix = rotation.as_matrix()  

    # 计算在世界坐标系中的位移向量  
    displacement_local = np.array([0, 0, extension_length])  # 沿局部z轴方向延长  
    displacement_world = rotation_matrix @ displacement_local  # 转换到世界坐标系  

    # 新的位置  
    new_position = np.array(position) + displacement_world  

    # 新的末端位姿 (保持原始的姿态)  
    new_pose = np.concatenate((new_position, orientation))  

    return new_pose  

def adjust_pose_for_physical_offset(  
        pose_xyzabc: np.ndarray,   
        offset_z_deg: float = 90.0  
    ):  
    # 1) 解析输入  
    px, py, pz, alpha, beta, gamma = pose_xyzabc  

    # 2) 将原来的欧拉角 -> 旋转矩阵 R_orig  
    R_orig = R.from_euler('xyz', [alpha, beta, gamma], degrees=True)  
    
    # 3) 偏载矩阵: R_offset = RotZ(+offset_z_deg)  
    R_offset = R.from_euler('z', offset_z_deg, degrees=True)  
    # 其逆: R_offset_inv = RotZ(-offset_z_deg)  
    R_offset_inv = R_offset.inv()  # 或 R.from_euler('z', -offset_z_deg, degrees=True)  

    R_correct_mat = R_offset_inv.as_matrix() @ R_orig.as_matrix()  
    R_correct = R.from_matrix(R_correct_mat)  
    alpha_c, beta_c, gamma_c = R_correct.as_euler('xyz', degrees=True)  
    p_orig = np.array([px, py, pz])  
    p_correct = R_offset_inv.apply(p_orig)  # apply() 用于旋转向量  
    
    return np.array([*p_correct, alpha_c, beta_c, gamma_c])  
