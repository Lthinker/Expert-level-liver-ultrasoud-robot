# from simulation import MotorUI, inverse_kinematics, H, D, A, K
import numpy as np
import time
from communicate.comutil import create_client, read_float_values
import socket
import json
from scipy.spatial.transform import Rotation as ScR
import robotcontrol
with open(r'C:\Users\xuyixiao\Desktop\Seafile\私人资料库\和子涵的paper\CODE\42_control_ code\superpara.json', 'r') as f:
    superpara = json.load(f)
    PORT_SEND = superpara.get('botport', 10101) 
class Robot:
    # 接口函数
    def getArmPos(self):
        pose_quat = self._getPose_() 
        pose_rotvec = robotcontrol.quat2rec(np.array(pose_quat))
        return pose_rotvec

    def movel_tool(self,d_pose):
        # d_pose是旋转矢量
        d_pose_quat = robotcontrol.rvec2quat(d_pose) # 转换为四元数
        self._movel_(d_pose_quat) 

    def getTheoPose(self):
        if hasattr(self, 'TheoPose') and self.TheoPose is not None:  
            return self.TheoPose
        else:  
            return self._getPose_() 
    # 一次把所有命令都发过去
    def movel_waypoints(self,**kwargs):
        array_str_all = ''
        for p in kwargs['pose']:
            p_quat = robotcontrol.rvec2quat(p)
            if isinstance(p_quat, np.ndarray):  
                array_str = ",".join([str(val) for val in p_quat]) 
            else:
                array_str = str(p_quat)  
            
            if len(array_str_all) == 0:
                array_str_all = array_str
            else:
                array_str_all = array_str_all + '|' + array_str
            command = f"Waypoints {array_str_all}"
        for f in kwargs['force']:
            if isinstance(f, np.ndarray):  
                array_str = ",".join([str(val) for val in f]) 
            else:
                array_str = str(f)  
            
            if len(array_str_all) == 0:
                array_str_all = array_str
            else:
                array_str_all = array_str_all + '|' + array_str
            command = command + f"| Force {array_str_all}"
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:  
            s.connect(('127.0.0.1', PORT_SEND))  
            s.sendall(command.encode('utf-8'))  
            response = s.recv(1024)  
            print("_movel_执行的Server response:", response.decode('utf-8'))  
        start_time = time.time()  
        timeout = 600  # 60秒超时  
        while time.time() - start_time < timeout:    
            time.sleep(0.05)  # 每秒查询一次  
            try:  
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:  
                    s.connect(('127.0.0.1', PORT_SEND))  
                    s.sendall(b"CheckMovelDone")  
                    response = s.recv(1024)  
                    status = response.decode('utf-8')  
                    # print(status)
                    if status[:9] == 'MovelDone':
                        print(f'movel done 返回时间{time.time()}')
                        pose_str = status[9:]  
                        pose_array = list(map(float, pose_str.split(',')))  
                        self.TheoPose = pose_array
                        return "MovelDone"  
                    elif status == 'MovelInit':
                        # print(status)  
                        pass
                    elif status == 'Moveling':  
                        # print(status)  
                        pass
                    elif status == 'MovelError':
                        return status
                    else:
                        assert(0)
            except Exception as e:  
                print(f"Error checking motion status: {e}")  
                return "MovelError"  
        return "MovelTimeOut"

    def movel_waypoints_old(self,**kwargs):
        for p in kwargs['pose']:
            p_quat = robotcontrol.rvec2quat(p)
            response = self._movel_(p_quat)
            print("movel_waypoints单步的response为",response,' ',time.time())
            time.sleep(0.05)

    def _movel_(self, pose_quat):  
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:  
            s.connect(('127.0.0.1', PORT_SEND))  
            if isinstance(pose_quat, np.ndarray):  
                array_str = ",".join([str(val) for val in pose_quat])  
            else:  
                array_str = str(pose_quat)  
            command = f"Movel {array_str}"  
            s.sendall(command.encode('utf-8'))  
            response = s.recv(1024)  
            print("_movel_执行的Server response:", response.decode('utf-8'))  
        start_time = time.time()  
        timeout = 600  

        while time.time() - start_time < timeout:    
            time.sleep(0.05)   
            try:  
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:  
                    s.connect(('127.0.0.1', PORT_SEND))  
                    s.sendall(b"CheckMovelDone")  
                    response = s.recv(1024)  
                    status = response.decode('utf-8')  
                    # print(status)
                    if status[:9] == 'MovelDone':
                        print(f'movel done 返回时间{time.time()}')
                        pose_str = status[9:]  
                        pose_array = list(map(float, pose_str.split(',')))  
                        self.TheoPose = pose_array
                        return "MovelDone"  
                    elif status == 'MovelInit':
                        # print(status)  
                        pass
                    elif status == 'Moveling':  
                        # print(status)  
                        pass
                    elif status == 'MovelError':
                        return status
                    else:
                        assert(0)
            except Exception as e:  
                print(f"Error checking motion status: {e}")  
                return "MovelError"  
        return "MovelTimeOut"
    
    def Get6DForce(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:  
            s.connect(('127.0.0.1', PORT_SEND))  
            s.sendall(b'Get6DForce')  
            response = json.loads(s.recv(1024).decode('utf-8'))
            # print(response)
            force = response["force"]
            return [force['Fx'],force['Fy'],force['Fz'],
                    force['Tx'],force['Ty'],force['Tz'],
                    ]  
    def _getPose_(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:  
            s.connect(('127.0.0.1', PORT_SEND))  
            s.sendall(b'GetTcpPose')  
            response = json.loads(s.recv(1024).decode('utf-8'))
            # print(response)
            return response["pose"] 

    def initForce(self):
        pass
    def get_arm_pos_basic(self): 
        pass

    def get_arm_force(self):
        while(1):
            try:
                force, time_stamp = read_float_values(self.client)
                break
            except Exception as err:
                print(f'Connecting with force sensor error {err}, retrying ...')
                time.sleep(0.001)
        return force, time_stamp
    
    def movel(self,data_pose):
        self.movel_bacis(data_pose)

    def get_arm_pos_basic(self): 
        return data_pose
    
if __name__ == "__main__":
    try:
        robot = Robot() 
        robot._getPose_() # work
        robot._movel_(np.array([0,0,-9.92,0,0,0,1]))
    except Exception as e:
        print(e)

