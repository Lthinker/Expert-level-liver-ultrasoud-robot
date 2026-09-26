from communicate.comutil import high_speed_collect, create_client
from communicate.imageutil import VideoIO, high_speed_image
from communicate.posutil import PosAgent, extend_along_orientation
from communicate.acqutil import AcqImageForce
import numpy as np
from einops import rearrange
import cv2
import time
import os
import pickle
import sys
import math3d as m3d
import numpy as np
import multiprocessing  
import threading  
import queue  
import shutil
from scipy.spatial.transform import Rotation as R  
from robotcontrol import update_pose_diffusionpolicy
import robotcontrol
import datetime
from communicate.comutil import read_float_values
import socket
import torch
import glob
from matplotlib import pyplot as plt
PORT = 65438
def robot_control_process(command_queue, response_queue):  
    ur_try = URTry()  
    while True:  
        if not command_queue.empty():  
            command, args = command_queue.get()  
            if command == 'getArmPos':  
                response = ur_try.getArmPos()  
                response_queue.put(response)  
            elif command == 'end_force_mode':
                ur_try.end_force_mode()
                response_queue.put('end force mode')
            elif command == 'movel_tool':  
                print('goingto move')
                d_pose, acc, vel, t = args['d_pose'], args['acc'], args['vel'], args['t']  
                ur_try.movel_tool(d_pose, acc, vel, t)  
                print('finish move')
                response_queue.put("movel_tool done")  
            elif command == 'movel_waypoints':  
                waypoints = args  
                ur_try.movel_waypoints(waypoints)  
                response_queue.put("movel_waypoints done")
            elif command == 'move_force':
                ur_try.move_force(**args)  
                response_queue.put("movel_force done")
            elif command == 'move_force_armpos':
                def track_arm_pose():  
                    while not stop_tracking.is_set():  
                        arm_pose = ur_try.getArmPos() 
                        response_queue.put({"current_pose": arm_pose}) 
                        time.sleep(0.05)  
                stop_tracking = threading.Event() 
                pose_tracking_thread = threading.Thread(target=track_arm_pose)  
                pose_tracking_thread.start()  
                print('goingto move in subprocess',time.time())
                ur_try.move_force_adaptive(**args)  
                join_start = time.time()
                stop_tracking.set()   
                pose_tracking_thread.join()
                print(f"Tracking thread joined after {time.time() - join_start:.2f} seconds")
                response_queue.put("movel_force done")

            elif command == 'move_force_zh':
                response = ur_try.move_force_zh(**args)  
                response_queue.put(response)
            elif command == 'get_force_base':
                response = ur_try.get_force_base()
                response_queue.put(response)
            elif command == 'update_force_base':
                response = ur_try.update_force_base()
                response_queue.put(response)
            elif command == 'get_tcp_force':
                response = ur_try.get_tcp_force()
                response_queue.put(response)
            elif command == 'set_force_remote':
                response = ur_try.set_force_remote(**args)
                response_queue.put("set_force_remote done")
            elif command == "STOP":  
                print("Process ending")  
                break  
        time.sleep(0.01)  

class USForceOfflineRead:
    def __init__(self,dpbuffer,fpcheckpoint=None,load_previous=False,fpofflinedata = ''):
        self.dpbuffer = dpbuffer
        self.load_previous = load_previous
        self.sixforcebase = np.array([0, 0, 0, 0, 0, 0])
        self.obs_pos_mode = 'euler'
        self.HistoryPos = []
        self.HistoryForce = []
        self.HistoryFrame = []
        self.TimeStamp = []

        self.HistoryPosall = []
        self.HistoryForceall = []
        self.HistoryFrameall = []
        self.TimeStampall = []
        self.classifier = None
        self.fpcheckpoint = fpcheckpoint
        self.recordall = []
        self.step_record_obs = []
        
        # offline: load data
        with open(fpofflinedata,'rb') as f:
            data = pickle.load(f)
        self.loadframes = np.array(data['frames'])
        self.loadforces = np.array(data['force'])
        self.loadposes = np.array(data['arm_pos'])
        
    def initpose(self):
        pass
    def send_command(self,command):  
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:  
            s.connect(('127.0.0.1', PORT))  
            s.sendall(command.encode('utf-8'))  
            response = s.recv(1024)  
            print("Server response:", response.decode('utf-8'))
    def read_image(self):
        # 接收图片数据  
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  
        self.client_socket.connect(('127.0.0.1', PORT))  
        self.client_socket.sendall(b"GET_FRAME")
        img_data = bytearray()  
        time_stamp = time.time()
        while True:  
            part = self.client_socket.recv(4096)  
            if not part:  
                break  
            img_data.extend(part)  

        img_array = np.frombuffer(img_data, dtype=np.uint8)  
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)          
        self.client_socket.close() 
        return image, time_stamp
    
    def read_force(self):
        while(1):
            try:
                force, time_stamp = read_float_values(self.client)
                break
            except Exception as err:
                print(f'Connecting with force sensor error {err}, retrying ...')
                time.sleep(0.001)
        return force, time_stamp

    def end_force_mode(self):
        self.command_queue.put(('end_force_mode', None))  
        while 1:
            if not self.response_queue.empty():  
                response = self.response_queue.get()     
                break
            else:
                time.sleep(0.01)
        
    def record_force(self):
        force, time_stamp = self.read_force()
        self.HistoryForce.append(force)
    
    def record_image(self):

        image, time_stamp = self.read_image()
        self.HistoryFrame.append(image)


    def setup(self,n_obs_steps):
        self.n_obs_steps = n_obs_steps

    def cal_rel_pose(self,n_obs_steps): 
        assert len(self.HistoryPos) != self.nposes,'forget to update the poses'
        self.nposes = len(self.HistoryPos) 

        HistoryPosLocal = self.HistoryPos[-n_obs_steps:]  
 
        posnow = HistoryPosLocal[-1]  
        posnow_quant = robotcontrol.rvec2quat(posnow)  
        abs_action_quant = [] 
        abs_action_euler = []

        for ii in range(0,len(HistoryPosLocal)-1): 
            q_prev = robotcontrol.rvec2quat(HistoryPosLocal[ii])[3:]
            t_prev = robotcontrol.rvec2quat(HistoryPosLocal[ii])[:3]
            q_curr = robotcontrol.rvec2quat(HistoryPosLocal[ii + 1])[3:]
            t_curr = robotcontrol.rvec2quat(HistoryPosLocal[ii + 1])[:3]
            
            T = robotcontrol.compute_relative_pose_element(q_prev, t_prev, q_curr, t_curr)
            t_rel = T[:3, 3]  
            R_rel = T[:3, :3]  
            r_rel = R.from_matrix(R_rel)  
            q_rel = r_rel.as_quat()  
            
            abs_action_quant.append(np.concatenate([t_rel, q_rel]))  
            abs_action_euler.append(np.concatenate([t_rel, r_rel.as_euler('xyz')]))  

        abs_action_quant.append(np.array([0,0,0,0,0,0,1]))
        abs_action_euler.append(np.array([0,0,0,0,0,0]))

        self.abs_action_quant = abs_action_quant
        self.abs_action_euler = abs_action_euler
    def setup_classifier(self,classifier):
        self.classifier = classifier

    def warmup(self):
        pass

    def close(self):
        pass
        
    def initenv(self):
        pass
    
    def robotstep(self,action):
        pass
   
    def get_sixforcebase(self):
        return self.sixforcebase
    
    def update_sixforcebase(self,num_measure=10):
        sixforcebase = np.zeros(6)
        for ii in range(0,num_measure):
            sixforce, timestamp = self.read_force()
            time.sleep(0.001)
            sixforcebase = sixforcebase + sixforce
        self.sixforcebase = sixforcebase/num_measure
        return self.get_sixforcebase()

    def GetState(self,timepoints = [],n_obs_steps = 5):
        self.cal_rel_pose(n_obs_steps)
        HistoryForceLocal = self.HistoryForce[-n_obs_steps:] 
        HistoryFrameLocal = self.HistoryFrame[-n_obs_steps:] 
        force = np.array(HistoryForceLocal)
        frame = np.array(HistoryFrameLocal) 

        if self.obs_pos_mode == 'euler':
            posseq = self.abs_action_euler
        elif self.obs_pos_mode == 'quat':
            posseq = self.abs_action_quant
        else:
            assert(0)
        pos = np.array(posseq,dtype=np.float32)
        pos_abs = self.HistoryPos[-n_obs_steps:]
        return force, frame, pos, pos_abs

    def scalenormimg(self,img):
        img = cv2.resize(img, (400,400) ,interpolation=cv2.INTER_AREA)
        img = rearrange(img,'h w c -> c h w')
        return np.expand_dims(img,0)/255.0

    def GetObs(self,timepoints = [],n_obs_steps = 5):
        force_seq, image_seq, pos_seq, pos_abs_seq = self.GetState(timepoints=timepoints, n_obs_steps = n_obs_steps) 
        force_seq = force_seq[-n_obs_steps:] 
        image_seq = image_seq[-n_obs_steps:] 

        image_seq_resize = [cv2.resize(item, (400,400) ,interpolation=cv2.INTER_AREA) for item in image_seq]
        image_seq_resize = np.stack(image_seq_resize, axis=0)
        image_seq_resize = rearrange(image_seq_resize,'n h w c -> n c h w')
        force_seq = np.array(force_seq) 
        pos_seq = np.array(pos_seq)
        done = np.array([False for ii in range(force_seq.shape[0])])


        newdata = {
            'obs': {
                'image': np.expand_dims(image_seq_resize,0)/255.0, 
                'force_state': np.expand_dims(force_seq,0),
            },
            'action': np.expand_dims(np.concatenate([force_seq,pos_seq],axis=1),0),
            'pos_abs_seq': np.array(pos_abs_seq),
        }
        return newdata, done
    
    def sensor_force_2_tcp_force(self,force,theta_degrees = 26.28  ): 
        if not isinstance(force, np.ndarray):  
            force = np.array(force)  
        assert(len(force.shape)==1)
        theta_radians = np.radians(theta_degrees)  
        
        # 构建旋转矩阵  
        self.rotation_matrix = np.array([  
            [np.cos(theta_radians), -np.sin(theta_radians), 0],  
            [np.sin(theta_radians),  np.cos(theta_radians), 0],  
            [0,                     0,                     1]  
        ])  

        tcp_force = np.dot(self.rotation_matrix, force[:3])  
        tcp_torque = np.dot(self.rotation_matrix, force[3:])
        return np.concatenate((tcp_force, tcp_torque))
    
    def transform_force_and_torque(self,sixforce, offset=[0, 0, 0.2]):    
        force = np.array(sixforce[:3])  
        torque = np.array(sixforce[3:])  
        offset = np.array(offset)  
        assert(len(force.shape)==1)
        
        new_torque = torque + np.cross(offset, force)  

        return np.concatenate((force, new_torque))

    def get_tcp_force_in_tcp_frame(self):  
        arm_pose = self.getArmPos()  
        tcp_force_base = self.get_tcp_force() 
        return self.cal_tcp_force_in_tcp_frame(arm_pose,tcp_force_base) 
    def cal_tcp_force_in_tcp_frame(self,arm_pose,tcp_force_base):    
        position = np.array(arm_pose[:3])  
        rotvec = np.array(arm_pose[3:])    
        rotation_matrix = R.from_rotvec(rotvec).as_matrix()  

        force_base = np.array(tcp_force_base[:3])  # [Fx, Fy, Fz]  
        torque_base = np.array(tcp_force_base[3:]) # [Mx, My, Mz]  

        force_tcp_frame = np.dot(rotation_matrix.T, force_base)  
        torque_tcp_frame = (np.dot(rotation_matrix.T, torque_base))  
        return np.concatenate((force_tcp_frame, torque_tcp_frame))   
       
    def get_tcp_force_in_base_frame(self):  
        arm_pose = self.getArmPos()  
        tcp_force_tcp = self.get_tcp_force_in_tcp_frame()   
        return self.cal_tcp_force_in_base_frame(self,arm_pose,tcp_force_tcp) 
    
    def cal_tcp_force_in_base_frame(self,arm_pose,tcp_force_tcp):  
        position = np.array(arm_pose[:3])  
        rotvec = np.array(arm_pose[3:])  
        rotation_matrix = R.from_rotvec(rotvec).as_matrix()  

        force_tcp = np.array(tcp_force_tcp[:3])  # [Fx, Fy, Fz]  
        torque_tcp = np.array(tcp_force_tcp[3:]) # [Mx, My, Mz]  
        force_base_frame = np.dot(rotation_matrix, force_tcp)  
        torque_base_frame = np.dot(rotation_matrix, torque_tcp)  
        torque_base_frame += np.cross(position, force_base_frame)  

        return np.concatenate((force_base_frame, torque_base_frame)) 

    def RecordHistoryPos(self,pos):
        self.HistoryPos.append(pos)

    def InitHistoryPos(self,n_obs_steps):
        for ii in range(0,n_obs_steps):
            self.RecordHistoryPos(self.getArmPos())

    def step(self,
             action=None,
             init=False,
             robforcebase=np.array([-2.11751355,  2.40337215, -2.52234725, -0.00851105,  0.0900982,   0.01731503]),
             sixforcebase=np.array([ 1.66440001,  0.6642,      2.0904,     -0.008,       0.021,      -0.01      ]),
             obs_dict=None,
             load_previous=False,
             topindices=[],
             lastobs = None): 
        action = action[0] 
        usewaypoint = 1
        useforcewaypoint = 0
        learntraj = 1

        if init:
            self.globalstep = 0
            timepoints = [time.time()]*self.n_obs_steps
            step_n_obs_steps = self.n_obs_steps
            self.nposes = 0
            for ii in range(0,self.n_obs_steps):
                # Online: read from sensor
                # self.record_force()
                # self.record_image()
                # self.HistoryPos.append(self.getArmPos())
                # Offline: load from existing file
                self.HistoryForce.append(self.loadforces[ii])
                self.HistoryPos.append(self.loadposes[ii])
                self.HistoryFrame.append(self.loadframes[ii])
        
        else:
            timepoints = []                 
            # Take the last "attentioned" pose
            init_pos = lastobs['pos_abs_seq'][topindices[-1]]
            init_pos_quat = robotcontrol.rvec2quat(init_pos)
            current_pose_6d = robotcontrol.rvec2quat(init_pos)
            waypoints = []
            arm_pos = self.getArmPos()
            doneimages = []
            pose_nexts = []
            force_list = []
            # Online: update the force sensor real time
            # sixforcebase_tmp = self.update_sixforcebase() 
            # robforcebase_tmp = self.update_robforcebase()
            # Offline: assume zero force base
            sixforcebase_tmp = np.array([0,0,0,0,0,0]) 
            robforcebase_tmp = np.array([0,0,0,0,0,0])
            
            for kk in range(0,5):
                start = time.time()    
                operation = action[kk][6:]
                pred_force_in_sensor = action[kk][:6] 
                current_pose_6d = update_pose_diffusionpolicy(operation = operation, current_pose_6d = init_pos_quat)
                pose_next = robotcontrol.quat2rec(current_pose_6d)
                pose_nexts.append(pose_next)
                waypoints.append({'pose': pose_next, 'a':0.1, 'v':0.1, 't':0})

                if 1:
                    robforcebase_tcp = self.cal_tcp_force_in_tcp_frame(arm_pos,robforcebase_tmp)
                    pred_force_in_sensor_norm = pred_force_in_sensor - sixforcebase_tmp
                    pred_force_in_tcp_norm = self.sensor_force_2_tcp_force(pred_force_in_sensor_norm,theta_degrees=80)
                    pred_force_in_tcp_norm = self.transform_force_and_torque(pred_force_in_tcp_norm)
                    
                    force_in_tcp = pred_force_in_tcp_norm + robforcebase_tcp
                    thres_x = 5; thres_y = 5; thres_z = 3;
                    selection_vector = [0,0,1,0,0,0]
                    force_in_tcp[2] = min(abs(force_in_tcp[2]),15) # for safety
                    assert(pred_force_in_sensor[2]<10) # for safety
                    force_list.append(force_in_tcp)
            # Online: run on the robot
            # movel_begin_time = time.time()
            # print('going to move in main process',movel_begin_time)
            # movel_out = self.move_force_armpos({ 
            #     'pose': pose_nexts,  
            #     'a':0.5, 
            #     'v':1, 
            #     't':0,
            #     'r':0.0, 
            #     'movetype':'l',
            #     'f_type':3,
            #     'task_frame':arm_pos, 
            #     'selection_vector':selection_vector, 
            #     'wrench':list(force_in_tcp),
            #     'force_list': force_list,
            #     'force_threshold':[15, 15, 15, 200, 200, 200],
            #     'limits': [0.05, 0.05, 0.01,1, 1, 1], 
            #     'wait':True 
            #     }) 

            # Offline: print the pose and force
            pose_nexts = np.array(pose_nexts)
            np.save(os.path.join('Outputaction','ExeAction.npy'),np.concatenate([force_list,pose_nexts],axis=-1))
            np.save(os.path.join('Outputaction','PredAction.npy'),action)
            exit()

        newdata, done = self.GetObs(timepoints = timepoints,n_obs_steps = step_n_obs_steps)
        reward = None
        info = None
        return newdata, reward, done, info
    
    def get_tcp_force(self):
        self.command_queue.put(('get_tcp_force', None))  
        while 1:
            if not self.response_queue.empty():  
                tcp_force = self.response_queue.get()     
                break
            else:
                time.sleep(0.01)
        return tcp_force
    def getArmPos(self):
        # Online: read arm pose from sensor real time
        # self.command_queue.put(('getArmPos', None))  
        # while 1:
        #     if not self.response_queue.empty():  
        #         arm_pos = self.response_queue.get()     
        #         break
        #     else:
        #         time.sleep(0.01)
        # return arm_pos
        # Offline: load the arm pose from the latest arm pose in the history
        return self.HistoryPos[-1]

    def movel_tool(self,movel_args):
        self.command_queue.put(('movel_tool', movel_args ))  
        while 1:
            if not self.response_queue.empty():  
                movel_out = self.response_queue.get()  
                break
            else:
                time.sleep(0.01)
        return movel_out
    def movel_waypoints(self,waypoints):
        self.command_queue.put(('movel_waypoints', waypoints ))  
        while 1:
            if not self.response_queue.empty():  
                movel_out = self.response_queue.get()  
                break
            else:
                time.sleep(0.01)
        return movel_out
    def move_force(self,movel_args):
        self.command_queue.put(('move_force', movel_args ))  
        # print('move_force, ',movel_args)
        while 1:
            if not self.response_queue.empty():  
                movel_out = self.response_queue.get()  
                break
            else:
                time.sleep(0.01)
        return movel_out

    def move_force_armpos(self,movel_args): 
        self.command_queue.put(('move_force_armpos', movel_args ))  
        # print('move_force, ',movel_args)
        image_list = []
        force_list = []
        pose_list = []
        time_list = []
        while 1:
            if not self.response_queue.empty():  
                movel_out = self.response_queue.get()  
                if isinstance(movel_out,str):
                    break
                else:
                    image, time_stamp = self.read_image()
                    force, time_stamp = self.read_force()
                    pose_list.append(movel_out['current_pose'])
                    image_list.append(image)
                    force_list.append(force)
                    time_list.append(time_stamp)
            else:
                time.sleep(0.01)
        print('in function stop') 
        return {'image_list':image_list, 'force_list':force_list, 'pose_list':pose_list, 'time_list':time_list}
    
    
    def move_force_zh(self,movel_args):
        self.command_queue.put(('move_force_zh', movel_args ))  
        while 1:
            if not self.response_queue.empty():  
                movel_out = self.response_queue.get()  
                break
            else:
                time.sleep(0.01)
        return movel_out
    def get_robforcebase(self):
        self.command_queue.put(('get_force_base', None ))  
        while 1:
            if not self.response_queue.empty():  
                movel_out = self.response_queue.get()  
                break
            else:
                time.sleep(0.001)
        return movel_out
    def set_force_remote(self,args):
        self.command_queue.put(('set_force_remote', args ))  
        while 1:
            if not self.response_queue.empty():  
                movel_out = self.response_queue.get()  
                break
            else:
                time.sleep(0.01)
        return movel_out
    def update_robforcebase(self):
        self.command_queue.put(('update_force_base', None ))  
        while 1:
            if not self.response_queue.empty():  
                movel_out = self.response_queue.get()  
                break
            else:
                time.sleep(0.001)
        return movel_out
    
def read_latest_two_files(directory):  
    files = []
    filetimes = []
    for file in os.listdir(directory):
        if file.endswith('.pkl'):
            try:
                filetimes.append(os.path.getmtime(os.path.join(directory, file)))
                files.append(file)
            except Exception as err:
                pass
    filetimes_sort = np.argsort(filetimes)
    files = np.array(files)[filetimes_sort]

    if len(files) >= 2:  
        latest_files = files[-2:]  
        # print(latest_files)
        data = []  
        for file in latest_files:  
            filepath = os.path.join(directory, file) 
            while(1): 
                try:
                    with open(filepath, 'rb') as f:  
                        data.append(pickle.load(f)) 
                        break 
                    # print(f"Read data from {filepath}")  
                except Exception as err:
                    print('pickle load error: ',err)
                    time.sleep(0.01)
        return data  
    else:  
        print("Not enough files to read the latest two.")  
        return []

def sec2datetime(sec):
    dt = datetime.datetime.fromtimestamp(sec)
    formatted_time = dt.strftime("%Y%m%d_%H%M%S_%f")
    return formatted_time

def read_lookup_timepoint(directory,timepoints=[]):  
    frames = []
    forces = []
    sixforcebases = []

    files = []
    filetimes = []
    while(1):
        for file in os.listdir(directory):
            if file.endswith('.pkl'):
                try:
                    filetimes.append(os.path.getmtime(os.path.join(directory, file)))
                    files.append(file)
                except Exception as err:
                    pass
        filetimes_sort = np.argsort(filetimes)
        filetimes_sorted = np.array(filetimes)[filetimes_sort]
        files_sorted = np.array(files)[filetimes_sort]
        if timepoints[-1] <= filetimes_sorted[-1]:
            break 

    for timepoint in timepoints:
        index = np.searchsorted(filetimes_sorted,timepoint,side='right')
        file = files_sorted[index]
        filepath = os.path.join(directory, file)         
        while(1): 
            try:
                with open(filepath, 'rb') as f:  
                    data = pickle.load(f) 
                    break 
            except Exception as err:
                print('pickle load error: ',err)
                time.sleep(0.01)
        time_record = data['time_record']
        index_in = np.searchsorted(time_record,timepoint,side='right'); index_in = min(index_in,len(time_record)-1)
        frames.append(data['frame'][index_in])
        forces.append(data['force'][index_in])
        sixforcebases.append(data['sixforcebase'])
    return {'frames':frames,'forces':forces,"sixforcebase":sixforcebases}


def parse_filename_timestamp(filename):
    import re
    from datetime import datetime
    pattern = r'result_\d+_(\d{8})_(\d{6})_(\d+)\.pkl'

    # 匹配文件名
    match = re.match(pattern, filename)

    if match:
        date_str = match.group(1)
        time_str = match.group(2)
        microsecond_str = match.group(3)

        datetime_str = date_str + time_str

        dt = datetime.strptime(datetime_str, '%Y%m%d%H%M%S')

        microsecond = int(microsecond_str[:6].ljust(6, '0'))
        dt = dt.replace(microsecond=microsecond)

        return dt
    else:
        raise ValueError("文件名格式不匹配")


class URTry:
    def __init__(self):
        TCP = 0.220 
        rz = np.radians(180-53)

        SERVER_PORT = '192.168.1.11'
        self.robotModel = URBasic.robotModel.RobotModel()
        self.rob = URBasic.urScriptExt.UrScriptExt(host=SERVER_PORT, robotModel=self.robotModel, hasForceTorque=True)
        self.rob.reset_error()
        self.rob.set_tcp((0, 0, TCP, 0, 0, rz)) # 
        self.rob.set_gravity([0, 0, 10])
        self.rob.set_payload_mass(0.3)
        self.robforcebase = np.array([0, 0, 0, 0, 0, 0])
    def getArmPos(self):
        """
        :return: ([float])->  np.ndarray Position (x, y, z) of Baxter left gripper
        """
        return self.rob.get_actual_tcp_pose()

    def movel_tool(self, d_pose, acc, vel, t=0):
        self.rob.movel(d_pose, a=acc, v=vel, t=t)

    def movel_waypoints(self,waypoints):
        self.rob.movel_waypoints(waypoints)

    def set_force_remote(self,task_frame=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], selection_vector=[0, 0, 0, 0, 0, 0],
                         wrench=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], limits=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1]):
        
        self.rob.set_force_remote(task_frame=task_frame, selection_vector=selection_vector,
                         wrench=wrench, limits=limits)
    def end_force_mode(self):
        self.rob.end_force_mode()
    def move_force(self,
                    pose,
                    a,
                    v,
                    t,
                    r=0.0,
                    movetype='l',
                    task_frame=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    selection_vector=[0, 0, 0, 0, 0, 0],
                    wrench=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    limits=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                    f_type=2,
                    wait=True,
                    q=None
                   ):
        self.rob.move_force(pose=pose,
                            a=a,
                            v=v,
                            t=t,
                            r=r,
                            movetype=movetype,
                            task_frame=task_frame,
                            selection_vector=selection_vector,
                            wrench=wrench,
                            limits=limits,
                            f_type=f_type,
                            wait=wait,
                            q=q)
    def move_force_adaptive(self,
                pose,
                a,
                v,
                t,
                r=0.0,
                movetype='l',
                task_frame=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                selection_vector=[0, 0, 0, 0, 0, 0],
                wrench=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                force_list = [],
                force_threshold = [3.0, 3.0, 10.0, 20, 20, 20],
                limits=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                f_type=2,
                wait=True,
                q=None
                ):
        self.rob.move_force_adaptive(pose=pose,
                            a=a,
                            v=v,
                            t=t,
                            r=r,
                            movetype=movetype,
                            task_frame=task_frame,
                            selection_vector=selection_vector,
                            wrench=wrench,
                            limits=limits,
                            f_type=f_type,
                            wait=wait,
                            q=q,
                            force_list = force_list,
                            force_threshold = force_threshold)
    def move_force_zh(self,
                pose,
                a,
                v,
                t,
                r=0.0,
                movetype='l',
                task_frame=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                selection_vector=[0, 0, 0, 0, 0, 0],
                wrench=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                limits=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                f_type=2,
                wait=True,
                q=None
                ):
        self.rob.move_force_zh(pose=pose,
                            a=a,
                            v=v,
                            t=t,
                            r=r,
                            movetype=movetype,
                            task_frame=task_frame,
                            selection_vector=selection_vector,
                            wrench=wrench,
                            limits=limits,
                            f_type=f_type,
                            wait=wait,
                            q=q)
    def get_tcp_force(self):
        return self.rob.get_tcp_force()

    def get_tcp_force_norm(self):
        force_norm = self.rob.get_tcp_force()-self.robforcebase
        return force_norm
    
    def update_force_base(self,num_measure=10):
        self.robforcebase = np.array([0, 0, 0, 0, 0, 0])
        for ii in range(0,num_measure):
            self.robforcebase = self.robforcebase + self.rob.get_tcp_force()
        self.robforcebase = self.robforcebase/num_measure
        return self.robforcebase
    def set_force_base(self,forcebase):
        self.robforcebase = forcebase

    def get_force_base(self):
        return self.robforcebase

    def sensor_force_2_tcp_force(self,force):
        assert(0)
        if not isinstance(force, np.ndarray):  
            force = np.array(force)  
        assert(len(force.shape)==1)
        theta_degrees = 26.28  
        theta_radians = np.radians(theta_degrees)  
        
        # 构建旋转矩阵  
        self.rotation_matrix = np.array([  
            [np.cos(theta_radians), -np.sin(theta_radians), 0],  
            [np.sin(theta_radians),  np.cos(theta_radians), 0],  
            [0,                     0,                     1]  
        ])  

        tcp_force = np.dot(self.rotation_matrix, force[:3])  
        tcp_torque = np.dot(self.rotation_matrix, force[3:])
        return np.concatenate((tcp_force, tcp_torque))
        
    def get_tcp_force_in_tcp_frame(self):  
        arm_pose = self.getArmPos() 
        tcp_force_base = self.get_tcp_force() 

        position = np.array(arm_pose[:3])  # [x, y, z] 
        rotvec = np.array(arm_pose[3:])    # [rx, ry, rz]  
        rotation_matrix = R.from_rotvec(rotvec).as_matrix()  
  
        force_base = np.array(tcp_force_base[:3])  # [Fx, Fy, Fz]  
        torque_base = np.array(tcp_force_base[3:]) # [Mx, My, Mz]  

        force_tcp_frame = np.dot(rotation_matrix.T, force_base)  

        # M_tcp = R * M_b + p x (R * F_b)  
        torque_tcp_frame = (np.dot(rotation_matrix.T, torque_base))  
        return np.concatenate((force_tcp_frame, torque_tcp_frame))  
    def transform_force_and_torque(self,sixforce, offset=[0, 0, -0.034]):  
        force = np.array(sixforce[:3])  
        torque = np.array(sixforce[3:])  
        offset = np.array(offset)  
        assert(len(force.shape)==1)
        new_torque = torque + np.cross(offset, force)  
        return np.concatenate((force, new_torque))  


def update_pose(current_pose, relative_pose):  
    t_abs = current_pose['position']  
    q_abs = current_pose['orientation']  
    t_rel = relative_pose['delta_position']  
    q_rel = relative_pose['delta_orientation']  
  
    r_abs = R.from_quat(q_abs)  
    r_rel = R.from_quat(q_rel)  
    r_new = r_abs * r_rel  
    q_new = r_new.as_quat()  

    t_new = t_abs + r_abs.apply(t_rel)  

    next_pose = {  
        'position': t_new,  
        'orientation': q_new  
    }  
    return next_pose  

import numpy as np  
from scipy.spatial.transform import Rotation as R  
def rvec2quat(tcp_pose): 
    position = tcp_pose[0:3]  # [x, y, z]  
    rotvec = tcp_pose[3:6]  # [rx, ry, rz]  
    rotation = R.from_rotvec(rotvec)  
    quat = rotation.as_quat()  
    current_pose = {  
        'position': position,                
        'orientation': quat.tolist()        
    }  
    return current_pose

def quat2rec(current_pose):  
    x, y, z = current_pose['position']  
    quaternion = current_pose['orientation']  # [qx, qy, qz, qw]  
    quaternion = quaternion / np.linalg.norm(quaternion)  

    rotation = R.from_quat(quaternion)  
 
    rot_vec = rotation.as_rotvec()  
    rx, ry, rz = rot_vec  

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

def euler_to_rotation_vector(roll, pitch, yaw):  
    roll = np.radians(roll)  
    pitch = np.radians(pitch)  
    yaw = np.radians(yaw)  
  
    R_x = np.array([[1, 0, 0],  
                    [0, np.cos(roll), -np.sin(roll)],  
                    [0, np.sin(roll), np.cos(roll)]])  
    
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],  
                    [0, 1, 0],  
                    [-np.sin(pitch), 0, np.cos(pitch)]])  
    
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],  
                    [np.sin(yaw), np.cos(yaw), 0],  
                    [0, 0, 1]])  
 
    R = R_z @ R_y @ R_x   
    theta = np.arccos((np.trace(R) - 1) / 2)  
 
    if theta != 0:  
        n = np.array([  
            R[2, 1] - R[1, 2],  
            R[0, 2] - R[2, 0],  
            R[1, 0] - R[0, 1]  
        ]) / (2 * np.sin(theta))  
    else:  
        n = np.array([0, 0, 0]) 
    rotation_vector = theta * n  
    return rotation_vector  

def euler_to_quaternion(roll, pitch, yaw):  
    q_w = (np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) +  
            np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2))  
    
    q_x = (np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) -  
            np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2))  
    
    q_y = (np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) +  
            np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2))  
    
    q_z = (np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) -  
            np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2))  

    return np.array([q_x, q_y, q_z, q_w])  
import numpy as np  
from scipy.spatial.transform import Rotation as R  

def compute_pose_error_with_rotvec(tA, rA, tB, rB):  
    translation_error = np.linalg.norm(tA - tB)   
    rotationA = R.from_rotvec(rA)  
    rotationB = R.from_rotvec(rB)  
    rotation_error = rotationA.inv() * rotationB 
    rotation_error_angle = rotation_error.magnitude()  
    rotation_error_angle_deg = np.degrees(rotation_error_angle)  

    return translation_error, rotation_error_angle_deg  



def save_images_to_video(image_list, video_path, frame_rate=30):   
    if not isinstance(image_list, list) or len(image_list) == 0:  
        raise ValueError("image_list must be a non-empty list of images.")  
     
    first_frame = image_list[0]  
    if not isinstance(first_frame, np.ndarray):  
        raise ValueError("The frames in image_list must be numpy arrays.")  
    frame_size = (first_frame.shape[1], first_frame.shape[0])  

    for frame in image_list:  
        if not isinstance(frame, np.ndarray):  
            raise ValueError("Each frame in image_list must be a numpy array.")  
        if frame.shape[1] != frame_size[0] or frame.shape[0] != frame_size[1]:  
            raise ValueError("All frames in image_list must have the same dimensions.")  
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')   
    video_writer = cv2.VideoWriter(video_path, fourcc, frame_rate, frame_size)  

    for frame in image_list:  
        video_writer.write(frame)  
    
    video_writer.release()  

    print(f"Video saved to: {video_path}")  


def mp4_to_numpy(video_path, resize=None):  
    cap = cv2.VideoCapture(video_path)  
    if not cap.isOpened():  
        raise ValueError(f"无法打开视频文件：{video_path}")  
    
    frames = []  

    while True:  
        ret, frame = cap.read()  
        if not ret: 
            break    
        if resize is not None:  
            frame = cv2.resize(frame, resize)  

        frames.append(frame)  
    
    return frames