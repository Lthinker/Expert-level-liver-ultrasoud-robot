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
from robotcontrol import update_pose_diffusionpolicy, update_pose_diffusionpolicy_inverse
import robotcontrol
import datetime
from communicate.comutil import read_float_values
import socket
import torch
import glob

from scipy.spatial.transform import Rotation as R
from communicate import XYXRobot
import copy
import pdb
USERNAME = 'xuyixiao'
FPROOT = rf'C:\Users\{USERNAME}\Desktop\Seafile\私人资料库\和子涵的paper\CODE\Policy'

import json
with open(r'C:\Users\xuyixiao\Desktop\Seafile\私人资料库\和子涵的paper\CODE\42_control_ code\superpara.json', 'r') as f:
    superpara = json.load(f)
    PORT = superpara.get('imgport', 10020)

def async_save_snapshot(
    fpforcesave, fpposesave, fptimestamp, fpdecisionstep,
    fptimestampall, fpposesaveall, fpforcesaveall,
    dpimagesave, dpimagesaveall,
    force_selected, pose_selected, ts_selected, step_n_obs_steps,
    force_all, pose_all, ts_all,
    frames_selected, frames_all
):
    def save():
        start = time.time()
        save_vector_list_to_txt(fpforcesave, force_selected)
        save_vector_list_to_txt(fpposesave, pose_selected)
        save_to_txt(fptimestamp, ts_selected)
        save_to_txt(fpdecisionstep, [step_n_obs_steps])

        save_to_txt(fptimestampall, ts_all)
        save_vector_list_to_txt(fpposesaveall, pose_all)
        save_vector_list_to_txt(fpforcesaveall, force_all)

        for frame, ts in zip(frames_selected, ts_selected):
            ts_str = f"{ts:.6f}".replace('.', '_')
            filename = f'image_{ts_str}.png'
            cv2.imwrite(os.path.join(dpimagesave, filename), frame)

        for frame, ts in zip(frames_all, ts_all):
            ts_str = f"{ts:.6f}".replace('.', '_')
            filename = f'image_{ts_str}.png'
            cv2.imwrite(os.path.join(dpimagesaveall, filename), frame)

        print('savetime (async):', time.time() - start)

    threading.Thread(target=save, daemon=True).start()

def save_vector_list_to_txt(filepath, data_list, precision=6):
    with open(filepath, 'a') as f:
        for vec in data_list:
            line = '\t'.join([f'{x:.{precision}f}' for x in vec])
            f.write(line + '\n')

def save_to_txt(path, list_of_values, precision=6):
    with open(path, 'a') as f:
        for item in list_of_values:
            if isinstance(item, (list, tuple)):
                f.write('\t'.join([f'{x:.{precision}f}' for x in item]) + '\n')
            else:
                f.write(f'{item:.{precision}f}\n')

def robot_control_process(command_queue, response_queue):
    ur_try = XYXRobot.Robot()
    while True:
        if not command_queue.empty():
            command, args = command_queue.get()
            if command == 'getArmPos':
                response = ur_try.getArmPos()
                response_queue.put(response)
            elif command == 'getTheoPose':
                response = ur_try.getTheoPose()
                response_queue.put(response)
            elif command == 'Get6DForce':
                response = ur_try.Get6DForce()
                response_queue.put(response)
            elif command == 'movel_tool':
                print('goingto move')
                d_pose, acc, vel, t = args['d_pose'], args['acc'], args['vel'], args['t']
                ur_try.movel_tool(d_pose)
                print('finish move')
                response_queue.put("movel_tool done")
            elif command == 'movel_waypoints':
                def track_arm_pose():
                    while not stop_tracking.is_set():
                        arm_pose = ur_try.getArmPos()
                        force = ur_try.Get6DForce()
                        response_queue.put({"current_pose": arm_pose,"force":force})
                        time.sleep(0.05)
                stop_tracking = threading.Event()
                pose_tracking_thread = threading.Thread(target=track_arm_pose)
                pose_tracking_thread.start()
                ur_try.movel_waypoints(**args)
                join_start = time.time()
                stop_tracking.set()
                pose_tracking_thread.join()
                response_queue.put("movel_waypoints done")
            elif command == 'end_force_mode':
                ur_try.end_force_mode()
                response_queue.put('end force mode')
            elif command == 'move_force':
                ur_try.move_force(**args)
                response_queue.put("movel_force done")
            elif command == 'move_force_armpos':
                def track_arm_pose():
                    while not stop_tracking.is_set():
                        arm_pose = ur_try.getArmPos()
                        response_queue.put({"current_pose": arm_pose})
                        time.sleep(0.1)
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

class USForceOnlineRead:
    def __init__(self,dpbuffer,fpcheckpoint=None,load_previous=False):
        self.dpbuffer = dpbuffer
        self.load_previous = load_previous

        if 1:
            self.command_queue = multiprocessing.Queue()
            self.response_queue = multiprocessing.Queue()
            self.robot_process = multiprocessing.Process(target=robot_control_process, args=(self.command_queue, self.response_queue))
            self.robot_process.start()

        self.initpose()

        self.fpsavemarker = os.path.join(r'Marksave','saveonetime.txt')
        self.fpsavemarkerall = os.path.join(r'Marksave','saveonetimeall.txt')
        self.MustSaveMarker = os.path.join(r'Marksave','MustSaveStop.txt')
        self.TmpStop = os.path.join(r'Marksave','TemperalyStop.txt')

        if os.path.exists(self.fpsavemarker):
            os.remove(self.fpsavemarker)
        if os.path.exists(self.fpsavemarkerall):
            os.remove(self.fpsavemarkerall)
        if os.path.exists(self.MustSaveMarker):
            os.remove(self.MustSaveMarker)
        if os.path.exists(self.TmpStop):
            os.remove(self.TmpStop)

        self.send_command("STOP_RECORDING")
        self.send_command("START_RECORDING")
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
        self.actionall = []
        self.theoreticalpose_quat = np.array([0,0,0,0,0,0,1])

        self.fpinit = '/Data3/lzhdata3/diffusion_policy_general/debugoutput'
        dpresultsaveexp = os.path.join(rf'C:\Users\{USERNAME}\Desktop\recordresult',os.path.basename(os.path.dirname(os.path.dirname(self.fpcheckpoint)))    )
        if not os.path.exists(dpresultsaveexp):
            os.makedirs(dpresultsaveexp)
        now = datetime.datetime.now()
        date_time_str = now.strftime("%Y-%m-%d-%H-%M-%S")
        dpexpsave = os.path.join(dpresultsaveexp, date_time_str)
        if not os.path.exists(dpexpsave):
            os.makedirs(dpexpsave)

        self.fpforcesave = os.path.join(dpexpsave,'forcesave.txt')
        self.fptimestamp = os.path.join(dpexpsave,'timestamp.txt')
        self.fpposesave = os.path.join(dpexpsave,'posesave.txt')
        self.dpimagesave = os.path.join(dpexpsave,'imagesave')
        self.fpdecisionstep = os.path.join(dpexpsave,'decisionstep.txt')
        if not os.path.exists(self.dpimagesave):
            os.makedirs(self.dpimagesave)

        self.fpforcesaveall = os.path.join(dpexpsave,'forcesaveall.txt')
        self.fptimestampall = os.path.join(dpexpsave,'timestampall.txt')
        self.fpposesaveall = os.path.join(dpexpsave,'posesaveall.txt')
        self.dpimagesaveall = os.path.join(dpexpsave,'imagesaveall')
        if not os.path.exists(self.dpimagesaveall):
            os.makedirs(self.dpimagesaveall)


    def initpose(self):
        self.fpinit = 'saveforhuman.pkl'

    def send_command(self,command):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect(('127.0.0.1', PORT))
            s.sendall(command.encode('utf-8'))
            response = s.recv(1024)
            print("Server response:", response.decode('utf-8'))
    def read_image(self):
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
        self.command_queue.put(('Get6DForce', None))
        while 1:
            if not self.response_queue.empty():
                arm_pos = self.response_queue.get()
                break
            else:
                time.sleep(0.01)
        return arm_pos, time.time()

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
        force_base = np.array(tcp_force_base[:3])
        torque_base = np.array(tcp_force_base[3:])
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
        force_tcp = np.array(tcp_force_tcp[:3])
        torque_tcp = np.array(tcp_force_tcp[3:])
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
        startstep = time.time()
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

                self.record_force()
                self.record_image()
                self.HistoryPos.append(self.getArmPos())
                self.TimeStamp.append(time.time())
            self.theoreticalpose_quat = robotcontrol.rvec2quat(self.getArmPos())
            print('theoreticalpose_quat',self.theoreticalpose_quat)
        else:
            timepoints = []
            try:
                init_pos_quat = self.getTheoPose()
            except:
                init_pos_quat = self.getArmPos()
                print('cannot get theopose, get arm pos')
            print('init_pos_quat as reference',init_pos_quat)
            arm_pos = init_pos_quat
            current_pose_6d = init_pos_quat
            print('执行部分',robotcontrol.quat_to_euler(robotcontrol.rvec2quat(action[-1][6:])[-4:]))
            waypoints = []
            doneimages = []
            pose_nexts = []
            force_list = []
            sixforcebase_tmp = self.update_sixforcebase()
            action[:,6:9] = action[:,6:9]*1000
            for kk in range(0,5):
                start = time.time()
                operation = action[kk][6:]
                pred_force_in_sensor = action[kk][:6]
                self.actionall.append(action[kk])
                def CalTcpMove( original_pose_quat = np.array([0, 0, 0, 0, 0, 0,1]),
                                tcp2thing = np.array([0,0,0,np.pi,0,np.pi/2]),
                                action = np.array([0,0,0,0,0,0.2])
                                ):
                    thing_pose_quat = update_pose_diffusionpolicy(operation=tcp2thing,current_pose_6d=original_pose_quat)
                    thing_after_action_quat = update_pose_diffusionpolicy(operation=action,current_pose_6d=thing_pose_quat)
                    tcp_after_action_quat = update_pose_diffusionpolicy_inverse(operation=tcp2thing,current_pose_6d=thing_after_action_quat)
                    return tcp_after_action_quat
                current_pose_6d = CalTcpMove(action = operation, original_pose_quat = init_pos_quat)
                pose_next = robotcontrol.quat2rec(current_pose_6d)
                pose_nexts.append(pose_next)
                waypoints.append({'pose': pose_next, 'a':0.1, 'v':0.1, 't':0})
                force_in_tcp = self.sensor_force_2_tcp_force(pred_force_in_sensor,theta_degrees=0)
                force_in_tcp[2] = min(abs(force_in_tcp[2]),15)
                assert(pred_force_in_sensor[2]<10)
                force_list.append(force_in_tcp)

            if os.path.exists(self.TmpStop):
                self.globalstep = self.globalstep + self.n_obs_steps
                timepoints = [time.time()]*self.n_obs_steps
                step_n_obs_steps = self.n_obs_steps
                self.nposes = 0
                for ii in range(0,self.n_obs_steps):
                    self.record_force()
                    self.record_image()
                    self.HistoryPos.append(self.getArmPos())
                    self.TimeStamp.append(time.time())
                self.theoreticalpose_quat = robotcontrol.rvec2quat(self.getArmPos())
            elif (not os.path.exists(self.MustSaveMarker)):
                movel_out = self.movel_waypoints({
                    'pose': pose_nexts,
                    'force': force_list
                    })

                def evenly_select(lst, k):
                    n = len(lst)
                    if n <= k:
                        return lst
                    step = (n - 1) / (k - 1)
                    indices = [round(i * step) for i in range(k)]
                    return [lst[i] for i in indices]

                selfunc = evenly_select;selnum = min(200,len(movel_out['image_list']))
                timepoints.extend(selfunc(movel_out['time_list'],selnum))
                self.HistoryPos.extend(selfunc(movel_out['pose_list'],selnum))
                self.HistoryForce.extend(selfunc(movel_out['force_list'],selnum))
                self.HistoryFrame.extend(selfunc(movel_out['image_list'],selnum))
                step_n_obs_steps = max(5,len(selfunc(movel_out['image_list'],selnum)))
                self.TimeStamp.extend(selfunc(movel_out['time_list'],selnum))

                self.step_record_obs.append(step_n_obs_steps)

                self.HistoryFrameall.extend(movel_out['image_list'])
                self.HistoryPosall.extend(movel_out['pose_list'])
                self.HistoryForceall.extend(movel_out['force_list'])
                self.TimeStampall.extend(movel_out['time_list'])
                self.globalstep = self.globalstep + action.shape[0]
                start = time.time()

                force_sel_snapshot = copy.deepcopy(self.HistoryForce[-selnum:])
                pose_sel_snapshot = copy.deepcopy(self.HistoryPos[-selnum:])
                ts_sel_snapshot = copy.deepcopy(self.TimeStamp[-selnum:])
                step_snapshot = step_n_obs_steps
                frames_sel_snapshot = copy.deepcopy(self.HistoryFrame[-selnum:])

                force_all_snapshot = copy.deepcopy(movel_out['force_list'])
                pose_all_snapshot = copy.deepcopy(movel_out['pose_list'])
                ts_all_snapshot = copy.deepcopy(movel_out['time_list'])
                frames_all_snapshot = copy.deepcopy(movel_out['image_list'])

                async_save_snapshot(
                    self.fpforcesave, self.fpposesave, self.fptimestamp, self.fpdecisionstep,
                    self.fptimestampall, self.fpposesaveall, self.fpforcesaveall,
                    self.dpimagesave, self.dpimagesaveall,
                    force_sel_snapshot, pose_sel_snapshot, ts_sel_snapshot, step_snapshot,
                    force_all_snapshot, pose_all_snapshot, ts_all_snapshot,
                    frames_sel_snapshot, frames_all_snapshot
                )
                print('主程序保存时间：',time.time()-start)

                with torch.no_grad():
                    prob = self.classifier.predict_action({'obs':{'image':self.scalenormimg(self.HistoryFrame[-1])}})['pred']
                done = prob >0.95
                self.recordall.append({'rawimage':self.HistoryFrame[-1],'arm_pos':arm_pos,'force':self.HistoryForce[-1],'prob':prob})
                if done:
                    doneimages.append({'rawimage':self.HistoryFrame[-1],
                                        'scaleimage':self.scalenormimg(self.HistoryFrame[-1]),
                                        'doneprob':prob,
                                        'step':self.globalstep,
                                        'pose':self.HistoryPos[-1],
                                        'force':self.HistoryForce[-1]
                                        })
                    os.makedirs(self.MustSaveMarker,exist_ok=True)
                print('4 ',time.time()-start,' prob:',prob)
                print(1)
                self.theoreticalpose_quat = CalTcpMove(action = operation, original_pose_quat = self.theoreticalpose_quat)
                print('theoreticalpose_quat',self.theoreticalpose_quat)

            print('Run one step function time',time.time()-startstep)
            if os.path.exists(self.fpsavemarker) or os.path.exists(self.MustSaveMarker):
                self.fpinit = 'debugoutput'
                dpresultsaveexp = os.path.join(rf'C:\Users\{USERNAME}\Desktop\recordresult',os.path.basename(os.path.dirname(os.path.dirname(self.fpcheckpoint)))    )
                if not os.path.exists(dpresultsaveexp):
                    os.makedirs(dpresultsaveexp)
                now = datetime.datetime.now()
                date_time_str = now.strftime("%Y-%m-%d-%H-%M-%S")
                video_filename = os.path.join(dpresultsaveexp, f'{date_time_str}.mp4')
                pickle_filename = os.path.join(dpresultsaveexp,f'{date_time_str}.pkl')

                with open(pickle_filename,'wb') as f:
                    pickle.dump({'timestamp':self.TimeStampall,'rawimage':video_filename,'arm_pos':self.HistoryPosall,'force':self.HistoryForceall,'action':self.actionall},f)
                save_images_to_video(self.HistoryFrameall, video_filename, frame_rate=17)
                video_filename = os.path.join(dpresultsaveexp, f'{date_time_str}decision.mp4')
                pickle_filename = os.path.join(dpresultsaveexp,f'{date_time_str}decision.pkl')

                with open(pickle_filename,'wb') as f:
                    pickle.dump({'timestamp':self.TimeStamp,'rawimage':video_filename,'arm_pos':self.HistoryPos,'force':self.HistoryForce,'decisionstep':self.step_record_obs,'action':self.actionall},f)
                save_images_to_video(self.HistoryFrame, video_filename, frame_rate=17)
                if os.path.exists(self.fpsavemarker):
                    os.remove(self.fpsavemarker)
                if os.path.exists(self.fpsavemarkerall) or os.path.exists(self.MustSaveMarker):
                    time.sleep(30)
                    self.send_command("STOP_RECORDING")
                    os.remove(self.fpsavemarkerall)
                    import pdb
                    pdb.set_trace()
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
        self.command_queue.put(('getArmPos', None))
        while 1:
            if not self.response_queue.empty():
                arm_pos = self.response_queue.get()
                break
            else:
                time.sleep(0.01)
        return arm_pos
    def getTheoPose(self):
        self.command_queue.put(('getTheoPose', None))
        while 1:
            if not self.response_queue.empty():
                arm_pos = self.response_queue.get()
                break
            else:
                time.sleep(0.01)
        return arm_pos


    def movel_tool(self,movel_args):
        self.command_queue.put(('movel_tool', movel_args ))
        while 1:
            if not self.rdesponse_queue.empty():
                movel_out = self.response_queue.get()
                break
            else:
                time.sleep(0.01)
        return movel_out
    def movel_waypoints(self,movel_args):
        self.command_queue.put(('movel_waypoints', movel_args ))

        image_list = []
        force_list = []
        pose_list = []
        time_list = []
        while 1:
            if os.path.exists(self.MustSaveMarker):
                break

            if not self.response_queue.empty():
                movel_out = self.response_queue.get()
                if isinstance(movel_out,str):
                    break
                else:
                    image, time_stamp = self.read_image()
                    pose_list.append(movel_out['current_pose'])
                    image_list.append(image)
                    force_list.append( movel_out['force'])
                    time_list.append(time_stamp)
            else:
                time.sleep(0.01)
        print('in function stop')
        return {'image_list':image_list, 'force_list':force_list, 'pose_list':pose_list, 'time_list':time_list}

    def move_force(self,movel_args):
        self.command_queue.put(('move_force', movel_args ))

        while 1:
            if not self.response_queue.empty():
                movel_out = self.response_queue.get()
                break
            else:
                time.sleep(0.01)
        return movel_out

    def move_force_armpos(self,movel_args):
        self.command_queue.put(('move_force_armpos', movel_args ))
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
                    pose_list.append(movel_out['current_pose'])
                    image_list.append(image)
                    force_list.append(movel_out['force'])
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
        data = []
        for file in latest_files:
            filepath = os.path.join(directory, file)
            while(1):
                try:
                    with open(filepath, 'rb') as f:
                        data.append(pickle.load(f))
                        break
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
    position = tcp_pose[0:3]
    rotvec = tcp_pose[3:6]
    rotation = R.from_rotvec(rotvec)
    quat = rotation.as_quat()
    current_pose = {
        'position': position,
        'orientation': quat.tolist()
    }
    return current_pose

def quat2rec(current_pose):
    x, y, z = current_pose['position']
    quaternion = current_pose['orientation']
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