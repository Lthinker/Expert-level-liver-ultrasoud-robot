'''
Zihan Li
Acquire the image and the force together
'''
from communicate.comutil import read_float_values
from communicate.imageutil import frammask2
from communicate.posutil import PosAgent

class AcqImageForce:
    def __init__(self,client,stream,posagent):
        self.client = client # 力流
        self.stream = stream # 视频流
        self.posagent = posagent # 位姿流

    def AcqSingle(self):
        '''
            加载数据与图像后处理
        '''
        force, frame = self.AcqSingleRaw()
        frame = self.imageprocess(frame)
        pos = self.posagent.get_pos()
        return force, frame, pos

    def AcqSingleRaw(self):
        '''
            仅加载数据
        '''
        ret, frame = self.stream.read()
        force, time = read_float_values(self.client)
        return force, frame

    def forceprocess(self,force):
        '''
            处理力数据
        '''
        return force
    def imageprocess(self,frame):
        '''
            处理图像数据
        '''
        frame = frammask2(frame)
        return frame
    
    def AcqBatch(self,batchsize = 16):
        '''
            批量加载数据与图像后处理
        '''
        forces = []
        frames = []
        poses = []
        for ii in range(0,batchsize):
            force, frame, pos = self.AcqSingle()
            forces.append(force)
            frames.append(frame)
            poses.append(pos)
        return forces, frames, poses
    def close(self):
        self.stream.release()
        self.client.close()
        self.posagent.close()
        
