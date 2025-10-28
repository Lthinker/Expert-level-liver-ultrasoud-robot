import cv2
from matplotlib import pyplot as plt
import time
import asyncio
import numpy as np

class VideoIO:
    def __init__(self,port=0):
        self.stream = cv2.VideoCapture(port)
        if not self.stream.isOpened():
            print("Error: Could not open video stream.")
        else:
            print("Video stream opened successfully.")
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))  # 读取视频格式
        # 设置分辨率
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        fps = self.stream.get(cv2.CAP_PROP_FPS)

def high_speed_image(stream,duration = 10):
    start = time.time()
    frames = []
    delt = []
    while(time.time()-start < duration):
        bt = time.time()
        ret, frame = stream.read()
        frames.append(frame)
        et = time.time()
        delt.append(et-bt)
        time.sleep(0.01)
    return frames
def frammask_nogreen(frame): 
    x0 = 120
    x1 = 900
    y0 = 320
    y1 = 1380

    cropframe = frame[x0:x1,y0:y1,:]
    # return cropframe
    mmask = np.ones([cropframe.shape[0],cropframe.shape[1],3],dtype=np.uint8)
    mmask[0:50,340:400,:] = 0
    barmask = np.ones([cropframe.shape[0],cropframe.shape[1],3],dtype=np.uint8)
    barmask[:300,850:,:] = 0
    cropframe_mmask = cropframe * mmask
    cropframe_mmask_barmask = cropframe_mmask * barmask
    return cropframe_mmask_barmask
def frammask2(frame):
    x0 = 120
    x1 = 900
    y0 = 320
    y1 = 1400

    cropframe = frame[x0:x1,y0:y1,:]
    # return cropframe
    mmask = np.ones([cropframe.shape[0],cropframe.shape[1],3],dtype=np.uint8)
    mmask[0:50,340:400,:] = 0
    barmask = np.ones([cropframe.shape[0],cropframe.shape[1],3],dtype=np.uint8)
    # barmask[:300,850:,:] = 0
    barmask[:300,1000:,:] = 0
    cropframe_mmask = cropframe * mmask
    cropframe_mmask_barmask = cropframe_mmask * barmask
    return cropframe_mmask_barmask

def frammask3(frame):
    x0 = 130
    x1 = 760
    y0 = 880
    y1 = 1640

    cropframe = frame[x0:x1,y0:y1,:]
    mmask = np.ones([cropframe.shape[0],cropframe.shape[1],3],dtype=np.uint8)
    mmask[0:57,200:250,:] = 0
    barmask = np.ones([cropframe.shape[0],cropframe.shape[1],3],dtype=np.uint8)
    cropframe_mmask = cropframe * mmask
    cropframe_mmask_barmask = cropframe_mmask * barmask
    return cropframe_mmask_barmask

MRGreenBar_580_803_at790 =  np.load('MRGreenBar_580_803_at790.npy')

def frammask4(frame):
    nx = 150
    ny = 840
    cropframe = frame[nx:nx+580,ny:ny+803]

    mmask = np.ones(cropframe.shape,dtype=np.uint8)
    mmask[0:45,250:290] = 0
    addbar = MRGreenBar_580_803_at790
    cropframe[:,790:] = addbar
    cropframe_mask = cropframe * mmask
    return cropframe_mask

def frammask5(frame):
    nx = 150
    ny = 880+20
    cropframe = frame[nx:nx+580,ny:ny+750]
    mmask = np.ones(cropframe.shape,dtype=np.uint8)
    mmask[0:45,200:245] = 0

    cropframe_mask = cropframe * mmask
    return cropframe_mask
