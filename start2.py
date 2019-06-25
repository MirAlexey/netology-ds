from matplotlib import pyplot as plt
import numpy as np
import cv2
import argparse
import datetime
import imutils
import time
from  PIL import Image
import sys
import os
sys_paths = ['../']
for p in sys_paths:
    p = os.path.abspath(p)
    if p not in sys.path:
        sys.path.append(p)

import time
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from matplotlib import patches, patheffects
from torch.autograd import Variable
from torchvision.transforms import ToTensor
import utils.util as util
import data.util as data_util
import models.modules.EDVR_arch as EDVR_arch


from yolov3_pytorch.utils import *
from yolov3_pytorch.yolov3 import *
from yolov3_pytorch.yolov3_tiny import *

from dface.core.detect import create_mtcnn_net, MtcnnDetector
import dface.core.vision as vision

import multiprocessing as mp
#import dlib
from scipy import spatial
import pickle

def take_pred3(queue_in, queue_out):

    #model = torch.load(args.model, map_location=lambda storage, loc: storage)
    #model = model.cuda()


    model = torch.load('model_epoch_150_1.pth')
    model = model.cuda()
    while True:
        if not queue_in.empty():
            img_l = queue_in.get()
            img_yuv  = cv2.cvtColor(img_l, cv2.COLOR_BGR2YUV)
            cb, cr, y = img_yuv[:, :, 2], img_yuv[:, :, 1], img_yuv[:, :, 0]
            data = (ToTensor()(y)).view(1, -1, y.size[1], y.size[0])
            data = data.cuda
            out = model(data)
            out = out.cpu()
            out_img_y = out.data[0].numpy()
            out_img_y *= 255.0
            out_img_y = out_img_y.clip(0, 255)
            out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')
            out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
            out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
            out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')
            queue_out.put(np.array(out_img))


def single_forward(model, imgs):
    with torch.no_grad():
        model_output = model(imgs)
        if isinstance(model_output, list) or isinstance(model_output, tuple):
            output = model_output[0]
        else:
            output = model_output
    return output

def take_pred2(queue_in, queue_out):
    #print('_in_2')
    model_path = 'experiments/pretrained_models/EDVR_Vimeo90K_SR_L.pth'
    model = EDVR_arch.EDVR(128, 7, 8, 5, 40, predeblur=False, HR_in=False)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.cuda()
    while True:
        if not queue_in.empty():
            img_l = queue_in.get()
            imgs = img_l[:, :, :, :, [2, 1, 0]]
            imgs = torch.from_numpy(np.ascontiguousarray(np.transpose(imgs, (0, 1, 4, 2, 3)))).float()
            imgs = imgs.type('torch.cuda.FloatTensor')
            imgs = torch.cat((imgs,  torch.flip(imgs, (-2, -1))), -1)
            output = single_forward(model, imgs)
            out_1, out_2 = torch.split(output, 336, -1)
            out = (out_1 + torch.flip(out_2, (-2, -1)))/2
            out1 = out.permute(0, 2, 3, 1)
            out1 = out1[:,:,:,[2,1,0]].cpu().numpy()
            out1 = np.squeeze(out1, axis=0)
            queue_out.put(out1)

def take_pred1(queue_in, queue_out):
    # используем обычные методы бикубической интерполяции
    kernel = np.array([[-1, -2, -1], [-2, 22, -2], [-1, -2, -1]]) / 10
    while True:
        if not queue_in.empty():
            img_l = queue_in.get()
            sx_x = int(img_l.shape[1]*3)
            sx_y = int(img_l.shape[0]*3)
            out = cv2.resize(img_l, (sx_x, sx_y), interpolation=cv2.INTER_CUBIC)
            out = cv2.filter2D(out, -1, kernel)
            outl = np.asarray(out)
            queue_out.put(outl)


def take_pred(queue_in, queue_out):
    print('into')
    n = 0

    model = Yolov3(num_classes=80)
    model.load_state_dict(torch.load('data/models/yolov3_coco_01.h5'))
    model = model.cuda()

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    kernel = np.array([[-1, -2, -1], [-2, 22, -2], [-1, -2, -1]]) / 10

    while True:
        if not queue_in.empty():
            n+=1
            data = queue_in.get()
            if type(data) == type('s'):

                print('output')
                break
            sz = 416

            img_bg = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
            img_yuv = cv2.cvtColor(img_bg, cv2.COLOR_BGR2YUV)

            img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])

            frame = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
            frame = cv2.filter2D(frame, -1, kernel)
            img_resized = cv2.resize(frame, (sz, sz))

            img_torch = torch.from_numpy(img_resized/255)
            img_torch = img_torch[np.newaxis, :]
            img_torch = img_torch.permute(0, 3, 1, 2)

            img_torch = img_torch.type('torch.cuda.FloatTensor')
            all_boxes = model.predict_img(img_torch, conf_thresh=0.3)[0]

            queue_out.put(all_boxes)


# def filter_box(new_box, tracker):
#
#         pos = tracker.get_position()
#         t_startX = int(pos.left())
#         t_startY = int(pos.top())
#         t_endX = int(pos.right())
#         t_endY = int(pos.bottom())
#         b_startX = int(new_box[0])
#         b_startY = int(new_box[1])
#         b_endX = int(new_box[2])
#         b_endY = int(new_box[3])
#         #x1, y1, x2, y2
#         #print('t',t_startX,t_startY,t_endX,t_endY)
#         #print('b',b_startX,b_startY,b_endX,b_endY)
#
#         dx = (min(t_endX, b_endX) - max(t_startX, b_startX))
#         dy = (min(t_endY, b_endY) - max(t_startY, b_startY))
#
#         intersection = dx * dy if (dx > 0 and dy > 0) else 0.
#
#         anch_square = (b_endX - b_startX) * (b_endY - b_startY)
#         rect_square = (t_endX - t_startX) * (t_endY - t_startY)
#         union = anch_square + rect_square - intersection
#         return (intersection / union) > 0.7




if __name__ == '__main__':

    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass




    #cap = cv2.VideoCapture('/home/user/netology/netology_git/data_diplom/data_video_1/Шаман 9 марта 2015-1080.mp4')
    #cap = cv2.VideoCapture('/home/user/netology/netology_git/data_diplom/data_obraz_1/TownCentreXVID.avi')
    #cap = cv2.VideoCapture('/home/user/netology/netology_git/data_diplom/data_video_1/Ёлочная 2017-1080.mp4')
    #cap = cv2.VideoCapture('/home/user/netology/netology_git/data_diplom/data_obraz_1/obraz1.mp4')
    #cap = cv2.VideoCapture('/home/user/netology/netology_git/data_diplom/data_obraz_1/obraz2.mp4')
    #cap = cv2.VideoCapture('/home/user/netology/netology_git/data_diplom/data_obraz_1/obraz3.mp4')
    #cap = cv2.VideoCapture('/home/user/netology/netology_git/data_diplom/data_obraz_1/obraz3.mp4.vob')
    #cap = cv2.VideoCapture('/home/user/netology/netology_git/data_diplom/data_obraz_1/obraz7.mp4')
    #cap = cv2.VideoCapture('/home/user/netology/netology_git/data_diplom/data_obraz_1/Vtorzhenije.Pohititelej.Tel.1956.XviD.DVDRip.Korsar.avi')
    cap = cv2.VideoCapture('/home/user/netology/netology_git/data_diplom/data_obraz_1/TownCentreXVID.avi.vob')

    #cv2.VideoCapture.set( cv2.CV_CAP_PROP_FPS, 30 )





    ret, frame = cap.read()
    if not ret:
        print('Ой')
    scale_percent = 20

    sh = frame.shape
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    scale_factor_optfl = scale_percent / 100
    x_sc = frame.shape[1]
    y_sc = frame.shape[0]
    clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8,8))
    kernel = np.array([[-1,-2,-1],[-2,22,-2],[-1,-2,-1]])/10
    all_boxes = []
    boxes2 = []
    array_of_frame = np.zeros([30, frame.shape[0], frame.shape[1], frame.shape[2]])
    dict_of_pred = {}
    head_frame = 0
    tail_frame = 0
    current_frame = 0
    array_of_frame[head_frame] = cv2.filter2D(frame, -1, kernel)
    for i in range(15):
        ret, frame = cap.read()
        if not ret:
            break
        head_frame = (head_frame + 1) % 30
        array_of_frame[head_frame] = frame.copy()

    q_in1 = mp.Queue()
    q_out1 = mp.Queue()
    q_in = mp.Queue()
    q_out = mp.Queue()

    p1 = mp.Process(target=take_pred, args=( q_out, q_in))
    p1.start()
    p2 = mp.Process(target=take_pred2, args=(q_out1, q_in1))
    p2.start()


    flag_set_tracker_point = False
    set_tracker_point = [0,0]

    def mouse_drawing(event, x, y, flags, params):
        global set_tracker_point
        global flag_set_tracker_point
        print('event',event)
        if event == cv2.EVENT_LBUTTONDOWN:

            set_tracker_point = [x,y]
            flag_set_tracker_point = True

    cv2.namedWindow("frame")
    cv2.setMouseCallback("frame", mouse_drawing)

    # tracker
    # [0:6] - x, y, w, h, flow_x, flow_y координаты, размеры и проекции вектора оптического потока трекера
    # [6:12]- x, y, w, h, flow_X, flow_y координаты, размеры и проекции вектора оптического потока предсказанного бокса
    # [12]  - статус текущего отслеживаемого трекера
    # [13]  - время жизни без привязки к новому боксу для трекера в кадрах (задается параметром time_life_box)
    time_life_box = 5.0
    #                      0   1   2   3   4   5   6   7   8   9   10  11   12  13
    trackers = np.array([[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0, 0.1, time_life_box]])
    status_frame = True
    status_crop = True
    frame_crop = None
    frame_under_crop = None
    frame_under_crop_out = None
    view_tracker = 0
    last_cur_tr_point = [0,0]
    while(cap.isOpened()):
        cur_time = time.clock()
        ret, frame = cap.read()
        if not ret:
            break
        head_frame = (head_frame + 1) % 30

        array_of_frame[head_frame] = frame.copy()

        if status_frame:
            #print('in')
            status_frame = False
            q_out.put(frame)
            current_frame = head_frame

        if not q_in.empty():
            peoples = q_in.get()
            faces = []
            dict_of_pred[current_frame] = {'peoples': peoples, 'faces': faces}
            status_frame = True

        frame = array_of_frame[tail_frame]/255

        if tail_frame in dict_of_pred.keys():
            all_boxes = dict_of_pred[tail_frame]['peoples']
            boxes2 = dict_of_pred[tail_frame]['faces']
            del dict_of_pred[tail_frame]

        num_update_tr = set([])
        if len(all_boxes) > 0 or len(boxes2) > 0:
            curr_frame = cv2.resize(cv2.cvtColor(array_of_frame[tail_frame].astype('uint8'), cv2.COLOR_BGR2GRAY), (width, height))
            next_frame = cv2.resize(cv2.cvtColor(array_of_frame[(tail_frame+1)%30].astype('uint8'), cv2.COLOR_BGR2GRAY), (width, height))
            flow = cv2.calcOpticalFlowFarneback(curr_frame, next_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            flow_x = flow[..., 0]+1e-10
            flow_y = flow[..., 1]+1e-10
            tr_tree = spatial.KDTree(trackers[:,6:8])
            if len(all_boxes) > 0:
                all_boxes = nms(all_boxes, 0.4)
                print('box and tracker',len(all_boxes) , len(trackers))
                for x1, y1, x2, y2, conf, cls_conf, cls_pred in all_boxes[::-1]:
                    if x1>2 or x2>2 or y1>2 or y2>2 or cls_pred != 0:
                        continue
                    x, y, w, h = int(x1*x_sc), int(y1*y_sc), int(x2*x_sc), int(y2*y_sc)

                    xs, ys, xf, yf = max(0,int((x - w / 2)*scale_factor_optfl)), max(0,int((y - h / 2)*scale_factor_optfl)), max(1,int((x + w / 2)*scale_factor_optfl)), max(1,int((y + h / 2)*scale_factor_optfl))
                    cur_flow_x = ((np.mean(flow_x[ys:yf, xs:xf]))+1e-10)*5
                    cur_flow_y = ((np.mean(flow_y[ys:yf, xs:xf]))+1e-10)*5

                    ans = tr_tree.query([x,y], k=10, distance_upper_bound=30)
                    ans_l = ans[0][ans[0] < 30]
                    ans_n = ans[1][:len(ans_l)]
                    cur_tr = -1
                    if len(ans_n) > 0:
                        if len(ans_l) == 1:
                            cur_tr = ans_n[0]
                        else:
                            tr_flow_tree = spatial.KDTree(trackers[ans_n, 4:6])
                            ans = tr_tree.query([cur_flow_x, cur_flow_y],k=2, distance_upper_bound=(abs(cur_flow_x) + abs(cur_flow_y)))
                            ans_l = ans[0][ans[0] < 30]
                            ans_n = ans[1][:len(ans_l)]
                            if len(ans_n) > 0:
                                cur_tr = ans_n[0]
                    if cur_tr < 0:
                        #print('add tr')
                        trackers = np.vstack((trackers, np.array([x,y,w,h,cur_flow_x, cur_flow_y,x,y,w,h,cur_flow_x, cur_flow_y,0.1, time_life_box])))

                    else:
                        trackers[cur_tr, 6:12] = np.array([x, y, w, h, cur_flow_x, cur_flow_y])
                        trackers[cur_tr, 13] = time_life_box
                        num_update_tr.add(cur_tr)
                        if trackers[cur_tr, 12] > 1:
                            trackers[cur_tr, 13] = time_life_box*2

            all_boxes = []
            boxes2 = []
        trackers[:, [0,1]] = trackers[:, [0,1]] + trackers[:, [4,5]] + (trackers[:, [6,7]] - trackers[:, [0,1]])/5
        trackers[:, [2,3]] = trackers[:, [2,3]] + (trackers[:, [8,9]] - trackers[:, [2,3]]) / 5
        trackers[:, [4,5]] = trackers[:, [4,5]] + (trackers[:, [10,11]] - trackers[:, [4,5]]) / 5
        trackers[:, [6,7]] = trackers[:, [6,7]] + trackers[:, [10,11]]
        trackers[:, 13] = trackers[:, 13] - 1
        trackers = trackers[trackers[:, 13] > -1]
        if len(trackers) > 0:
            tr_tree = spatial.KDTree(trackers[:, 0:2])
        else:
            tr_tree = None

        hot_tr = trackers[trackers[:, 12] > 1]

        if (len(hot_tr) == 0) or (hot_tr[0,13] < 0) or flag_set_tracker_point:
            if len(trackers) > 0:
                ground_point = set_tracker_point if flag_set_tracker_point else last_cur_tr_point
                if tr_tree is not None:
                    flag_set_tracker_point = False
                    ans = tr_tree.query(ground_point)
                    trackers[:, 12] = 0.1
                    trackers[ans[1], 12] = 1.1
                    trackers[ans[1], 13] = time_life_box*2
                else:
                    trackers[0, 12] = 1.1
                    trackers[0, 13] = 10
            else:
                trackers = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, time_life_box]])
        else:
            last_cur_tr_point = hot_tr[0,0:2]

        if len(trackers) > 0:
            for tr in trackers:
                t_X = int(tr[6] + 1e-10)
                t_Y = int(tr[7] + 1e-10)
                t_W = int((tr[8] + 1e-10) / 2)
                t_H = int((tr[9] + 1e-10) / 2)
                color = (0, 0, 0)
                weight = 1
                cv2.rectangle(frame, (t_X - t_W, t_Y - t_H), (t_X + t_W, t_Y + t_H), color, weight)
                t_X = int(tr[0] + 1e-10)
                t_Y = int(tr[1] + 1e-10)
                t_W = int((tr[2] + 1e-10) / 2)
                t_H = int((tr[3] + 1e-10) / 2)
                t_fx = int(tr[4]*7 + 1e-10)
                t_fy = int(tr[5]*7 + 1e-10)
                color = (0, 255, 0) if tr[12] < 1 else (255, 0, 0)
                weight = 2 if tr[13] > 2 else 1
                cv2.rectangle(frame, (t_X - t_W, t_Y - t_H), (t_X + t_W, t_Y + t_H), color, weight)
                cv2.line(frame,(t_X,t_Y-25), (t_X+t_fx,t_Y-25+t_fy), (0,255,0), 1)
                cv2.circle(frame, (t_X,t_Y-25), 3 ,(0,255,0))
                cv2.circle(frame, (t_X+t_fx,t_Y-25+t_fy), 1, (0, 255, 0))

            if status_crop:
                pos = trackers[trackers[:,12] > 1]
                if len(pos) > 0:
                    pos = pos[0]
                    if pos[0] != 0.0 and pos[1] != 0.0:
                        shape = frame.shape
                        t_startX = max(0,int(pos[0] - 42))
                        t_startY = max(0,int(pos[1] - 82))
                        t_endX = int(t_startX + 84)
                        t_endY = int(t_startY + 104)
                        if t_endX > shape[1]:
                            t_startX,t_endX = shape[1]-84, shape[1]
                        if t_endY > shape[0]:
                            t_startY,t_endY = shape[0]-104, shape[0]

                        frame_start = (tail_frame + 1) % 30
                        frame_finish = (tail_frame + 8) % 30
                        frame_under_crop = array_of_frame[(tail_frame + 4)%30, t_startY:t_endY, t_startX:t_endX] / 255
                        if frame_start < frame_finish:
                            frames = array_of_frame[frame_start:frame_finish, t_startY:t_endY, t_startX:t_endX] / 255
                        else:
                            frames = np.vstack((array_of_frame[frame_start:30, t_startY:t_endY, t_startX:t_endX], array_of_frame[0:frame_finish, t_startY:t_endY, t_startX:t_endX]))/255
                        frames = np.expand_dims(frames, axis=0)
                        q_out1.put(frames)
                        status_crop = False

            if not q_in1.empty():
                frame_crop = q_in1.get()
                sx_x, sx_y, sx_c = frame_crop.shape
                out2 = cv2.resize(frame_under_crop, (sx_y, sx_x), interpolation=cv2.INTER_CUBIC)
                out2 = cv2.filter2D(out2, -1, kernel)
                frame_crop = np.hstack((frame_crop, out2))
                cv2.imshow('crop', frame_crop)
                status_crop = True
        if (tail_frame+1)%30 != head_frame:
            tail_frame = (tail_frame+1)%30
        cv2.imshow('frame',frame)
        delta_time = time.clock()-cur_time

        if cv2.waitKey(int(max(1.0, 30-delta_time*1000))) & 0xFF == ord('q'):
            q_out.put('q')
            break

    p1.join()
    cap.release()
    cv2.destroyAllWindows()
