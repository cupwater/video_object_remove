import time
import subprocess as sp
from torch.utils import data
import cv2
import numpy as np
import os
import glob
from scipy import ndimage, signal
import pdb

class Object():
    pass

def stick_background(args):    
    opt = Object()
    opt.search_range = 4  # fixed as 4: search range for flow subnetworks
    opt.result_path = 'results/inpainting'
    img_size = [1900, 3378]
    opt.save_image = True
    opt.save_video = True
    background = cv2.imread('data/background.jpg')
    def createVideoClip(clip, folder, name, size=[256, 256]):

        vf = clip.shape[0]
        command = ['ffmpeg',
                   '-y',  # overwrite output file if it exists
                   '-f', 'rawvideo',
                   '-s', '%dx%d' % (size[1], size[0]),  # '256x256', # size of one frame
                   '-pix_fmt', 'rgb24',
                   '-r', '25',  # frames per second
                   '-an',  # Tells FFMPEG not to expect any audio
                   '-i', '-',  # The input comes from a pipe
                   '-vcodec', 'libx264',
                   '-b:v', '1500k',
                   '-vframes', str(vf),  # 5*25
                   '-s', '%dx%d' % (size[1], size[0]),  # '256x256', # size of one frame
                   folder + '/' + name]
        # sfolder+'/'+name
        pipe = sp.Popen(command, stdin=sp.PIPE, stderr=sp.PIPE)
        out, err = pipe.communicate(clip.tostring())
        pipe.wait()
        pipe.terminate()
        print(err)
    rects = np.loadtxt( os.path.join(args.data, 'rects.txt') ).astype(np.int32)
    DAVIS_ROOT =os.path.join('results', args.data)
    img_root  = os.path.join(DAVIS_ROOT + '_frame')
    mask_root = os.path.join(DAVIS_ROOT + '_mask')
    num_frames = len(glob.glob(os.path.join(img_root, '*.jpg')))
    save_path = os.path.join(opt.result_path, args.data.split('/')[-1])

    if not os.path.exists(save_path) and opt.save_image:
        os.makedirs(save_path)

    out_frames = []
    for i in range(num_frames):
        try:
            mask_file = os.path.join(mask_root, '{:05d}.png'.format(i))
            mask = cv2.imread(mask_file).astype(np.uint8)
        except:
            mask_file = os.path.join(mask_root, '00000.png')
            mask = cv2.imread(mask_file).astype(np.uint8)
        mask = mask[:,:,0]
        # expand 
        w_k = np.ones((10, 6))
        mask2 = signal.convolve2d(mask.astype(np.float), w_k, 'same')
        mask2 = 1 - (mask2 == 0)
        mask_ = np.float32(mask2)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (args.mask_dilation, args.mask_dilation))
        mask = cv2.dilate(mask_, kernel)
        mask = mask.reshape(mask.shape[0], -1, 1)
        mask = mask.repeat(3, axis=2)

        img_file = os.path.join(img_root, '{:05d}.jpg'.format(i))
        img = cv2.imread(img_file)
        inverse_masks = 1 - mask
        stick_img = img.copy() * inverse_masks + background * mask

        if i%50==0:
            print('{}th frame of {} is being processed'.format(str(i), str(num_frames)))
        
        if args.visualization:
            cv2.imshow('Inpainting', stick_img)
            key = cv2.waitKey(1)
            if key > 0:
                break
        if opt.save_image:
            cv2.imwrite(os.path.join(save_path, '%05d.png' % (i+1)), stick_img)
        out_frames.append(stick_img)

    if opt.save_video:
        final_clip = np.stack(out_frames)
        video_path = opt.result_path
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        createVideoClip(final_clip, video_path, '%s.mp4' % (seq_name), [DTset.shape[0], DTset.shape[1]])
        print('Predicted video clip saving')
    if args.visualization:
        cv2.destroyAllWindows()

