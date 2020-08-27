import time
import subprocess as sp
from torch.utils import data
from inpainting.davis import DAVIS
from inpainting.davis_seg import DAVISSeg
from inpainting.model import generate_model
from inpainting.utils import *

import pdb

MOVE_DIRECT_UP = -1
MOVE_DIRECT_DOWN = 1
MOVE_DIRECT_LEFT = -1
MOVE_DIRECT_RIGHT = 1

class Object():
    pass

def inpaint_seg(args):
    opt = Object()
    opt.crop_size = 512
    
    opt.double_size = True if opt.crop_size == 512 else False

    opt.search_range = 4  # fixed as 4: search range for flow subnetworks
    opt.pretrain_path = 'cp/save_agg_rec_512.pth'
    opt.result_path = 'results/inpainting'

    opt.model = 'vinet_final'
    opt.batch_norm = False
    opt.no_cuda = False  # use GPU
    opt.no_train = True
    opt.test = True
    opt.t_stride = 3
    opt.loss_on_raw = False
    opt.prev_warp = True
    opt.save_image = True
    opt.save_video = True
    roi_size = opt.crop_size*2

    img_size = [1900, 3378]

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
    # segment a frame into different spatial areas according to position of target object
    def segment(rects):
        init_rect = rects[0]
        # get the moving distance of target object 
        move_dis = rects[-1, :2] - init_rect[:2]
        direct_y = MOVE_DIRECT_RIGHT if move_dis[0] > 0 else MOVE_DIRECT_LEFT
        direct_x = MOVE_DIRECT_DOWN if move_dis[1] > 0 else MOVE_DIRECT_UP

        start_ypos = int(init_rect[0] if move_dis[0] > 0 else init_rect[0]+init_rect[2])
        start_xpos = int(init_rect[1] if move_dis[1] > 0 else init_rect[1]+init_rect[3])

        target_end_yposs  = rects[:,0] + rects[:,2] if move_dis[0] > 0 else rects[:,0]
        target_end_xposs  = rects[:,1] + rects[:,3] if move_dis[0] > 0 else rects[:,1]

        target_start_yposs  = rects[:,0] + rects[:,2] if move_dis[0] > 0 else rects[:,0]
        target_start_xposs  = rects[:,1] + rects[:,3] if move_dis[0] > 0 else rects[:,1]

        segs_area_list = []
        seg_idxs = []


        for i in range(rects.shape[0]):
            # make sure the target is still in the current roi area
            if abs(target_end_xposs[i] - start_xpos) < roi_size and abs(target_end_yposs[i] - start_ypos) < roi_size:
                seg_idxs.append(i)
                continue
            else:
                _startx = start_xpos if direct_x>0 else (0 if start_xpos-roi_size<0 else start_xpos-roi_size)
                _starty = start_ypos if direct_y>0 else (0 if start_ypos-roi_size<0 else start_ypos-roi_size)
                _endx   = start_xpos if direct_x<0 else (img_size[0] if start_xpos+roi_size > img_size[0] else start_xpos+roi_size )
                _endy   = start_ypos if direct_y<0 else (img_size[1] if start_ypos+roi_size > img_size[1] else start_ypos+roi_size )
                segs_area_list.append((seg_idxs,[_startx, _starty, _endx, _endy]))
                seg_idxs = []
                start_xpos = target_start_xposs[i]
                start_ypos = target_start_yposs[i]
        
        _startx = start_xpos if direct_x>0 else (0 if start_xpos-roi_size<0 else start_xpos-roi_size)
        _starty = start_ypos if direct_y>0 else (0 if start_ypos-roi_size<0 else start_ypos-roi_size)
        _endx   = start_xpos if direct_x<0 else (img_size[0] if start_xpos+roi_size > img_size[0] else start_xpos+roi_size )
        _endy   = start_ypos if direct_y<0 else (img_size[1] if start_ypos+roi_size > img_size[1] else start_ypos+roi_size )

        segs_area_list.append((seg_idxs,[_startx, _starty, _endx, _endy]))
        return segs_area_list
    segs_areas_list = segment(rects)
    ########## DAVIS
    DAVIS_ROOT =os.path.join('results', args.data)
    DTset = DAVISSeg(DAVIS_ROOT, mask_dilation=args.mask_dilation, segs_areas_list=segs_areas_list, size=(opt.crop_size, opt.crop_size))
    DTloader = data.DataLoader(DTset, batch_size=1, shuffle=False, num_workers=1)

    def to_img(x):
        tmp = (x[0, :, 0, :, :].cpu().data.numpy().transpose((1, 2, 0)) + 1) / 2
        tmp = np.clip(tmp, 0, 1) * 255.
        return tmp.astype(np.uint8)

    model, _ = generate_model(opt)
    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    model.eval()
    ts = opt.t_stride
    # folder_name = 'davis_%d' % (int(opt.crop_size))
    pre = 15
    with torch.no_grad():
        for seq, (origin_imgs, inputs, masks, areas_list, info) in enumerate(DTloader):
            
            # in_frame = to_img(inputs)
            # cv2.imwrite('in_frame.jpg',in_frame )
            # in_mask = to_img(masks)
            # cv2.imwrite('in_mask.jpg',in_mask )

            print('processing sequence {}'.format(str(seq)))
            start_x, start_y, end_x, end_y = areas_list
            idx = torch.LongTensor([i for i in range(pre - 1, -1, -1)])
            pre_inputs = inputs[:, :, :pre].index_select(2, idx)
            pre_masks = masks[:, :, :pre].index_select(2, idx)
            inputs = torch.cat((pre_inputs, inputs), 2)
            masks = torch.cat((pre_masks, masks), 2)

            bs = inputs.size(0)
            num_frames = inputs.size(2)
            seq_name = info['name'][0]

            save_path = os.path.join(opt.result_path, seq_name)
            if not os.path.exists(save_path) and opt.save_image:
                os.makedirs(save_path)

            inputs = 2. * inputs - 1
            inverse_masks = 1 - masks
            masked_inputs = inputs.clone() * inverse_masks

            masks = to_var(masks)
            masked_inputs = to_var(masked_inputs)
            inputs = to_var(inputs)

            total_time = 0.
            in_frames = []
            out_frames = []

            lstm_state = None
            print('num_frames: {}'.format(str(num_frames)))
            print('len of origin images {}'.format(str(len(origin_imgs))))
            for t in range(num_frames):
                masked_inputs_ = []
                masks_ = []

                if t < 2 * ts:
                    masked_inputs_.append(masked_inputs[0, :, abs(t - 2 * ts)])
                    masked_inputs_.append(masked_inputs[0, :, abs(t - 1 * ts)])
                    masked_inputs_.append(masked_inputs[0, :, t])
                    masked_inputs_.append(masked_inputs[0, :, t + 1 * ts])
                    masked_inputs_.append(masked_inputs[0, :, t + 2 * ts])
                    masks_.append(masks[0, :, abs(t - 2 * ts)])
                    masks_.append(masks[0, :, abs(t - 1 * ts)])
                    masks_.append(masks[0, :, t])
                    masks_.append(masks[0, :, t + 1 * ts])
                    masks_.append(masks[0, :, t + 2 * ts])
                elif t > num_frames - 2 * ts - 1:
                    masked_inputs_.append(masked_inputs[0, :, t - 2 * ts])
                    masked_inputs_.append(masked_inputs[0, :, t - 1 * ts])
                    masked_inputs_.append(masked_inputs[0, :, t])
                    masked_inputs_.append(masked_inputs[0, :, -1 - abs(num_frames - 1 - t - 1 * ts)])
                    masked_inputs_.append(masked_inputs[0, :, -1 - abs(num_frames - 1 - t - 2 * ts)])
                    masks_.append(masks[0, :, t - 2 * ts])
                    masks_.append(masks[0, :, t - 1 * ts])
                    masks_.append(masks[0, :, t])
                    masks_.append(masks[0, :, -1 - abs(num_frames - 1 - t - 1 * ts)])
                    masks_.append(masks[0, :, -1 - abs(num_frames - 1 - t - 2 * ts)])
                else:
                    masked_inputs_.append(masked_inputs[0, :, t - 2 * ts])
                    masked_inputs_.append(masked_inputs[0, :, t - 1 * ts])
                    masked_inputs_.append(masked_inputs[0, :, t])
                    masked_inputs_.append(masked_inputs[0, :, t + 1 * ts])
                    masked_inputs_.append(masked_inputs[0, :, t + 2 * ts])
                    masks_.append(masks[0, :, t - 2 * ts])
                    masks_.append(masks[0, :, t - 1 * ts])
                    masks_.append(masks[0, :, t])
                    masks_.append(masks[0, :, t + 1 * ts])
                    masks_.append(masks[0, :, t + 2 * ts])

                masked_inputs_ = torch.stack(masked_inputs_).permute(1, 0, 2, 3).unsqueeze(0)
                masks_ = torch.stack(masks_).permute(1, 0, 2, 3).unsqueeze(0)

                start = time.time()
                if not opt.double_size:
                    prev_mask_ = to_var(torch.zeros(masks_[:, :, 2].size()))  # rec given when 256
                prev_mask = masks_[:, :, 2] if t == 0 else prev_mask_
                prev_ones = to_var(torch.ones(prev_mask.size()))
                prev_feed = torch.cat([masked_inputs_[:, :, 2, :, :], prev_ones, prev_ones * prev_mask],
                                      dim=1) if t == 0 else torch.cat(
                    [outputs.detach().squeeze(2), prev_ones, prev_ones * prev_mask], dim=1)

                outputs, _, _, _, _ = model(masked_inputs_, masks_, lstm_state, prev_feed, t)
                if opt.double_size:
                    prev_mask_ = masks_[:, :, 2] * 0.5  # rec given whtn 512

                lstm_state = None
                end = time.time() - start
                if lstm_state is not None:
                    lstm_state = repackage_hidden(lstm_state)

                total_time += end
                if t > pre:
                    print('{}th frame of {} is being processed'.format(t - pre, seq_name))
                    out_frame = to_img(outputs)
                    origin_img = origin_imgs[t-pre][0,:,:,:].data.numpy()
                    out_frame = cv2.resize(out_frame, (int(end_y)-int(start_y), int(end_x)-int(start_x)))
                    origin_img[start_x:end_x, start_y:end_y] = out_frame
                    out_frame = origin_img
                    if args.visualization:
                        cv2.imshow('Inpainting', origin_img)
                        key = cv2.waitKey(1)
                        if key > 0:
                            break
                    if opt.save_image:
                        cv2.imwrite(os.path.join(save_path, '%05d.png' % (t - pre)), origin_img)
                    out_frames.append(origin_img[:, :, ::-1])
        if opt.save_video:
            final_clip = np.stack(out_frames)
            video_path = opt.result_path
            if not os.path.exists(video_path):
                os.makedirs(video_path)

            createVideoClip(final_clip, video_path, '%s.mp4' % (seq_name), [DTset.shape[0], DTset.shape[1]])
            print('Predicted video clip saving')
        if args.visualization:
            cv2.destroyAllWindows()

