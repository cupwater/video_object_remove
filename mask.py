# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import os
import glob
from get_mask.test import *
from get_mask.models.custom import Custom


def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or \
        video_name.endswith('mp4'):
        cap = cv2.VideoCapture(video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob.glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame


def mask(args):
    # Setup device
    args.config = 'get_mask/experiments/siammask/config_davis.json'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # Setup Model
    cfg = load_config(args)
    siammask = Custom(anchors=cfg['anchors'])
    if args.resume:
        assert isfile(args.resume), '{} is not a valid file'.format(args.resume)
        siammask = load_pretrain(siammask, args.resume)
    
    siamask = siammask.cuda()
    siammask.eval()

    # Parse Image file
    # img_files = sorted(glob.glob(join(args.base_path, '*.jp*')))
    img_files = get_frames(args.data)
    ims = [imf for imf in img_files]

    if args.visualization:
        # Select ROI
        cv2.namedWindow("Get_mask", cv2.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty("SiamMask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        try:
            init_rect = cv2.selectROI('Get_mask', ims[0], False, False)
            x, y, w, h = init_rect
        except:
            exit()
    else:
        # the init_rect should be provided in argparse, the rect should be 'x,y,w,h'
        init_rect = args.init_rect
        x, y, w, h =  [int(v) for v in init_rect.split(',')]

    toc = 0
    counter = 0

    if not os.path.exists(os.path.join('results', '{}_mask'.format(args.data))):
        os.makedirs(os.path.join('results', '{}_mask'.format(args.data)))
        os.makedirs(os.path.join('results', '{}_frame'.format(args.data)))

    rects = []
    for f, im in enumerate(ims):
        tic = cv2.getTickCount()
        if f == 0:  # init
            target_pos = np.array([x + w / 2, y + h / 2])
            target_sz = np.array([w, h])
            state = siamese_init(im, target_pos, target_sz, siammask, cfg['hp'])  # init tracker
        elif f > 0:  # tracking
            state = siamese_track(state, im, mask_enable=True, refine_enable=True)  # track
            # save the position of traget object
            pos = list(state['target_pos']) + list(state['target_sz'])
            # rect records the left=-up and right-down position
            rect = [pos[0] - 1.1*pos[2]/2, pos[1] - 1.1*pos[3]/2, 1.2*pos[2], 1.2*pos[3]]
            rects = rects + rect
            location = state['ploygon'].flatten()
            mask = state['mask'] > state['p'].seg_thr
            mask = (mask * 255.).astype(np.uint8)
            cv2.imwrite('results/{}_mask/{:05d}.png'.format(args.data, counter), mask)
            cv2.imwrite('results/{}_frame/{:05d}.jpg'.format(args.data, counter), im)
            counter += 1

            im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]
            if args.visualization:
                cv2.polylines(im, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
                cv2.imshow('Get_mask', im)
                key = cv2.waitKey(1)
                if key > 0:
                    break

        toc += cv2.getTickCount() - tic
    
    rects = np.array(rects).reshape(-1, 4)
    np.savetxt(os.path.join(args.data, 'rects.txt'), rects, fmt='%d')

    toc /= cv2.getTickFrequency()
    fps = f / toc
    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visualization!)'.format(toc, fps))
    if args.visualization:
        cv2.destroyAllWindows()

