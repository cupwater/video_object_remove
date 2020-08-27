import argparse
from mask import mask
from inpaint import inpaint
from inpaint_segment import inpaint_seg
from stick_background import stick_background


parser = argparse.ArgumentParser(description='Demo')

parser.add_argument('--resume', default='cp/SiamMask_DAVIS.pth', type=str,
                    metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--data', default='data/Human6', help='videos or image files')
parser.add_argument('--mask-dilation', default=32, type=int, help='mask dilation when inpainting')
parser.add_argument('--visualization', action="store_true", help='whether to draw')
parser.add_argument('--init-rect', default="None", type=str, help='the initial position of object')

args = parser.parse_args()
if args.init_rect == "None":
    args.init_rect = open('data/init_rect.txt').readlines()
    args.init_rect = args.init_rect[int(args.data.split('/')[-1])].strip('\n')
# mask(args)
stick_background(args)

