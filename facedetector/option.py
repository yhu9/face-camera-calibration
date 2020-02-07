import argparse

#############################

parser = argparse.ArgumentParser(description='land mark detection')

parser.add_argument("--path",default='../face_example.png',type=str,help="directory to process")
parser.add_argument("--testimg",default=True,type=bool,help="test an image")
parser.add_argument("--dir", default="../data/Webcam/masa_only/checker_board_and_face/train/")
parser.add_argument("--matfile", default="output/lm_closeup.mat")
parser.add_argument("--mode",default='checkerboard',type=str,help="mode = checkerboard or face")
parser.add_argument("--outpath",default='tmp',type=str,help="mode = checkerboard or face")

args = parser.parse_args()

