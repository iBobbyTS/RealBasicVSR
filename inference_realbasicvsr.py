import argparse
import glob
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import subprocess

import cv2
import mmcv
import numpy as np
import torch
from mmcv.runner import load_checkpoint
from mmedit.core import tensor2img

from realbasicvsr.models.builder import build_model
import configs.ffmpeg

VIDEO_EXTENSIONS = ('.mp4', '.mov')


def parse_args():
	parser = argparse.ArgumentParser(
		description='Inference script of RealBasicVSR')
	parser.add_argument('config', help='test config file path')
	parser.add_argument('checkpoint', help='checkpoint file')
	parser.add_argument('input_dir', help='directory of the input video')
	parser.add_argument('output_dir', help='directory of the output video')
	parser.add_argument(
		'--max_seq_len',
		type=int,
		default=None,
		help='maximum sequence length to be processed')
	parser.add_argument(
		'--is_save_as_png',
		type=bool,
		default=True,
		help='whether to save as png')
	parser.add_argument(
		'--fps', type=float, default=25, help='FPS of the output video')
	args = parser.parse_args()

	return args


def init_model(config, checkpoint=None):
	"""Initialize a model from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str): Which device the model will deploy. Default: 'cuda:0'.

    Returns:
        nn.Module: The constructed model.
    """

	if isinstance(config, str):
		config = mmcv.Config.fromfile(config)
	elif not isinstance(config, mmcv.Config):
		raise TypeError('config must be a filename or Config object, '
						f'but got {type(config)}')
	config.model.pretrained = None
	config.test_cfg.metrics = None
	model = build_model(config.model, test_cfg=config.test_cfg)
	if checkpoint is not None:
		checkpoint = load_checkpoint(model, checkpoint)

	model.cfg = config  # save the config in the model for convenience
	model.eval()

	return model


def main():
	args = parse_args()

	# initialize the model
	model = init_model(args.config, args.checkpoint)

	# read images
	file_extension = os.path.splitext(args.input_dir)[1]
	if file_extension in VIDEO_EXTENSIONS:  # input is a video file
		video_reader = mmcv.VideoReader(args.input_dir)
		ori_w = video_reader.width
		ori_h = video_reader.height
		inputs = []
	elif file_extension == '':  # input is a directory
		inputs = []
		input_paths = sorted(glob.glob(f'{args.input_dir}/*'))
		for input_path in input_paths:
			img = mmcv.imread(input_path, channel_order='rgb')
			inputs.append(img)
	else:
		raise ValueError('"input_dir" can only be a video or a directory.')
	out_w = ori_w * 4
	out_h = ori_h * 4

	for i, img in enumerate(inputs):
		img = torch.from_numpy(img / 255.).permute(2, 0, 1).float()
		inputs[i] = img.unsqueeze(0)
	# Only for imgs
	# inputs = torch.stack(inputs, dim=1)

	# map to cuda, if available
	cuda_flag = False
	if torch.cuda.is_available():
		model = model.cuda()
		cuda_flag = True
	# Configure out pipe

	if os.path.splitext(args.output_dir)[1] in VIDEO_EXTENSIONS:
		output_dir = os.path.dirname(args.output_dir)
		mmcv.mkdir_or_exist(output_dir)
		video_writer = subprocess.Popen(
			'ffmpeg -pix_fmt bgr24 -f rawvideo -s {}x{} -r {} -i - {} -c:v {} -vtag {} -pix_fmt {} -crf {} -preset {} -c:a copy {} -y'.format(
				out_w, out_h, video_reader.fps,
				configs.ffmpeg.resize, configs.ffmpeg.vcodec, configs.ffmpeg.vtag, configs.ffmpeg.pix_fmt,
				configs.ffmpeg.crf, configs.ffmpeg.preset, args.output_dir
			), shell=True, stdin=subprocess.PIPE
		)

	with torch.no_grad():
		if isinstance(args.max_seq_len, int):
			outputs = []
			for i in range(0, video_reader.frame_cnt, args.max_seq_len):
				imgs = torch.empty((1, args.max_seq_len, ori_h, ori_w, 3), dtype=torch.uint8)
				for j in range(args.max_seq_len):
					imgs[0, j, :, :, :] = torch.from_numpy(video_reader.next())
				if cuda_flag:
					imgs = imgs.cuda()
				imgs = imgs.permute(0, 1, 4, 2, 3).float() / 255.
				outputs = model(imgs, test_mode=True)['output']
				outputs = (outputs[0].permute(0, 2, 3, 1)*255.0).round().clamp(0, 255).byte().cpu().numpy()
				for output in outputs:
					video_writer.stdin.write(output.tobytes())
		else:
			if cuda_flag:
				inputs = inputs.cuda()
			outputs = model(inputs, test_mode=True)['output'].cpu()

	if os.path.splitext(args.output_dir)[1] in VIDEO_EXTENSIONS:
		video_writer.communicate()
		video_writer.terminate()
	else:
		mmcv.mkdir_or_exist(args.output_dir)
		for i in range(0, outputs.size(1)):
			output = tensor2img(outputs[:, i, :, :, :])
			filename = os.path.basename(input_paths[i])
			if args.is_save_as_png:
				file_extension = os.path.splitext(filename)[1]
				filename = filename.replace(file_extension, '.png')
			mmcv.imwrite(output, f'{args.output_dir}/{filename}')


if __name__ == '__main__':
	main()
