import sys, os
import keras
import cv2
import traceback
import tensorflow as tf

from src.keras_utils import load_model
from glob import glob
from os.path import splitext, basename
from src.utils import im2single
from src.keras_utils import load_model, detect_lp
from src.label import Shape, writeShapes


def adjust_pts(pts, lroi):
	return pts * lroi.wh().reshape((2, 1)) + lroi.tl().reshape((2, 1))


if __name__ == '__main__':
	gpus = tf.config.experimental.list_physical_devices('GPU')
	if gpus:
		# Restrict TensorFlow to only allocate 1GB of memory on the first GPU
		try:
			tf.config.experimental.set_virtual_device_configuration(
				gpus[0],
				[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
			logical_gpus = tf.config.experimental.list_logical_devices('GPU')
			print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
		except RuntimeError as e:
			# Virtual devices must be set before GPUs have been initialized
			print(e)

	try:

		input_dir = sys.argv[1]
		output_dir = input_dir

		lp_threshold = .5

		wpod_net_path = sys.argv[2]
		wpod_net = load_model(wpod_net_path)

		imgs_paths = glob('%s/*car.png' % input_dir)

		print('Searching for license plates using WPOD-NET')

		for i, img_path in enumerate(imgs_paths):

			print(('\t Processing %s' % img_path))

			bname = splitext(basename(img_path))[0]
			Ivehicle = cv2.imread(img_path)

			ratio = float(max(Ivehicle.shape[:2])) / min(Ivehicle.shape[:2])
			side = int(ratio * 288.)
			bound_dim = min(side + (side % (2 ** 4)), 608)
			print(("\t\tBound dim: %d, ratio: %f" % (bound_dim, ratio)))

			Llp, LlpImgs, _ = detect_lp(wpod_net, im2single(Ivehicle), bound_dim, 2 ** 4, (240, 80), lp_threshold)

			if len(LlpImgs):
				Ilp = LlpImgs[0]
				Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
				Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)

				s = Shape(Llp[0].pts)

				cv2.imwrite('%s/%s_lp.png' % (output_dir, bname), Ilp * 255.)
				writeShapes('%s/%s_lp.txt' % (output_dir, bname), [s])

	except:
		traceback.print_exc()
		sys.exit(1)

	sys.exit(0)
