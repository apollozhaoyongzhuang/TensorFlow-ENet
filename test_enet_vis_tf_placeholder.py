import tensorflow as tf
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
from enet import ENet, ENet_arg_scope
from preprocessing import preprocess
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import cv2

# import matplotlib
# matplotlib.use('Agg')



os.environ['CUDA_VISIBLE_DEVICES'] = '0'
slim = tf.contrib.slim

#============INPUT ARGUMENTS================
flags = tf.app.flags

#Directories
flags.DEFINE_string('dataset_dir', './dataset', 'The dataset directory to find the train, validation and test images.')
flags.DEFINE_string('checkpoint_dir', './checkpoint', 'The checkpoint directory to restore your mode.l')
flags.DEFINE_string('logdir', './log/original_test', 'The log directory for event files created during test evaluation.')
flags.DEFINE_boolean('save_images', True, 'If True, saves 10 images to your logdir for visualization.')

#Evaluation information
flags.DEFINE_integer('num_classes', 12, 'The number of classes to predict.')
flags.DEFINE_integer('batch_size', 1, 'The batch_size for evaluation.')

flags.DEFINE_integer('image_height', 544, "The input height of the images.")
flags.DEFINE_integer('image_width', 960, "The input width of the images.")

# flags.DEFINE_integer('image_height', 1024, "The input height of the images.")
# flags.DEFINE_integer('image_width', 2048, "The input width of the images.")

flags.DEFINE_integer('num_epochs', 10, "The number of epochs to evaluate your model.")

#Architectural changes
flags.DEFINE_integer('num_initial_blocks', 1, 'The number of initial blocks to use in ENet.')
flags.DEFINE_integer('stage_two_repeat', 2, 'The number of times to repeat stage two.')
flags.DEFINE_boolean('skip_connections', False, 'If True, perform skip connections from encoder to decoder.')

FLAGS = flags.FLAGS

#==========NAME HANDLING FOR CONVENIENCE==============
num_classes = FLAGS.num_classes
batch_size = FLAGS.batch_size
image_height = FLAGS.image_height
image_width = FLAGS.image_width
num_epochs = FLAGS.num_epochs

save_images = FLAGS.save_images

#Architectural changes
num_initial_blocks = FLAGS.num_initial_blocks
stage_two_repeat = FLAGS.stage_two_repeat
skip_connections = FLAGS.skip_connections

dataset_dir = FLAGS.dataset_dir
checkpoint_dir = FLAGS.checkpoint_dir
photo_dir = os.path.join(FLAGS.logdir, "images0.2mask0.8")   #addWeighted(image_resize, 0.8, color_mask_uint8, 0.2, 0)
logdir = FLAGS.logdir

#===============PREPARATION FOR TRAINING==================
#Checkpoint directories
checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)

#Dataset directories
# image_files = sorted([os.path.join(dataset_dir, 'Cityscapes_test', file) for file in os.listdir(dataset_dir + "/Cityscapes_test") if file.endswith('.png')])
# annotation_files = sorted([os.path.join(dataset_dir, "Cityscapes_testannot", file) for file in os.listdir(dataset_dir + "/Cityscapes_testannot") if file.endswith('.png')])

# #Dataset directories
# image_files = sorted([os.path.join(dataset_dir, 'test', file) for file in os.listdir(dataset_dir + "/test") if file.endswith('.png')])
# annotation_files = sorted([os.path.join(dataset_dir, "testannot", file) for file in os.listdir(dataset_dir + "/testannot") if file.endswith('.png')])
#
# num_batches_per_epoch = len(image_files) / batch_size
# num_steps_per_epoch = num_batches_per_epoch


def gray_convert_color(var):
    return {
            0: [1.000, 1.000, 1.000],
            1: [1.000, 1.000, 0.000],
            2: [0.667, 0.333, 0.000],
            3: [0.000, 0.333, 0.500],
            4: [0.333, 1.000, 0.500],
            5: [0.000, 1.000, 0.000],
            6: [0.333, 0.000, 1.000],
            7: [0.667, 0.667, 1.000],
            8: [1.000, 0.000, 0.000],
            9: [0.000, 0.333, 0.000],
            10: [0.000, 0.000, 0.333],
            11: [0.143, 0.143, 0.143],
    }.get(var,'error')

#=============EVALUATION=================
def run():
    with tf.Graph().as_default() as graph:
        tf.logging.set_verbosity(tf.logging.INFO)

        images_placeholder = tf.placeholder(tf.float32, [batch_size, image_height, image_width, 3], name='rgb_image')
        #Create the model inference
        with slim.arg_scope(ENet_arg_scope()):
            logits, probabilities = ENet(images_placeholder,
                                         num_classes,
                                         batch_size=batch_size,
                                         is_training=True,
                                         reuse=None,
                                         num_initial_blocks=num_initial_blocks,
                                         stage_two_repeat=stage_two_repeat,
                                         skip_connections=skip_connections)

        # Set up the variables to restore and restoring function from a saver.
        exclude = []
        variables_to_restore = slim.get_variables_to_restore(exclude=exclude)

        saver = tf.train.Saver(variables_to_restore)
        def restore_fn(sess):
            return saver.restore(sess, checkpoint_file)

        #Define your supervisor for running a managed session. Do not run the summary_op automatically or else it will consume too much memory
        sv = tf.train.Supervisor(logdir = logdir, summary_op = None, init_fn=restore_fn)


        #Run the managed session   sv.managed_session()   ENet
        with sv.managed_session() as sess:

            #Save the images  our_video
            if save_images:
                if not os.path.exists(photo_dir):
                    os.mkdir(photo_dir)
                kernel = np.ones((10, 10), np.uint8)
                while True:
                    image_files = sorted(
                        [os.path.join(dataset_dir, 'our_video', file) for file in os.listdir(dataset_dir + "/our_video") if
                         file.endswith('.jpg')])
                    if len(image_files) > 0:
                        # Load the files into one input queue

                        time_all_image_start = time.time()
                        for image_name in image_files:
                            time_run = time.time()
                            image = cv2.imread(image_name)
                            image_resize = cv2.resize(image, (image_width, image_height), interpolation=cv2.INTER_CUBIC)
                            image_resize_float32 = image_resize.astype('float32')
                            image_resize_float32 = image_resize_float32[:, :, ::-1]
                            image_resize_float32 = image_resize_float32 / 255.0
                            batch_x = np.zeros([batch_size, image_height, image_width, 3])
                            batch_x_float32 = batch_x.astype('float32')
                            batch_x_float32[0] = image_resize_float32
                            feed_dict = {images_placeholder: batch_x_float32}
                            probabilities_numpy = sess.run(probabilities, feed_dict=feed_dict)
                            predictions_val = np.argmax(probabilities_numpy, -1)

                            time_Preprocessing_and_Predict_end = time.time()
                            print('One image Preprocessing_and_Predict cost (second)', time_Preprocessing_and_Predict_end - time_run)


                            for i in range(batch_size):  # predictions_val_tuple.shape[0]
                                predicted_image = predictions_val[i]
                                # plt.imshow(predicted_image)
                                time_morphologyEx_start = time.time()
                                predicted_image_uint8 = predicted_image.astype('uint8')
                                predicted_image_closing = cv2.morphologyEx(predicted_image_uint8, cv2.MORPH_OPEN, kernel)
                                color_mask = np.ones(image_resize.shape, np.float)
                                time_morphologyEx_end = time.time()
                                print('morphologyEx cost (second)', time_morphologyEx_end - time_morphologyEx_start)

                                time_color_mask_start = time.time()
                                # for x in range(predicted_image.shape[0]):
                                #     for y in range(predicted_image.shape[1]):
                                #         color_mask[x, y, :] = gray_convert_color(predicted_image_closing[x, y])

                                # for label in range(11):
                                for label in np.unique(predicted_image_closing):
                                    xy = np.where(predicted_image_closing == label)
                                    color_mask[xy[0], xy[1], :] = gray_convert_color(label)

                                color_mask_uint8 = (color_mask * 255).astype('uint8')
                                time_color_mask_end = time.time()
                                print('color_mask cost (second)', time_color_mask_end - time_color_mask_start)

                                # ###开操作去噪声
                                # plt.subplot(1, 2, 1)
                                # plt.imshow(predicted_image)
                                # plt.subplot(1, 2, 2)
                                # # original_image = cv2.imread(image_name)
                                # plt.imshow(predicted_image_closing)

                                time_addWeighted_start = time.time()
                                overlapping = cv2.addWeighted(image_resize, 0.2, color_mask_uint8, 0.8, 0)
                                time_addWeighted_end = time.time()
                                print('addWeighted cost (second)', time_addWeighted_end - time_addWeighted_start)

                                # plt.imshow(overlapping)

                                time_result_save_start = time.time()
                                # cv2.imwrite(photo_dir + "/image_" + str(image_name)[10:], overlapping)  #str(image_name)[15:]
                                cv2.imwrite(photo_dir + "/" + str(image_name)[20:], overlapping)
                                time_result_save_end = time.time()
                                print('result_save cost (second)', time_result_save_end - time_result_save_start)


                                # plt.savefig(photo_dir + "/image_" + str(image_name)[15:])
                                # cv2.imwrite(photo_dir + "/image_" + str(image_files[step * num_epochs + i])[15:], predicted_image)

                            time_run_end = time.time()
                            print('One image cost (second)', time_run_end - time_run)

                        time_all_image_end = time.time()
                        logging.info('There are %.4f image in all', len(image_files))
                        print('totally cost (second)', time_all_image_end - time_all_image_start)

                        #delete images_files
                        # python删除文件夹下所有文件

                    else:
                        print('There is no new images to deal with')
                        continue






                # images = np.zeros([1, 360, 480, 3], dtype=np.float32)
                #
                # for step in range(int(num_steps_per_epoch)):
                #     # Compute summaries every 10 steps and continue evaluating
                #     time_run = time.time()
                #
                #     # image_numpy = images.eval(session=sess)
                #     feed_dict = {images_placeholder: images}
                #     probabilities_numpy = sess.run(probabilities, feed_dict=feed_dict)
                #     predictions_val = np.argmax(probabilities_numpy, -1)
                #
                #     time_run_end = time.time()
                #
                #     print('totally cost (second)', time_run_end - time_run)
                #
                #     for i in range(batch_size):   #predictions_val_tuple.shape[0]
                #         predicted_image = predictions_val[i]
                #         plt.imshow(predicted_image)
                #         plt.savefig(photo_dir + "/image_" + str(image_files[step])[15:])
                #         # cv2.imwrite(photo_dir + "/image_" + str(image_files[step * num_epochs + i])[15:], predicted_image)


if __name__ == '__main__':
    time_start = time.time()
    run()
    time_end = time.time()
    print('totally cost (second)', time_end - time_start)