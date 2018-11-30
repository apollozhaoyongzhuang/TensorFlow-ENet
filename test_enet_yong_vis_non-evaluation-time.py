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
flags.DEFINE_integer('batch_size', 10, 'The batch_size for evaluation.')

flags.DEFINE_integer('image_height', 380, "The input height of the images.")
flags.DEFINE_integer('image_width', 460, "The input width of the images.")

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
photo_dir = os.path.join(FLAGS.logdir, "images")
logdir = FLAGS.logdir

#===============PREPARATION FOR TRAINING==================
#Checkpoint directories
checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)

#Dataset directories
# image_files = sorted([os.path.join(dataset_dir, 'Cityscapes_test', file) for file in os.listdir(dataset_dir + "/Cityscapes_test") if file.endswith('.png')])
# annotation_files = sorted([os.path.join(dataset_dir, "Cityscapes_testannot", file) for file in os.listdir(dataset_dir + "/Cityscapes_testannot") if file.endswith('.png')])

# #Dataset directories  our_images
# image_files = sorted([os.path.join(dataset_dir, 'our_images', file) for file in os.listdir(dataset_dir + "/our_images") if file.endswith('.png')])
image_files = sorted([os.path.join(dataset_dir, 'test', file) for file in os.listdir(dataset_dir + "/test") if file.endswith('.png')])
annotation_files = sorted([os.path.join(dataset_dir, "testannot", file) for file in os.listdir(dataset_dir + "/testannot") if file.endswith('.png')])

num_batches_per_epoch = len(image_files) / batch_size
num_steps_per_epoch = num_batches_per_epoch


def colormap(rgb=False):
    color_list = np.array(
        [
            0.000, 0.447, 0.741,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 0.333, 0.500,
            0.000, 0.667, 0.500,
            0.000, 1.000, 0.500,
            0.333, 0.000, 0.500,
            0.333, 0.333, 0.500,
            0.333, 0.667, 0.500,
            0.333, 1.000, 0.500,
            0.667, 0.000, 0.500,
            0.667, 0.333, 0.500,
            0.667, 0.667, 0.500,
            0.667, 1.000, 0.500,
            1.000, 0.000, 0.500,
            1.000, 0.333, 0.500,
            1.000, 0.667, 0.500,
            1.000, 1.000, 0.500,
            0.000, 0.333, 1.000,
            0.000, 0.667, 1.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.167, 0.000, 0.000,
            0.333, 0.000, 0.000,
            0.500, 0.000, 0.000,
            0.667, 0.000, 0.000,
            0.833, 0.000, 0.000,
            1.000, 0.000, 0.000,
            0.000, 0.167, 0.000,
            0.000, 0.333, 0.000,
            0.000, 0.500, 0.000,
            0.000, 0.667, 0.000,
            0.000, 0.833, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 0.167,
            0.000, 0.000, 0.333,
            0.000, 0.000, 0.500,
            0.000, 0.000, 0.667,
            0.000, 0.000, 0.833,
            0.000, 0.000, 1.000,
            0.000, 0.000, 0.000,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857,
            1.000, 1.000, 1.000
        ]
    ).astype(np.float32)
    color_list = color_list.reshape((-1, 3)) * 255
    if not rgb:
        color_list = color_list[:, ::-1]
    return color_list

#=============EVALUATION=================
def run():
    with tf.Graph().as_default() as graph:
        tf.logging.set_verbosity(tf.logging.INFO)

        #===================TEST BRANCH=======================
        #Load the files into one input queue
        images = tf.convert_to_tensor(image_files)
        input_queue = tf.train.slice_input_producer([images], shuffle=False)

        #Decode the image and annotation raw content
        image = tf.read_file(input_queue[0])
        image = tf.image.decode_image(image, channels=3)
        preprocessed_image = preprocess(image, None, image_height, image_width)

        images = tf.train.batch([preprocessed_image], batch_size=batch_size,
                                             allow_smaller_final_batch=True)

        #Create the model inference
        with slim.arg_scope(ENet_arg_scope()):
            logits, probabilities = ENet(images,
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

        #State the metrics that you want to predict. We get a predictions that is not one_hot_encoded.
        predictions = tf.argmax(probabilities, -1)

        #Create the global step and an increment op for monitoring
        global_step = get_or_create_global_step()
        global_step_op = tf.assign(global_step, global_step + 1) #no apply_gradient method so manually increasing the global_step



        #Define your supervisor for running a managed session. Do not run the summary_op automatically or else it will consume too much memory
        sv = tf.train.Supervisor(logdir = logdir, summary_op = None, init_fn=restore_fn)

        #Run the managed session
        with sv.managed_session() as sess:

            #Save the images
            if save_images:
                if not os.path.exists(photo_dir):
                    os.mkdir(photo_dir)

                for step in range(int(num_steps_per_epoch)):
                    # Compute summaries every 10 steps and continue evaluating
                    time_run = time.time()
                    predictions_val = sess.run([predictions])
                    time_run_end = time.time()
                    predictions_val_tuple = predictions_val[0]

                    print('totally cost (second)', time_run_end - time_run)


                    for i in range(predictions_val_tuple.shape[0]):
                        predicted_annotation = predictions_val_tuple[i]

                        # plt.subplot(1, 2, 1)
                        plt.imshow(predicted_annotation)
                        # plt.subplot(1, 2, 2)
                        # plt.imshow(img)
                        plt.savefig(photo_dir + "/image_" + str(image_files[step * num_epochs + i])[15:])
                        # plt.savefig(photo_dir + "/image_" + str(step * num_epochs + i))

                        # img_color = np.ones( (3,predictions_val_tuple[i, :, :].shape[0],predictions_val_tuple[i,:,:].shape[1]), dtype=np.int8 )
                        # for i in range predictions_val_tuple[i, :, :].shape[0]:
                        #     for j in range predictions_val_tuple[i,:,:].shape[1]:
                        #
                        #         img_color
                        # img_color = predictions_val_tuple[i, :, :] * 8
                        # cv2.imwrite(photo_dir + "/image_" + str(step * num_epochs + i), predicted_annotation)

            # else
            #     print('直接输出结果（numpy）到手机端')
                # #Save the image visualizations for the first 10 images.
                # logging.info('Saving the images now...')
                # predictions_val = sess.run([predictions])
                # predictions_val_tuple = predictions_val[0]
                #
                # for i in range(10):
                #     predicted_annotation = predictions_val_tuple[i]
                #     # annotation = annotations_val[i]
                #
                #     plt.subplot(1,2,1)
                #     plt.imshow(predicted_annotation)
                #     plt.subplot(1,2,2)
                #     plt.imshow(img)
                #     plt.savefig(photo_dir+"/image_" + str(i))

if __name__ == '__main__':
    time_start = time.time()
    run()
    time_end = time.time()
    print('totally cost (second)', time_end - time_start)