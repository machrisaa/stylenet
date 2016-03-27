import time

import inspect
import os
import numpy as np
import skimage
import skimage.io
import skimage.transform
import tensorflow as tf

from neural_style import custom_vgg19
import stylenet_core


def get_filename(file):
    return os.path.splitext(os.path.basename(file))[0]


def render(content_file, style_file,
           content_region_file=None, style_region_file=None,
           random_init=False, load_saved_mapping=True, load_trained_image=False, blur_mapping=True,
           height=None, width=None,
           content_ratio=0., style3_ratio=3., style4_ratio=1., gram_ratio=0.001, diff_ratio=0.,
           epochs=300, output_file="./train/output%d.jpg"):
    print "render started:"

    # print info:
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    for i in args:
        print "    %s = %s" % (i, values[i])

    content_np = stylenet_core.load_image(content_file, height, width)
    style_np = stylenet_core.load_image(style_file, content_np.shape[0], content_np.shape[1])

    content_batch = np.expand_dims(content_np, 0)
    style_batch = np.expand_dims(style_np, 0)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    with tf.Session(config=tf.ConfigProto(gpu_options=(gpu_options), log_device_placement=False)) as sess:
        start_time = time.time()

        contents = tf.constant(content_batch, dtype=tf.float32, shape=content_batch.shape)
        styles = tf.constant(style_batch, dtype=tf.float32, shape=style_batch.shape)

        if random_init:
            var_image = tf.Variable(tf.truncated_normal(content_batch.shape, 0.5, 0.1))
        else:
            var_image = tf.Variable(contents)

        vgg_content = custom_vgg19.Vgg19()
        with tf.name_scope("content_vgg"):
            vgg_content.build(contents)

        vgg_style = custom_vgg19.Vgg19()
        with tf.name_scope("style_vgg"):
            vgg_style.build(styles)

        vgg_var = custom_vgg19.Vgg19()
        with tf.name_scope("variable_vgg"):
            vgg_var.build(var_image)

        with tf.name_scope("cost"):
            # style:
            # TODO change file name based on out file name
            style3file = "./train/%s-style_map_3" % (
                get_filename(content_file) + "-" + get_filename(style_file))
            style4file = "./train/%s-style_map_4" % (
                get_filename(content_file) + "-" + get_filename(style_file))

            if content_region_file is None or style_region_file is None:
                if style3_ratio is 0:
                    style_cost_3 = tf.constant(0.0)
                else:
                    style_cost_3 = stylenet_core.get_style_cost_patch2(sess, vgg_var.conv3_1,
                                                                       vgg_content.conv3_1,
                                                                       vgg_style.conv3_1,
                                                                       style3file,
                                                                       load_saved_mapping=load_saved_mapping)
                if style4_ratio is 0:
                    style_cost_4 = tf.constant(0.0)
                else:
                    style_cost_4 = stylenet_core.get_style_cost_patch2(sess, vgg_var.conv4_1,
                                                                       vgg_content.conv4_1,
                                                                       vgg_style.conv4_1,
                                                                       style4file,
                                                                       load_saved_mapping=load_saved_mapping)
            else:
                content_regions_np = stylenet_core.load_image(content_region_file, content_np.shape[0],
                                                              content_np.shape[1])
                style_regions_np = stylenet_core.load_image(style_region_file, content_np.shape[0],
                                                            content_np.shape[1])
                content_regions_batch = np.expand_dims(content_regions_np, 0)
                style_regions_batch = np.expand_dims(style_regions_np, 0)
                content_regions = tf.constant(content_regions_batch, dtype=tf.float32,
                                              shape=content_regions_batch.shape)
                style_regions = tf.constant(style_regions_batch, dtype=tf.float32,
                                            shape=style_regions_batch.shape)

                content_regions = vgg_var.avg_pool(content_regions, None)
                content_regions = vgg_var.avg_pool(content_regions, None)
                style_regions = vgg_var.avg_pool(style_regions, None)
                style_regions = vgg_var.avg_pool(style_regions, None)

                if style3_ratio is 0:
                    style_cost_3 = tf.constant(0.0)
                else:
                    style_cost_3 = stylenet_core.get_style_cost_patch2(sess,
                                                                       vgg_var.conv3_1,
                                                                       vgg_content.conv3_1,
                                                                       vgg_style.conv3_1,
                                                                       style3file,
                                                                       content_regions,
                                                                       style_regions,
                                                                       load_saved_mapping,
                                                                       blur_mapping=blur_mapping)

                content_regions = vgg_var.avg_pool(content_regions, None)
                style_regions = vgg_var.avg_pool(style_regions, None)

                if style4_ratio is 0:
                    style_cost_4 = tf.constant(0.0)
                else:
                    style_cost_4 = stylenet_core.get_style_cost_patch2(sess,
                                                                       vgg_var.conv4_1,
                                                                       vgg_content.conv4_1,
                                                                       vgg_style.conv4_1,
                                                                       style4file,
                                                                       content_regions,
                                                                       style_regions,
                                                                       load_saved_mapping,
                                                                       blur_mapping=blur_mapping)

            if gram_ratio is 0:
                style_cost_gram = tf.constant(0.0)
            else:
                style_cost_gram = stylenet_core.get_style_cost_gram(sess, vgg_style, vgg_var)

            # content:
            if content_ratio is 0:
                content_cost = tf.constant(.0)
            else:
                fixed_content = stylenet_core.get_constant(sess, vgg_content.conv4_2)
                content_cost = stylenet_core.l2_norm_cost(vgg_var.conv4_2 - fixed_content)

            # # smoothness:
            if diff_ratio is 0:
                diff_cost = tf.constant(.0)
            else:
                diff_filter_h = tf.constant([0, 0, 0, 0, -1, 1, 0, 0, 0], tf.float32, [3, 3, 1, 1])
                diff_filter_h = tf.concat(2, [diff_filter_h, diff_filter_h, diff_filter_h])
                diff_filter_v = tf.constant([0, 0, 0, 0, -1, 0, 0, 1, 0], tf.float32, [3, 3, 1, 1])
                diff_filter_v = tf.concat(2, [diff_filter_v, diff_filter_v, diff_filter_v])
                diff_filter = tf.concat(3, [diff_filter_h, diff_filter_v])
                filtered_input = tf.nn.conv2d(var_image, diff_filter, [1, 1, 1, 1], "VALID")
                diff_cost = stylenet_core.l2_norm_cost(filtered_input) * 1e7

            content_cost = content_cost * content_ratio
            style_cost_3 = style_cost_3 * style3_ratio
            style_cost_4 = style_cost_4 * style4_ratio
            style_cost_gram = style_cost_gram * gram_ratio
            diff_cost = diff_cost * diff_ratio
            cost = content_cost + style_cost_3 + style_cost_4 + style_cost_gram + diff_cost

        with tf.name_scope("train"):
            global_step = tf.Variable(0, name='global_step', trainable=False)

            optimizer = tf.train.AdamOptimizer(learning_rate=0.02)
            gvs = optimizer.compute_gradients(cost)

            training = optimizer.apply_gradients(gvs, global_step=global_step)

        print "Net generated:", (time.time() - start_time)
        start_time = time.time()

        with tf.name_scope("image_out"):
            image_out = tf.clip_by_value(tf.squeeze(var_image, [0]), 0, 1)

        saver = tf.train.Saver()

        checkpoint = tf.train.get_checkpoint_state("./train")
        if checkpoint and checkpoint.model_checkpoint_path and load_trained_image:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print "save restored:", checkpoint.model_checkpoint_path
        else:
            tf.initialize_all_variables().run()
            print "all variables init"

        print "Var init: %d" % (time.time() - start_time)

        step_out = 0
        start_time = time.time()
        for i in xrange(epochs):
            if i % 5 == 0:
                img = sess.run(image_out)
                img_out_path = output_file % step_out
                skimage.io.imsave(img_out_path, img)
                print "img saved: ", img_out_path

            step_out, content_out, style_patch3_out, style_patch4_out, style_gram_out, diff_cost_out, cost_out \
                , _ = sess.run(
                [global_step, content_cost, style_cost_3, style_cost_4, style_cost_gram, diff_cost, cost,
                 training])

            duration = time.time() - start_time
            print "Step %d: cost:%.10f\t(%.1f sec)" % (step_out, cost_out, duration), \
                "\t content:%.5f, style_3:%.5f, style_4:%.5f, gram:%.5f, diff_cost_out:%.5f" \
                % (content_out, style_patch3_out, style_patch4_out, style_gram_out, diff_cost_out)

            if (i + 1) % 10 == 0:
                saved_path = saver.save(sess, "./train/saves-" + get_filename(content_file),
                                        global_step=global_step)
                print "net saved: ", saved_path

        img = sess.run(image_out)
        img_out_path = output_file % step_out
        skimage.io.imsave(img_out_path, img)
        print "img saved: ", img_out_path


def render_gen(content_file, style_file,
               content_region_file=None, style_region_file=None,
               random_init=False, load_saved_mapping=True, load_trained_image=False, blur_mapping=True,
               height=None, width=None,
               content_ratio=0, style3_ratio=3., style4_ratio=1., gram_ratio=0.001, diff_ratio=0.,
               gen_epochs=80, max_gen=3, pyramid=True, max_reduction_ratio=.8, final_epochs=200):
    for gen in xrange(max_gen):
        if gen is 0:
            gen_content_file = content_file
            height = stylenet_core.load_image(content_file, height, width).shape[0]
        else:
            gen_content_file = ("./train/output-g" + str(gen - 1) + "-%d.jpg") % gen_epochs

        output_file = "./train/output-g" + str(gen) + "-%d.jpg"
        output_file_final = output_file % gen_epochs
        if os.path.isfile(output_file_final):
            print output_file_final, "exist. move to next generation"
            continue

        tf.reset_default_graph()
        ot = time.time()
        print "----------- %d generation started -----------" % gen

        if pyramid and gen == max_gen - 1:
            h = height
            epochs = final_epochs
            cr = 0
            gr = 0
            bm = blur_mapping
        else:
            h = int(height * (gen * (1.0 - max_reduction_ratio) / max_gen + max_reduction_ratio))
            epochs = gen_epochs
            cr = content_ratio
            gr = gram_ratio
            bm = False

        render(
            content_file=gen_content_file,
            style_file=style_file,
            content_region_file=content_region_file,
            style_region_file=style_region_file,
            random_init=random_init,
            load_saved_mapping=load_saved_mapping,
            load_trained_image=load_trained_image,
            blur_mapping=bm,
            height=h,
            width=width,
            content_ratio=cr,
            style3_ratio=style3_ratio,
            style4_ratio=style4_ratio,
            gram_ratio=gr,
            diff_ratio=diff_ratio,
            epochs=epochs,
            output_file=output_file)
        print "----------- %d generation finished in %d sec -----------\n" % (gen, time.time() - ot)


if __name__ == "__main__":
    # for testing:

    # no generation
    # render("./test_data/cat_h.jpg", "./test_data/cat-water-colour.jpg", height=500)

    # with generation
    render_gen("./images/husky_paint.jpg", "./test_data/husky_real.jpg",
               "./images/husky_paint_region.jpg", "./test_data/husky_real_region.jpg", height=500)
