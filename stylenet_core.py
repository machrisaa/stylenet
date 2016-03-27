import time

import tensorflow as tf
import numpy as np

import skimage
import skimage.io
import skimage.transform


def load_image(path, height=None, width=None):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    if height is not None and width is not None:
        ny = height
        nx = width
    elif height is not None:
        ny = height
        nx = img.shape[1] * ny / img.shape[0]
    elif width is not None:
        nx = width
        ny = img.shape[0] * nx / img.shape[1]
    else:
        ny = img.shape[0]
        nx = img.shape[1]
    return skimage.transform.resize(img, (ny, nx))


def l2_norm(f, reduction_indices=None, keep_dims=False):
    return tf.sqrt(tf.reduce_sum(tf.square(f), reduction_indices=reduction_indices, keep_dims=keep_dims))


def l2_norm_cost(v):
    dim = v.get_shape().as_list()
    size = reduce(lambda x, y: x * y, dim)
    return tf.reduce_sum(tf.square(v)) / (size ** 2)


def l2_normalise(v, reduction_indices):
    return v / l2_norm(v, reduction_indices=reduction_indices, keep_dims=True)


def get_constant(sess, t):
    value = sess.run(t)
    return tf.constant(value)


def gram_matrix(v):
    assert isinstance(v, tf.Tensor)
    v.get_shape().assert_has_rank(4)

    dim = v.get_shape().as_list()
    v = tf.reshape(v, [dim[1] * dim[2], dim[3]])
    if dim[1] * dim[2] < dim[3]:
        return tf.matmul(v, v, transpose_b=True)
    else:
        return tf.matmul(v, v, transpose_a=True)


def get_style_cost_gram(sess, vgg_style, vgg_var):
    var_pool = [vgg_var.conv1_1, vgg_var.conv2_1, vgg_var.conv3_1, vgg_var.conv4_1,
                vgg_var.conv5_1]
    stys_pool = [vgg_style.conv1_1, vgg_style.conv2_1, vgg_style.conv3_1, vgg_style.conv4_1,
                 vgg_style.conv5_1]
    stys_pool = [get_constant(sess, gram_matrix(v)) for v in stys_pool]
    factor = 1000
    style_cost_array = []
    len = var_pool.__len__()
    for i in xrange(len):
        dim = var_pool[i].get_shape().as_list()
        size = reduce(lambda x, y: x * y, dim)
        G1 = gram_matrix(var_pool[i])
        G2 = stys_pool[i]
        layer_gram_lost = l2_norm_cost(G1 - G2) / (size ** 2)
        style_cost_array.append(layer_gram_lost)
    style_cost = reduce(lambda x, y: x + y, style_cost_array) / len * factor
    return style_cost


def _slice_patches_np(sess, v):
    v = tf.squeeze(v, [0])
    v = tf.expand_dims(v, 3)
    v_out = sess.run(v)
    assert isinstance(v_out, np.ndarray)

    dim = v_out.shape
    ot = time.time()
    print "slice started", dim
    h, w, d = dim[0], dim[1], dim[2]
    ph, pw = h - 2, w - 2
    pn = ph * pw

    filters_out = np.empty([pn, 3, 3, d, 1])
    k = 0
    for y in xrange(ph):
        for x in xrange(pw):
            s = v_out[y:y + 3, x:x + 3]
            filters_out[k] = s
            k += 1
    filters_out = np.squeeze(filters_out, 4)
    filters_out = np.transpose(filters_out, (1, 2, 3, 0))

    filters = tf.constant(filters_out, tf.float32)
    assert filters.get_shape().as_list() == [3, 3, d, pn]
    print "slice finished:", pn, time.time() - ot

    return filters


def _side(x, h, stride=3):
    if x < stride - 1:
        xf = x + 1
    elif x > h - stride:
        xf = h - x
    else:
        xf = stride
    return xf


def _overlap(h, w, strides=[3, 3]):
    v = []
    for y in xrange(h):
        yf = _side(y, h, strides[0])
        for x in xrange(w):
            xf = _side(x, w, strides[1])
            v.append(xf * yf)
    return tf.constant(v, tf.float32, [h, w])


def _join_patches_np(sess, patches, idx):
    ot = time.time()
    print "start joining patches"

    patches = tf.transpose(patches, [3, 0, 1, 2])
    patches_out = sess.run(patches)

    h = len(idx) + 2
    w = len(idx[0]) + 2
    d = patches_out.shape[3]
    p_sum = np.zeros([1, h, w, d])
    patches_map = {}
    for r in xrange(len(idx)):
        row = idx[r]
        for c in xrange(len(row)):
            i = row[c]
            p_out = patches_out[i]
            p_sum[0, r:r + 3, c:c + 3, ] += p_out
    print "patches map finished", len(patches_map), time.time() - ot
    p_sum = tf.constant(p_sum, tf.float32) / tf.reshape(_overlap(h, w, [3, 3]), [1, h, w, 1])
    target = get_constant(sess, p_sum)

    print "finished joining patches", time.time() - ot

    return target


def _create_patch(sess, content_input, style_input, content_regions, style_regions, blur_mapping):
    dim = content_input.get_shape().as_list()
    h, w, d = dim[1], dim[2], dim[3]
    ph, pw = h - 2, w - 2
    pn = ph * pw

    assert content_input.get_shape() == style_input.get_shape()
    real_style = style_input
    if content_regions is not None and style_regions is not None:
        assert content_regions.get_shape().as_list()[:3] == style_regions.get_shape().as_list()[:3]
        map_len = content_input.get_shape().as_list()[3]
        # mapped_content = tf.concat(3, [content_input] + [content_regions for _ in xrange(map_len)])
        # mapped_style = tf.concat(3, [style_input] + [style_regions for _ in xrange(map_len)])
        mapped_content = tf.concat(3, [content_input, tf.tile(content_regions, [1, 1, 1, map_len])])
        mapped_style = tf.concat(3, [style_input, tf.tile(style_regions, [1, 1, 1, map_len])])
    else:
        mapped_content = content_input
        mapped_style = real_style

    ot = time.time()

    patches = _slice_patches_np(sess, mapped_style)
    p_matrix = l2_normalise(patches, [0, 1, 2])

    conv_var = tf.nn.conv2d(mapped_content, p_matrix, [1, 1, 1, 1], "VALID", use_cudnn_on_gpu=False)

    content_slice = _slice_patches_np(sess, mapped_content)

    norm_reduce_matrix = l2_norm(content_slice, [0, 1, 2], True)
    norm_reduce_matrix = tf.reshape(norm_reduce_matrix, [1, ph, pw, 1])
    conv_var = conv_var / norm_reduce_matrix
    assert conv_var.get_shape().as_list() == [1, ph, pw, pn]

    if blur_mapping:
        # blur before max, may look more natural
        blur_size = 3
        blur = tf.constant(1, tf.float32, [blur_size, blur_size, 1, 1]) / (blur_size ** 2)
        blur = tf.tile(blur, [1, 1, pn, 1])
        conv_var = tf.nn.depthwise_conv2d(conv_var, blur, [1, 1, 1, 1], "SAME")

    max_arg = tf.arg_max(conv_var, 3)
    max_arg = tf.reshape(max_arg, [pn])
    max_arg_out = sess.run(max_arg)

    if real_style is not mapped_style:
        real_patches = _slice_patches_np(sess, real_style)
        assert real_patches.get_shape().as_list() == [3, 3, d, pn]
        patches = real_patches

    print "mapping calculation finished:", time.time() - ot

    assert patches.get_shape().as_list() == [3, 3, d, pn]

    print "mapping finished:"
    return max_arg_out, patches


def get_style_cost_patch2(sess, var_input, content_input, style_input, save_file,
                          content_regions=None, style_regions=None,
                          load_saved_mapping=True, blur_mapping=False):
    dim = var_input.get_shape().as_list()
    h, w, d = dim[1], dim[2], dim[3]
    ph, pw = h - 2, w - 2
    pn = ph * pw
    full_patch_file = save_file + "_full.npy"

    full_patch_out = None
    if load_saved_mapping:
        try:
            full_patch_out = np.load(full_patch_file)
        except:
            print "saved full patch not found"
    if full_patch_out is None:
        with tf.device("/cpu:0"):
            max_arg_out, patches = _create_patch(sess, content_input, style_input,
                                                 content_regions, style_regions, blur_mapping)
            assert patches.get_shape().as_list() == [3, 3, d, pn]

            max_arg_out = np.reshape(max_arg_out, [ph, pw])
            # print max_arg_out
            full_patch = _join_patches_np(sess, patches, max_arg_out)

            assert full_patch.get_shape() == var_input.get_shape()
            full_patch_out = sess.run(full_patch)
            np.save(full_patch_file, full_patch_out)

    full_patch = tf.constant(full_patch_out)

    cost = l2_norm_cost(var_input - full_patch)

    return cost
