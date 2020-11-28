import tensorflow as tf

""""
PRO-TAT*: By adding randomness to the coefficients, integratS three basic affine transformations
, namely translation, scaling, and rotation, into one procedure.
*not published yet
T: translation limit
S: scaling limit
R: rotation limit
"""


def tf_rand_cropping(x, length, crop_size=299):
    x_size = tf.to_float(length)
    frac = crop_size / x_size
    start_fraction_max = (x_size - crop_size) / x_size
    start_x = tf.random_uniform((), 0, start_fraction_max)
    start_y = tf.random_uniform((), 0, start_fraction_max)
    cropped = tf.image.crop_and_resize([x], boxes=[[start_y, start_x, start_y + frac, start_x + frac]],
                                       box_ind=[0], crop_size=[crop_size, crop_size])
    final_crop = tf.squeeze(cropped)
    return final_crop


def tf_rand_padding(x, length, pad_size=299):
    data = tf.expand_dims(x, 0)
    rnd = tf.random_uniform((), length, pad_size, dtype=tf.int32)
    rescaled = tf.image.crop_and_resize(data, [[0, 0, 1, 1]], [0], [rnd, rnd])
    h_rem = pad_size - rnd
    w_rem = pad_size - rnd
    pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
    pad_right = w_rem - pad_left
    pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
    pad_bottom = h_rem - pad_top
    padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0)
    padded.set_shape((1, pad_size, pad_size, 3))
    final_pad = tf.squeeze(padded)
    return final_pad


def defend_PROTAT(img, shift_limit=0.16, scale_limit=0.16, rotate_limit=4):
    # initialization
    angle = tf.random_uniform((), -rotate_limit, rotate_limit, dtype=tf.float32)
    scale = tf.random_uniform((), 1 - scale_limit, 1 + scale_limit, dtype=tf.float32)
    height = 299
    width = 299
    dx = tf.cast(tf.random_uniform((), -shift_limit, shift_limit, dtype=tf.float32) * width, tf.int64)
    dy = tf.cast(tf.random_uniform((), -shift_limit, shift_limit, dtype=tf.float32) * height, tf.int64)
    length = tf.cast(scale * height, dtype=tf.int32)
    nheight = scale * height
    nwidth = scale * width
    size = tf.squeeze(tf.reshape(tf.convert_to_tensor([nheight, nwidth], dtype=tf.int32), (1, -1)))

    # translation
    mid_img = tf.roll(img, shift=dx, axis=0)
    shifted_img = tf.roll(mid_img, shift=dy, axis=1)

    # rotate
    rotated_img = tf.contrib.image.rotate(shifted_img, angle * math.pi / 180, interpolation='BILINEAR')

    # scale
    scaled_img = tf.image.resize(rotated_img, size)
    final_img = tf.cond(scale > 1, lambda: tf_rand_cropping(scaled_img, length),
                        lambda: tf_rand_padding(scaled_img, length))
    return final_img



"""
RDG: Random distortion over grids.
Qiu, Han, et al. "Mitigating Advanced Adversarial Attacks with More Advanced Gradient Obfuscation Techniques." arXiv preprint arXiv:2005.13712 (2020).

num_steps: number of grids
distort_limit: distortion limit
"""
def defend_RDG(img,num_steps = 26,distort_limit = 0.33):
    upflag = tf.round(tf.random_uniform((1,), 0, 1, dtype=tf.float32))
    leftflag = tf.round(tf.random_uniform((1,), 0, 1, dtype=tf.float32))
    xstep = tf.constant(1.0) + tf.random_uniform((num_steps + 1,),
                                                 -distort_limit,
                                                 distort_limit,
                                                 dtype=tf.float32)
    ystep = tf.constant(1.0) + tf.random_uniform((num_steps + 1,),
                                                 -distort_limit,
                                                 distort_limit,
                                                 dtype=tf.float32)
    img_shape = tf.shape(img)
    height, width = img_shape[0], img_shape[1]
    x_step = width // num_steps
    y_step = height // num_steps
    xs = tf.range(0.0, tf.dtypes.cast(width, tf.float32), delta=x_step)
    ys = tf.range(0.0, tf.dtypes.cast(height, tf.float32), delta=y_step)
    prev = tf.constant(0.0)
    listvec_x = tf.zeros((1, 1))
    for i in range(num_steps + 1):
        start = tf.cast(xs[i], tf.int32)
        end = tf.cast(xs[i], tf.int32) + x_step
        cur = tf.cond(end > width, lambda: tf.cast(width, tf.float32),
                      lambda: prev + tf.cast(x_step, tf.float32) * xstep[i])
        end = tf.cond(end > width, lambda: width, lambda: end)
        listvec_x = tf.concat([listvec_x, tf.reshape(tf.linspace(prev, cur, end - start), (1, -1))], -1)
        prev = cur
    xx = tf.cast(tf.clip_by_value(tf.round(listvec_x), 0, 298), tf.int32)
    map_x = tf.tile(xx[:, 1:], (299, 1))
    xx2 = tf.reverse((298 * tf.ones_like(xx, dtype=tf.int32) - xx), [1])
    map_x2 = tf.tile(xx2[:, :299], (299, 1))
    prev = tf.constant(0.0)
    listvec_y = tf.zeros((1, 1))
    for i in range(num_steps + 1):
        start = tf.cast(ys[i], tf.int32)
        end = tf.cast(ys[i], tf.int32) + y_step
        cur = tf.cond(end > width, lambda: tf.cast(height, tf.float32),
                      lambda: prev + tf.cast(y_step, tf.float32) * ystep[i])
        end = tf.cond(end > width, lambda: width, lambda: end)
        listvec_y = tf.concat([listvec_y, tf.reshape(tf.linspace(prev, cur, end - start), (1, -1))], -1)
        prev = cur
    yy = tf.cast(tf.clip_by_value(tf.round(listvec_y), 0, 298), tf.int32)
    map_y = tf.tile(tf.transpose(yy)[1:, :], (1, 299))
    yy2 = tf.reverse((298 * tf.ones_like(yy, dtype=tf.int32) - yy), [1])
    map_y2 = tf.tile(tf.transpose(yy2)[:299, :], (1, 299))
    index_x = tf.cond(leftflag[0] > 0.5, lambda: tf.identity(map_x), lambda: tf.identity(map_x2))
    index_y = tf.cond(upflag[0] > 0.5, lambda: tf.identity(map_y), lambda: tf.identity(map_y2))
    index = tf.stack([index_y, index_x], 2)
    x_gd = tf.gather_nd(img, index)
    return x_gd

"""
CROP*: Random sized cropping

minlimit: the maximum scale that the img will remain
w2h: aspect ratio of crop.
*not published yet
"""


def defend_CROP(img,minlimit=0.66,w2h=0.91):
    crop_height = random.randint(int(299*minlimit),299)
    crop_width = int(crop_height*w2h)
    h_start = random.random()
    w_start = random.random()
    y1 = int((299 - crop_height) * h_start)
    y2 = y1 + crop_height
    x1 = int((299 - crop_width) * w_start)
    x2 = x1 + crop_width
    croped = tf.image.crop_and_resize([img], boxes=[[y1/299, x1/299, y2/299, x2/299]],
                             box_ind=[0], crop_size=[299, 299])
    final = tf.squeeze(croped)
    return final


"""
RAND*: Random padding
*not published yet

scalimit: the maximum scale
"""
def defend_RAND(x, length=299, scalimit=1.3):
    data = tf.expand_dims(x, 0)
    pad_size = int(scalimit * length)
    rnd = tf.random_uniform((), length, pad_size, dtype=tf.int32)
    rescaled = tf.image.crop_and_resize(data, [[0, 0, 1, 1]], [0], [rnd, rnd])
    h_rem = pad_size - rnd
    w_rem = pad_size - rnd
    pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
    pad_right = w_rem - pad_left
    pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
    pad_bottom = h_rem - pad_top
    padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0)
    padded.set_shape((1, pad_size, pad_size, 3))
    final_pad = tf.squeeze(padded)
    return final_pad