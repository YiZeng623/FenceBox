import numpy as np
import math
from skimage.transform import rescale, resize
import cv2
import random
import albumentations


""""
PRO-TAT*: By adding randomness to the coefficients, integratS three basic affine transformations
, namely translation, scaling, and rotation, into one procedure.
*not published yet
T: translation limit
S: scaling limit
R: rotation limit
"""
def padding(img, h, w):
    #  in case when you have odd number
    top_pad = np.floor((h - img.shape[0]) / 2).astype(np.uint16)
    bottom_pad = np.ceil((h - img.shape[0]) / 2).astype(np.uint16)
    right_pad = np.ceil((w - img.shape[1]) / 2).astype(np.uint16)
    left_pad = np.floor((w - img.shape[1]) / 2).astype(np.uint16)
    return np.copy(np.pad(img, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode='constant', constant_values=0))
def cropping(img,cropx,cropy):
    y,x = img.shape[:2]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]
def shifting(X, dx, dy):
    X = np.roll(X, dy, axis=0)
    X = np.roll(X, dx, axis=1)
    if dy>0:
        X[:dy, :] = 0
    elif dy<0:
        X[dy:, :] = 0
    if dx>0:
        X[:, :dx] = 0
    elif dx<0:
        X[:, dx:] = 0
    return X

def defend_PROTAT(img, T=0.16, S=0.16, R=4):
    # initialization
    angle = np.random.uniform(-R, R)
    scale = np.random.uniform(1 - S, 1 + S)
    dx = np.random.uniform(-T, T)
    dy = np.random.uniform(-T, T)
    height, width = img.shape[:2]
    center = (int(width / 2), int(height / 2))
    rotated_image = np.zeros_like(img)
    # translation
    shifted_image = shifting(img, int(dx * width), int(dy * height))
    # rotation
    for r in range(height):
        for c in range(width):
            #  apply rotation matrix
            y = (r - center[0]) * math.cos(angle * np.pi / 180.0) + (c - center[1]) * math.sin(angle * np.pi / 180.0)
            x = -(r - center[0]) * math.sin(angle * np.pi / 180.0) + (c - center[1]) * math.cos(angle * np.pi / 180.0)

            #  add offset
            y += center[0]
            x += center[1]

            #  get nearest index
            # a better way is linear interpolation
            x = round(x)
            y = round(y)

            # print(r, " ", c, " corresponds to-> " , y, " ", x)

            #  check if x/y corresponds to a valid pixel in input image
            if (x >= 0 and y >= 0 and x < width and y < height):
                rotated_image[r][c][:] = shifted_image[y][x][:]
    # scaling
    new_image = rescale(rotated_image, scale)
    if scale > 1:
        # center crop (original)
        scaled_image = cropping(new_image, img.shape[0], img.shape[1])
    else:
        # center padding (original)
        scaled_image = padding(new_image, img.shape[0], img.shape[1])
    return scaled_image

"""
RDG: Random distortion over grids.
Qiu, Han, et al. "Mitigating Advanced Adversarial Attacks with More Advanced Gradient Obfuscation Techniques." arXiv preprint arXiv:2005.13712 (2020).

num_steps: number of grids
distort_limit: distortion limit
"""
def defend_RDG(img,num_steps = 26,distort_limit = 0.33):
    xsteps = [1 + random.uniform(-distort_limit, distort_limit) for i in range(num_steps + 1)]
    ysteps = [1 + random.uniform(-distort_limit, distort_limit) for i in range(num_steps + 1)]
    height, width = img.shape[:2]

    x_step = width // num_steps
    xx = np.zeros(width, np.float32)
    prev = 0
    for idx, x in enumerate(range(0, width, x_step)):
        start = x
        end = x + x_step
        if end > width:
            end = width
            cur = width
        else:
            cur = prev + x_step * xsteps[idx]

        xx[start:end] = np.linspace(prev, cur, end - start)
        prev = cur

    y_step = height // num_steps
    yy = np.zeros(height, np.float32)
    prev = 0
    for idx, y in enumerate(range(0, height, y_step)):
        start = y
        end = y + y_step
        if end > height:
            end = height
            cur = height
        else:
            cur = prev + y_step * ysteps[idx]

        yy[start:end] = np.linspace(prev, cur, end - start)
        prev = cur
    xx = np.round(xx).astype(int)
    yy = np.round(yy).astype(int)
    xx[xx >= img.shape[0]] = img.shape[0]-1
    yy[yy >= img.shape[1]] = img.shape[1]-1
    map_x, map_y = np.meshgrid(xx, yy)
    # to speed up the mapping procedure, OpenCV 2 is adopted
    map_x = map_x.astype(np.float32)
    map_y = map_y.astype(np.float32)
    outimg = cv2.remap(img, map1=map_x, map2=map_y, interpolation=1, borderMode=4, borderValue=None)
    return outimg

"""
CROP*: Random sized cropping

minlimit: the maximum scale that the img will remain
w2h: aspect ratio of crop.
*not published yet
"""
def get_random_crop_coords(height, width, crop_height, crop_width, h_start, w_start):
    y1 = int((height - crop_height) * h_start)
    y2 = y1 + crop_height
    x1 = int((width - crop_width) * w_start)
    x2 = x1 + crop_width
    return x1, y1, x2, y2
def random_crop(img, crop_height, crop_width, h_start, w_start):
    height, width = img.shape[:2]
    if height < crop_height or width < crop_width:
        raise ValueError(
            "Requested crop size ({crop_height}, {crop_width}) is "
            "larger than the image size ({height}, {width})".format(
                crop_height=crop_height, crop_width=crop_width, height=height, width=width
            )
        )
    x1, y1, x2, y2 = get_random_crop_coords(height, width, crop_height, crop_width, h_start, w_start)
    img = img[y1:y2, x1:x2]
    return img

def defend_CROP(img,minlimit=0.66,w2h=0.91):
    crop_height = random.randint(int(img.shape[1]*minlimit),img.shape[1])
    crop_width = int(crop_height*w2h)
    h_start = random.random()
    w_start = random.random()
    return resize(random_crop(img, crop_height, crop_width, h_start, w_start),img.shape)


"""
RAND*: Random padding
*not published yet

scalimit: the maximum scale
"""
def defend_RAND(img,scalimit=1.3):
    maxvalue = np.int(img.shape[0] * scalimit)
    rnd = np.random.randint(img.shape[0],maxvalue,(1,))[0]
    rescaled = resize(img,(rnd,rnd))
    h_rem = maxvalue - rnd
    w_rem = maxvalue - rnd
    pad_left = np.random.randint(0,w_rem,(1,))[0]
    pad_right = w_rem - pad_left
    pad_top = np.random.randint(0,h_rem,(1,))[0]
    pad_bottom = h_rem - pad_top
    padded = np.pad(rescaled,((pad_top,pad_bottom),(pad_left,pad_right),(0,0)),'constant',constant_values = 0.5)
    padded = resize(padded,(img.shape[0],img.shape[0]))
    return padded

"""
ET: Elastic Transformation
Simard, Patrice Y., David Steinkraus, and John C. Platt. "Best practices for convolutional neural networks applied to visual document analysis." Icdar. Vol. 3. No. 2003. 2003.
alpha:60
sigma:10
aaf: 20
"""
def defend_ET(img,alpha=60, sigma=10, alpha_affine=20):
    aug = albumentations.ElasticTransform(p=1, alpha=alpha, sigma=sigma, alpha_affine=alpha_affine)
    augmented = aug(image=img.astype(np.float32))
    auged = augmented['image']
    return auged