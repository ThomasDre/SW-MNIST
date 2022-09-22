"""
Usage: 
    load_run -f=<filename>

Options:
    -f=<filename>
"""

from cProfile import label
import tensorflow as tf
from docopt import docopt
import cv2
import numpy as np
import matplotlib.pyplot as plt


def resize_to_28x28(img):
    img_h, img_w = img.shape
    dim_size_max = max(img.shape)

    if dim_size_max == img_w:
        im_h = (26 * img_h) // img_w
        if im_h <= 0 or img_w <= 0:
            print("Invalid Image Dimention: ", im_h, img_w, img_h)
        tmp_img = cv2.resize(img, (26,im_h),0,0,cv2.INTER_NEAREST)
    else:
        im_w = (26 * img_w) // img_h
        if im_w <= 0 or img_h <= 0:
            print("Invalid Image Dimention: ", im_w, img_w, img_h)
        tmp_img = cv2.resize(img, (im_w, 26),0,0,cv2.INTER_NEAREST)

    out_img = np.zeros((28, 28), dtype=np.ubyte)

    nb_h, nb_w = out_img.shape
    na_h, na_w = tmp_img.shape
    y_min = (nb_w) // 2 - (na_w // 2)
    y_max = y_min + na_w
    x_min = (nb_h) // 2 - (na_h // 2)
    x_max = x_min + na_h

    out_img[x_min:x_max, y_min:y_max] = tmp_img

    return out_img

def run_inference(img):
    tsr_img = resize_to_28x28(img)
    input_data = np.copy(tsr_img).reshape(1,28,28,1)

    # if floating_model:
    #    input_data = (np.float32(input_data) - input_mean) / input_std

    # Instantiate tensorflow interpreter and run inference on input_data


if __name__ == '__main__':
    arguments = docopt(__doc__, version='Naval Fate 2.0')
    file = "./images/testSample/" + arguments["-f"]

    image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    
    print(type(image))
    print(image)
    print(image.shape)


    #run_inference(image)


    print(image.shape)

    input = image.reshape(1,28,28,1)

    model = tf.keras.models.load_model('model') 
    pred = model.predict(input)
    probs = np.exp(pred) / np.sum(np.exp(pred))
    
    print(pred)
    print(probs)


    x = np.arange(10)
    fig = plt.figure()
    ax1 = fig.add_subplot(1,2,1)
    ax1.bar(x, probs[0])
    ax1.set_title("Probabiliy")
    ax1.set_xticks(x, ('0', '1', '2', '3','4','5','6','7','8','9'))
    ax2 = fig.add_subplot(1,2,2)
    ax2.set_title("Input")
    ax2.imshow(image)
    plt.show()


