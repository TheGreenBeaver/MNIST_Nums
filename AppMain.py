import numpy
from scipy import ndimage
from PIL import Image

# Disabling tensorflow GPU register warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow import keras


# Loading the image the user wants to examine
path = str(input('Path to the image of number:'))
# Example images at img5.png, img7.jpg and img9.jpg
image = Image.open(path)
width = image.size[0]
height = image.size[1]
pix = image.load()

left_border = width - 1
right_border = 0
top_border = height - 1
bottom_border = 0

step_1 = [[] for _ in range(height)]
# If the inverted gray value of a pixel is lower, it's considered blank to raise the contrast of the image
white_gray_threshold = 0.6

# Replacing all the colors in the picture with shades of gray
for i in range(width):
    for j in range(height):

        # Red, green and blue channel values for each pixel
        r = pix[i, j][0]
        g = pix[i, j][1]
        b = pix[i, j][2]

        gray = (r + g + b) // 3
        inv_gray = (255 - gray) / 255
        threshold_inv_gray = 0 if inv_gray < 0.6 else inv_gray

        if threshold_inv_gray != 0:
            if i < left_border:
                left_border = i
            if i > right_border:
                right_border = i
            if j < top_border:
                top_border = j
            if j > bottom_border:
                bottom_border = j

        step_1[j].append(threshold_inv_gray)

# Discarding the parts of the image that almost certainly don't contain any useful information
step_2 = numpy.array([line[left_border:right_border + 1] for line in step_1[top_border:bottom_border + 1]])
width = right_border - left_border + 1
height = bottom_border - top_border + 1

'''
The images in the set that the model was learned on were centered along their pixel center of mass, so the user's images
should be pre-processed in the same way
'''
center_of_mass = ndimage.center_of_mass(step_2)
center_x = int(center_of_mass[1])
center_y = int(center_of_mass[0])

# These values define where the supporting blocks of pixels would be placed
to_left = width - center_x > center_x
to_top = height - center_y > center_y

# Generating and appending the aligning blocks pf pixels
h_main = max(width - center_x - 1, center_x)
h_slave = width - h_main - 1
h_addition = h_main - h_slave
new_width = width + h_addition
h_addition_block = numpy.zeros((height, h_addition))
stack_order = (h_addition_block, step_2) if to_left else (step_2, h_addition_block)
step_3 = numpy.hstack(stack_order)

v_main = max(height - center_y - 1, center_y)
v_slave = height - v_main - 1
v_addition = v_main - v_slave
new_height = height + v_addition
v_addition_block = numpy.zeros((v_addition, new_width))
stack_order = (v_addition_block, step_3) if to_top else (step_3, v_addition_block)
step_4 = numpy.vstack(stack_order)

# Making the image square, same as the ones the model was trained on
landscape = new_width > new_height
bigger = max(new_height, new_width)
smaller = min(new_height, new_width)

whole_eq_addition = bigger - smaller

eq_addition_1 = whole_eq_addition // 2
eq_block_1_shape = (eq_addition_1, new_width) if landscape else (new_height, eq_addition_1)
eq_block_1 = numpy.zeros(eq_block_1_shape)

eq_addition_2 = whole_eq_addition - eq_addition_1
eq_block_2_shape = (eq_addition_2, new_width) if landscape else (new_height, eq_addition_2)
eq_block_2 = numpy.zeros(eq_block_2_shape)

step_5 = numpy.vstack((eq_block_1, step_4, eq_block_2)) if landscape else numpy.hstack((eq_block_1, step_4, eq_block_2))

# Resizing the user's image to the size of the ones that the model was trained on
training_img_size = 28
img = Image.fromarray(step_5 * 255)
r_img = img.resize(size=(training_img_size, training_img_size))
width = r_img.size[0]
height = r_img.size[1]

pix = r_img.load()
work_with = [[] for _ in range(28)]

# Generating the final matrix to feed the model
for i in range(width):
    for j in range(height):
        # Ensuring that no values have dropped below 0 due to calculation inaccuracy
        work_with[j].append(max(0, pix[i, j] / 255))

# Loading the trained model
model = keras.models.load_model('learnedModel')
# The original model worked with one-dimensional vectors, so the 28*28 matrix should be reshaped the proper way
prediction = model.predict(numpy.array(work_with).reshape((1, training_img_size ** 2)))
mp = max(prediction[0])
print(list(prediction[0]).index(mp))
