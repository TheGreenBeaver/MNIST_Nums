import numpy
from scipy import ndimage
from PIL import Image
from matplotlib import pyplot
from tensorflow import keras


path = str(input('Path to the image of number:'))
image = Image.open(path)
width = image.size[0]
height = image.size[1]
pix = image.load()

left_border = width - 1
right_border = 0
top_border = height - 1
bottom_border = 0

step_1 = [[] for _ in range(height)]

for i in range(width):
    for j in range(height):

        a = pix[i, j][0]
        b = pix[i, j][1]
        c = pix[i, j][2]

        gray = (a + b + c) // 3
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

step_2 = numpy.array([line[left_border:right_border + 1] for line in step_1[top_border:bottom_border + 1]])
width = right_border - left_border + 1
height = bottom_border - top_border + 1

center_of_mass = ndimage.center_of_mass(step_2)
center_x = int(center_of_mass[1])
center_y = int(center_of_mass[0])

to_left = width - center_x > center_x
to_top = height - center_y > center_y

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

img = Image.fromarray(step_5 * 255)
r_img = img.resize(size=(28, 28))
width = r_img.size[0]
height = r_img.size[1]

pix = r_img.load()
work_with = [[] for _ in range(28)]

for i in range(width):
    for j in range(height):
        work_with[j].append(max(0, pix[i, j] / 255))

pyplot.imshow(work_with)
pyplot.show()

# --- ---

model = keras.models.load_model('learnedModel')
prediction = model.predict(numpy.array(work_with).reshape((1, 784)))
mp = max(prediction[0])
print(list(prediction[0]).index(mp))






