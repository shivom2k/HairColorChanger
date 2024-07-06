import cv2
import keras
import numpy as np
import streamlit as st
import PIL
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageColor

url = 'https://sphinx-phoenix.github.io/BarberzBuzz/'

st.markdown(f'''
<a href={url}><button style="background-color: #EA4C89;border-radius: 8px;border-style: none;box-sizing: border-box;color:#FFFFFF;">Go Back</button></a>''',unsafe_allow_html=True)






######Load image##########
def load_image(image_file):
    img = Image.open(image_file)
    return img



st.title("Hairstyle generator")

model = keras.models.load_model(
    r'checkpoints/hairnet_matting_30.hdf5')   # Load saved model
model.summary()



#######Functions to change hair color##########
def imShow(image):
    height, width = image.shape[:2]
    resized_image = cv2.resize(
        image, (3 * width, 3 * height), interpolation=cv2.INTER_CUBIC)

    fig = plt.gcf()
    fig.set_size_inches(18, 10)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
    plt.show()


def predict(image, height=224, width=224):
    im = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    im = im / 255
    im = cv2.resize(im, (height, width))
    im = im.reshape((1,) + im.shape)

    pred = model.predict(im)

    mask = pred.reshape((224, 224))

    return mask


def Change_hair_color(image, color):
    global thresh

    #image = cv2.imread(img)
    mask = predict(image)

    kernel = np.ones((1, 1), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=1)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    thresh = 0.60  # Threshold used on mask pixels


    blue_mask = mask.copy()
    blue_mask[mask > thresh] = color[0]
    blue_mask[mask <= thresh] = 0

    green_mask = mask.copy()
    green_mask[mask > thresh] = color[1]
    green_mask[mask <= thresh] = 0

    red_mask = mask.copy()
    red_mask[mask > thresh] = color[2]
    red_mask[mask <= thresh] = 0

    blue_mask = cv2.resize(blue_mask, (image.shape[1], image.shape[0]))
    green_mask = cv2.resize(green_mask, (image.shape[1], image.shape[0]))
    red_mask = cv2.resize(red_mask, (image.shape[1], image.shape[0]))


    mask_n = np.zeros_like(image)
    mask_n[:, :, 0] = blue_mask
    mask_n[:, :, 1] = green_mask
    mask_n[:, :, 2] = red_mask

    alpha = 0.90
    beta = (1.0 - alpha) * 3
    out = cv2.addWeighted(image, alpha, mask_n, beta, 0.0)
    return out

    # name = 'test/results/new.jpg'
    # imShow(out)
    # cv2.imwrite(name, out)


# img_path = "soft copy.jpg"
# img = cv2.imread(img_path)

# print(img)

color= st.color_picker('Pick A Hair Color', '#00f900')
val=ImageColor.getcolor(color, "RGB")




#color = [124, 100, 250]  # Color to be used on hair

file = st.file_uploader("Please upload an image file", type=["jpg", "png", "jpeg"])


if file is not None:
    # Perform your Manupilations (In my Case applying Filters)
    img = load_image(file)
    st.write("Original Image")
    st.image(img)
    img1= np.array(img)
    im=Change_hair_color(img1, val)
    st.write("Image with selected hair color")

    st.image(im)




else:
    st.write("Make sure you image is in JPG/PNG/JPEG Format.")




#out_image=Change_hair_color(file, color)




