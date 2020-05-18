from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


train_datagen = ImageDataGenerator(
        # rescale=1./255,
        # shear_range=1,
        zoom_range=0.1,
        # brightness_range=[0,2],
        horizontal_flip=False)
test_datagen = ImageDataGenerator(rescale=1./255)
img = load_img('/home/peter-linux/Desktop/AGF/data2/train/NG_plastic/6a630.jpg')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in train_datagen.flow(x, batch_size=1,
                          save_to_dir='preview', save_prefix='cat', save_format='jpeg'):
    i += 1
    if i > 20:
        break