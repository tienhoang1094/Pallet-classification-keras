from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers.schedules import ExponentialDecay

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet',input_shape=(300,300,3), include_top=False)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
#x = Dense(1024, activation='relu')(x)
#x = Dropout(0.5)(x)
x = Dense(512,activation='relu')(x)
x = Dropout(0.3)(x)
# and a logistic layer -- let's say we have 3 classes
predictions = Dense(3, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers[:200]:
    layer.trainable = False

lr_schedule = ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=500,
    decay_rate=0.9)
optimizer = SGD(learning_rate=lr_schedule, momentum=0.9)

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(optimizer=optimizer, loss='categorical_crossentropy')


model_file = "models/epoch{epoch:02d}--test{loss:06f}--val{val_loss:.6f}.hdf5"
checkpoint = ModelCheckpoint(model_file, monitor='val_loss',verbose=1, save_best_only=True,mode='min')
callbacks_list = [checkpoint]

model.summary()


train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=1,
        width_shift_range=10,
        rotation_range = 1,
        brightness_range=[0,2],
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        './data/train',
        target_size=(300, 300),
        batch_size=12,
        shuffle=True)
validation_generator = test_datagen.flow_from_directory(
        './data/val',
        target_size=(300, 300),
        batch_size=12,
        shuffle=True)
model.fit(
        train_generator,
        steps_per_epoch=688//12,
        epochs=120,
        validation_data=validation_generator,
        validation_steps=85//12,
        callbacks=callbacks_list)
