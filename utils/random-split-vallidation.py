TRAIN_DIR = './datasets/training'
VALIDATION_DIR = './datasets/validation'



datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)



train_generator = datagen.flow_from_directory(
    TRAIN_DIR, 
    subset='training'
)

val_generator = datagen.flow_from_directory(
    TRAIN_DIR,
    subset='validation'
)