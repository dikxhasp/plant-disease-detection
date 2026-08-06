import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

dataset_path="PlantVillage/PlantVillage"
IMAGE_SIZE=(224,224)
BATCH_SIZE=32
EPOCHS=5

datagen=ImageDataGenerator(rescale=1./255,validation_split=0.2,rotation_range=20,zoom_range=0.2,horizontal_flip=True)

train_data=datagen.flow_from_directory(dataset_path,target_size=IMAGE_SIZE,batch_size=BATCH_SIZE,class_mode="categorical",subset="training")
validation_data=datagen.flow_from_directory(dataset_path,target_size=IMAGE_SIZE,batch_size=BATCH_SIZE,class_mode="categorical",subset="validation")

with open("labels.txt","w") as f:
    f.write("\n".join(train_data.class_indices.keys()))

base_model=MobileNetV2(weights="imagenet",include_top=False,input_shape=(224,224,3))
base_model.trainable=False

x=GlobalAveragePooling2D()(base_model.output)
x=Dense(128,activation="relu")(x)
x=Dropout(0.4)(x)
out=Dense(train_data.num_classes,activation="softmax")(x)

model=Model(base_model.input,out)
model.compile(optimizer=Adam(0.0001),loss="categorical_crossentropy",metrics=["accuracy"])
model.summary()

callbacks=[
EarlyStopping(monitor="val_loss",patience=2,restore_best_weights=True),
ModelCheckpoint("model.keras",monitor="val_accuracy",save_best_only=True)
]

history=model.fit(train_data,validation_data=validation_data,epochs=EPOCHS,callbacks=callbacks)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Accuracy")
plt.subplot(1,2,2)
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Loss")
plt.tight_layout()
plt.show()
print("Training completed.")
