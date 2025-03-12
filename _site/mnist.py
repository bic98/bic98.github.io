import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint
import keras

# Initialize Wandb
wandb.init(name = 'workshop_1', project="mnist_project-sgd")

# 데이터 생성
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 모델 생성
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation="sigmoid"),
    keras.layers.Dense(10, activation="softmax"),
])

# 컴파일 및 학습
model.compile(optimizer="sgd", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# Define the checkpoint callback
checkpoint_callback = WandbModelCheckpoint(filepath="models/mnins_sgd.keras", save_freq='epoch')

model.fit(
    x_train,
    y_train,
    epochs=128,
    validation_data=(x_test, y_test),
    callbacks=[
        WandbMetricsLogger(),
        checkpoint_callback,
    ],
)
