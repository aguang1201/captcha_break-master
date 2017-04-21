from captcha.image import ImageCaptcha
import numpy as np
import random
import string
from tqdm import tqdm
from keras.models import *
from keras.layers import *
import tensorflow as tf
from mobilenet import MobileNet

characters = string.digits + string.ascii_uppercase
width, height, n_len, n_class = 170, 80, 4, len(characters)+1

def gen(batch_size=64):
    X = np.zeros((batch_size, height, width, 3), dtype=np.uint8)
    y = [np.zeros((batch_size, n_class), dtype=np.uint8) for i in range(n_len)]
    generator = ImageCaptcha(width=width, height=height)
    while True:
        for i in range(batch_size):
            random_str = ''.join([random.choice(characters) for j in range(4)])
            X[i] = generator.generate_image(random_str)
            for j, ch in enumerate(random_str):
                y[j][i, :] = 0
                y[j][i, characters.find(ch)] = 1
        yield X, y

def evaluate(model, batch_num=20):
    batch_acc = 0
    generator = gen()
    for i in tqdm(range(batch_num)):
        X, y = next(generator)
        y_pred = model.predict(X)
        y_pred = np.argmax(y_pred, axis=2).T
        y_true = np.argmax(y, axis=2).T
        batch_acc += np.mean(map(np.array_equal, y_true, y_pred))
    return batch_acc / batch_num

def create_model():
    input_tensor = Input((height, width, 3))
    x = input_tensor
    for i in range(4):
        x = Convolution2D(32*2**i, 3, 3, activation='relu')(x)
        x = Convolution2D(32*2**i, 3, 3, activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)

    x = Flatten()(x)
    x = Dropout(0.25)(x)
    x = [Dense(n_class, activation='softmax', name='c%d'%(i+1))(x) for i in range(4)]
    model = Model(input=input_tensor, output=x)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    return model

config  = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.add_to_collection('graph_config', config)

model = create_model()
if os.path.isfile('cnn.h5'):
    model.load_weights('cnn.h5')

#model = MobileNet()
model.fit_generator(gen(), samples_per_epoch=51200, nb_epoch=5,
                    nb_worker=2, pickle_safe=True,
                    validation_data=gen(), nb_val_samples=1280)

evaluate(model)

model.save('cnn.h5')
#model.save('mobileNet.h5')