import matplotlib.pyplot as plt
from captcha_CNN import *

def decode(y):
    y = np.argmax(np.array(y), axis=2)[:,0]
    return ''.join([characters[x] for x in y])

#model=cm.create_model()
X, y = next(gen(1))
model.load_weights('cnn.h5')
y_pred = model.predict(X)
plt.title('real: %s\npred:%s'%(decode(y), decode(y_pred)))
plt.imshow(X[0], cmap='gray')