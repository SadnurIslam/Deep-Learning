'''

Task: Create a Keras model with specified layers
This code creates a Keras model with the specified architecture.

'''


from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
inputs = Input((1,))
h1 = Dense(3, activation='relu')(inputs)
h2 = Dense(2, activation='relu')(h1)
h3 = Dense(2, activation='relu')(h2)
outputs = Dense(1, activation='softmax')(h3)
model = Model(inputs, outputs)
model.summary()
