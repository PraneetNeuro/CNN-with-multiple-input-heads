import tensorflow as tf

def MI_CNN():
  
  # Weight initializer
  initializer = tf.keras.initializers.HeNormal()

  # Input layers
  input1 = tf.keras.layers.Input((28,28,1))
  input2 = tf.keras.layers.Input((28,28,1))

  # Convolutional -> Pooling -> Convolutional layers for Input1
  conv1 = tf.keras.layers.Conv2D(75, (5,5), kernel_initializer=initializer)(input1)
  conv1 = tf.keras.layers.LeakyReLU()(conv1)
  conv1 = tf.keras.layers.MaxPool2D()(conv1)
  conv1 = tf.keras.layers.Conv2D(150, (3,3), kernel_initializer=initializer)(conv1)
  conv1 = tf.keras.layers.LeakyReLU()(conv1)
  maxPool1 = tf.keras.layers.MaxPool2D()(conv1)
  conv2 = tf.keras.layers.Conv2D(300, (3,3), kernel_initializer=initializer)(maxPool1)
  conv2 = tf.keras.layers.LeakyReLU()(conv2)
  conv2 = tf.keras.layers.MaxPool2D()(conv2)

  # Convolutional -> Pooling -> Convolutional layers for Input2
  conv1_ = tf.keras.layers.Conv2D(75, (5,5), kernel_initializer=initializer)(input2)
  conv1_ = tf.keras.layers.LeakyReLU()(conv1_)
  conv1_ = tf.keras.layers.MaxPool2D()(conv1_)
  conv1_ = tf.keras.layers.Conv2D(150, (3,3), kernel_initializer=initializer)(conv1_)
  conv1_ = tf.keras.layers.LeakyReLU()(conv1_)
  maxPool1_ = tf.keras.layers.MaxPool2D()(conv1_)
  conv2_ = tf.keras.layers.Conv2D(300, (3,3), kernel_initializer=initializer)(maxPool1_)
  conv2_ = tf.keras.layers.LeakyReLU()(conv2_)
  conv2_ = tf.keras.layers.MaxPool2D()(conv2_)

  # Flatten the output of the convolutional layers for Input1 and Input2
  x_1 = tf.keras.layers.Flatten()(conv2)
  x_2 = tf.keras.layers.Flatten()(conv2_)

  # Compose the latent representation of the digit represented in Input1
  x_1 = tf.keras.layers.Dense(512, kernel_initializer=initializer)(x_1)
  x_1 = tf.keras.layers.LeakyReLU()(x_1)
  x_1 = tf.keras.layers.Dropout(0.2)(x_1)
  x_1 = tf.keras.layers.Dense(256, kernel_initializer=initializer)(x_1)
  x_1 = tf.keras.layers.LeakyReLU()(x_1)
  x_1 = tf.keras.layers.Dropout(0.2)(x_1)
  x_1 = tf.keras.layers.Dense(128, kernel_initializer=initializer)(x_1)
  x_1 = tf.keras.layers.LeakyReLU()(x_1)
  x_1 = tf.keras.layers.Dropout(0.2)(x_1)
  
  # Compose the latent representation of the digit represented in Input2
  x_2 = tf.keras.layers.Dense(512, kernel_initializer=initializer)(x_2)
  x_2 = tf.keras.layers.LeakyReLU()(x_2)
  x_2 = tf.keras.layers.Dropout(0.2)(x_2)
  x_2 = tf.keras.layers.Dense(256, kernel_initializer=initializer)(x_2)
  x_2 = tf.keras.layers.LeakyReLU()(x_2)
  x_2 = tf.keras.layers.Dropout(0.2)(x_2)
  x_2 = tf.keras.layers.Dense(128, kernel_initializer=initializer)(x_2)
  x_2 = tf.keras.layers.LeakyReLU()(x_2)
  x_2 = tf.keras.layers.Dropout(0.2)(x_2)
  
  # Concatenate the latent representations of Input1 and Input2
  x = tf.keras.layers.Concatenate()([x_1, x_2])

  # Compose the final representation of the digits represented in Input1 and Input2 which is then used to predict the sum of the digits
  x = tf.keras.layers.Dense(512, kernel_initializer=initializer)(x)
  x = tf.keras.layers.LeakyReLU()(x)
  x = tf.keras.layers.Dropout(0.2)(x)
  x = tf.keras.layers.Dense(128, kernel_initializer=initializer)(x)
  x = tf.keras.layers.LeakyReLU()(x)
  x = tf.keras.layers.Dropout(0.2)(x)
  x = tf.keras.layers.Dense(64, kernel_initializer=initializer)(x)
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.Dropout(0.2)(x)
  x = tf.keras.layers.Dense(32, kernel_initializer=initializer)(x)
  x = tf.keras.layers.LeakyReLU()(x)
  x = tf.keras.layers.Dropout(0.2)(x)
  x = tf.keras.layers.Dense(19, 'softmax', kernel_initializer=initializer)(x)
  
  return tf.keras.models.Model([input1, input2], x)
