import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Reshape, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization

import DataProcess
from CallBack import LossHistory

batch_size = 12
epochs = 10
# 训练数据
x_train, y_train, x_test, y_test, class_names = DataProcess.readData(path='D:\\迅雷下载\\npy',start= 0,each_count=10000)

model = Sequential()
# 模型转换
model.add(Reshape(target_shape=(28, 28, 1), input_shape=(784,)))
# 数据处理
#model.add(Lambda(lambda x: float(x) / 255))
model.add(BatchNormalization())

#第一段
model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='valid'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))

#第二段l
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
#第三段
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))

#第三段
#model.add(Conv2D(filters=128, kernel_size=(3, 3),strides=(1, 1), padding='same',activation='relu'))
#model.add(Conv2D(filters=128, kernel_size=(3, 3),strides=(1, 1), padding='same',activation='relu'))
#model.add(Conv2D(filters=100, kernel_size=(3, 3),strides=(1, 1), padding='same',activation='relu'))
#model.add(MaxPooling2D(pool_size=(3, 3),strides=(2, 2), padding='valid'))

#第四段
model.add(Flatten())
# model.add(Dense(256, activation='relu'))
# model.add(Dropout(0.5))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))

model.add(Dense(100, activation='relu'))
model.add(Dropout(0.25))

# Output Layer
model.add(Dense(units=len(class_names), activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])

history = LossHistory()
model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=1,callbacks=[history])

model.save('my_model_0.h5')
score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
history.loss_plot('epoch')
