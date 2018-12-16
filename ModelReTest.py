from tensorflow.python.keras.models import load_model

import DataProcess
from CallBack import LossHistory

model = load_model('my_model_0.h5')
model.summary()
x_train, y_train, x_test, y_test, class_names = DataProcess.readData(path='D:\\迅雷下载\\npy',start= 10001,each_count=12000)



history = LossHistory()
model.fit(x_train,y_train,batch_size=40,epochs=10,callbacks=[history])


model.save('my_model_1.h5')
score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
history.loss_plot('epoch')

