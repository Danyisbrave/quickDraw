from tensorflow.python.keras.models import load_model
import matplotlib.pyplot as plt
import DataProcess
from CallBack import LossHistory

x_train, y_train, x_test, y_test, class_names = DataProcess.readData(path='D:\\迅雷下载\\npy', start=50000,
                                                                     each_count=55000)

model = load_model('my_model_2.h5')

history = LossHistory()

score = model.evaluate(x_train, y_train, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
history.loss_plot('epoch')
