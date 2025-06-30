#обработка и рисование изображений
import os
import json
import numpy as np
import skimage.draw
import cv2  # работа с масками
import gdown  # скачивание архива
import shutil  # распаковка архива

# обработка и рисование изображений
import imageio
from PIL import Image

# рисование графиков
import matplotlib.pyplot as plt

# для создание модели
import tensorflow as tf
# Слои, которые потребуются
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose  # транспонированная свертка
from tensorflow.keras.layers import concatenate  # объединение тензоров

from tensorflow.keras.losses import binary_crossentropy  # функция ошибки
from sklearn.model_selection import train_test_split  # разделение данных на обучающее и проверочное множества

# Загрузка датасета
url = 'https://drive.google.com/uc?export=download&confirm=y&id=1jQ5IROvRCphNRJ_PJKUiOdmEUGn66xOD'
output = 'data.zip'
gdown.download(url, output, quiet=False)

shutil.unpack_archive('data.zip', "/content/")

# Обозначение испольуемых данных (картинки + json)
IMAGE_FOLDER = "/content/images"
PATH_ANNOTATION_JSON = 'sagittals.json'
annotations = json.load(open(PATH_ANNOTATION_JSON, 'r'))
imgs = annotations["_via_img_metadata"]

# Рисование масок
mask_ = []
img_ = []
for imgId in imgs:
    filename = imgs[imgId]['filename']
    regions = imgs[imgId]['regions']
    if len(regions) <= 0:
        continue
    points = []
    for ln in range(0, len(regions)):
        polygons = regions[ln]['shape_attributes']

        image_path = os.path.join(IMAGE_FOLDER, filename)

        # чтение высоты+ширины картинки
        image = cv2.imread(image_path)
        height, width = image.shape[:2]

        # создание пустой (чёрной) маски
        maskImage = np.zeros((height, width), dtype=np.uint8)
        countOfPoints = len(polygons['all_points_x'])
        for i in range(countOfPoints):
            x = int(polygons['all_points_x'][i])
            y = int(polygons['all_points_y'][i])
            points.append((x, y))

        contours = np.array(points)
        contours = contours.reshape((-1, 1, 2))

        # рисование
        for i in range(width):
            for j in range(height):
                if cv2.pointPolygonTest(contours, (i, j), False) > 0:
                    maskImage[j, i] = 1
        image = cv2.resize(image, (128, 128))
        img_.append(image/255.0)

        # сохранение результата
        maskImage = cv2.resize(maskImage, (128, 128))
        mask_.append(np.reshape(maskImage, (128, 128, 1)))

# Создание сети Unet
# Блок кодера
def EncoderMiniBlock(inputs,  # вход
                    n_filters=32,  # количество фильтров
                    dropout_prob=0.3,  # вероятность отброса
                    max_pooling=True):  # использовать ли субдискретизацию

    # Два сверточных слоя (выходы) с инициализацией.
    # Набивка 'Same' не изменяет пространственные размеры.
    conv = Conv2D(n_filters,
                  3,  # размер ядра
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(inputs)  # инициализация (только при создании)
    conv = Conv2D(n_filters,
                  3,  # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(conv)

    # Нормализация
    conv = BatchNormalization()(conv, training=False)

    # dropout, если задан
    if dropout_prob > 0:
        conv = tf.keras.layers.Dropout(dropout_prob)(conv)
    # Субдискретизация, если задано. Конкретно MaxPooling, уменьшает пространственные размеры в два раза
    if max_pooling:
        next_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
    else:
        next_layer = conv

    # выход слоев ДО субдискретизации, будет передаваться в другие блоки.
    skip_connection = conv

    return next_layer, skip_connection  # возвращаем выход блока и выход слоев до субдискретизации


# Блок декодера
def DecoderMiniBlock(prev_layer_input,  # выход предыдущего слоя (блока)
                    skip_layer_input,  # выход соответствующего блока кодера
                    n_filters=32):  # число фильтров

    # Транспонированная свертка увеличивает пространственный размер карты признаков в два раза
    up = Conv2DTranspose(
        n_filters,
        (3, 3),  # размер ядра
        strides=(2, 2),
        padding='same')(prev_layer_input)  # набивка, чтобы не уменьшался размер при выполнении свертки

    # Конкатенируем по каналам (измерение 3) выход транспонированной свертки и выход блока кодера
    merge = concatenate([up, skip_layer_input], axis=3)

    # Две свертки не изменяющие размеры, с инициализацией
    conv = Conv2D(n_filters,
                  3,  # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(merge)
    conv = Conv2D(n_filters,
                  3,  # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(conv)
    return conv

# Сборка Unet
def UNetCompiled(input_size=(128, 128, 3),  # размер изображения-входа
                n_filters=32,  # базовое число фильтров
                n_classes=3):  # число классов

    # Вход в сеть заданного размера
    inputs = Input(input_size)

    # Кодер
    # блок 1 принимает вход в сеть, число фильтров базовое, dropout нет, понижает размеры карты
    cblock1 = EncoderMiniBlock(inputs, n_filters, dropout_prob=0, max_pooling=True)
    # блок 2 принимает выход блока 1 (обратите внимание что блоки кодера возвращают два выхода),
    # число фильтров в два раза больше, dropout нет, понижает размеры карты
    cblock2 = EncoderMiniBlock(cblock1[0], n_filters*2, dropout_prob=0, max_pooling=True)
    # блок 3 принимает выход блока 2, число фильтров еще в два раза больше, dropout нет, понижает размеры карты
    cblock3 = EncoderMiniBlock(cblock2[0], n_filters*4, dropout_prob=0, max_pooling=True)
    # блок 4 принимает выход блока 3, число фильтров еще в два раза больше, dropout есть, понижает размеры карты
    cblock4 = EncoderMiniBlock(cblock3[0], n_filters*8, dropout_prob=0.3, max_pooling=True)
    # блок 5 принимает выход блока 4, число фильтров еще в два раза больше, dropout есть, НЕ понижает размеры карты
    cblock5 = EncoderMiniBlock(cblock4[0], n_filters*16, dropout_prob=0.3, max_pooling=False)

    # Декодер
    # блок 6 принимает выход блока 5, и второй выход (т.е. до слоя субдискретизации) блока 4,
    # число фильтров в два раза меньше чем у блока 5, повышает размеры карты
    ublock6 = DecoderMiniBlock(cblock5[0], cblock4[1], n_filters * 8)
    # блок 7 принимает выход блока 6, и второй выход блока 3, число фильтров в два раза меньше, повышает размеры карты
    ublock7 = DecoderMiniBlock(ublock6, cblock3[1], n_filters * 4)
    # блок 8 принимает выход блока 7, и второй выход блока 2, число фильтров в два раза меньше, повышает размеры карты
    ublock8 = DecoderMiniBlock(ublock7, cblock2[1], n_filters * 2)
    # блок 9 принимает выход блока 8, и второй выход блока 1, число фильтров в два раза меньше, повышает размеры карты
    ublock9 = DecoderMiniBlock(ublock8, cblock1[1], n_filters)

    # слой свертки без изменения размеров карты, число фильтров как у предыдущего блока
    conv9 = Conv2D(n_filters,
                   3,
                   activation='relu',
                   padding='same',
                   kernel_initializer='he_normal')(ublock9)
    # слой свертки без изменения размеров карты, число фильтров по количеству классов
    conv10 = Conv2D(n_classes, 1, padding='same')(conv9)

    # Создаем модель из слоев
    model = tf.keras.Model(inputs=inputs, outputs=conv10)
    return model

for i in range(len(mask_)):
    img_view = img_[i]  # путь к изображению
    mask_view = mask_[i]  # путь к его маске

    fig, arr = plt.subplots(1, 2, figsize=(15, 15))
    arr[0].imshow(img_view)  # рисуем изображение
    arr[0].set_title('Image ' + str(i))
    arr[1].imshow(mask_view)  # рисуем маску
    arr[1].set_title('Masked Image ' + str(i))

# Обработка
X = np.array(img_)
y = np.array(mask_)

print("X Shape:", X.shape)
print("Y shape:", y.shape)
# Классы : фон, объект, контур
print(np.unique(y))
# Демонстрация
image_index = 0
fig, arr = plt.subplots(1, 2, figsize=(15, 15))
arr[0].imshow(X[image_index])
arr[0].set_title('Processed Image')
arr[1].imshow(y[image_index, :, :])
arr[1].set_title('Processed Masked Image ')

# Разделяем данные на обучающие проверочные (80/20%)
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=123)

# Создаем сеть, 3 класса, 32 базовых каналов
unet = UNetCompiled(input_size=(128, 128, 3), n_filters=32, n_classes=3)
unet.summary()
# оптимизатор Adam, функция ошибки - кроссэнтропия которая применяет softmax к выходам сети, метрика - аккуратность
unet.compile(optimizer=tf.keras.optimizers.Adam(),
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics=['accuracy'])

# Обучение сети (40 эпох, время ~ 12 минут)
results = unet.fit(X_train, y_train, batch_size=4, epochs=40, validation_data=(X_valid, y_valid))

fig, axis = plt.subplots(1, 2, figsize=(20, 5))
axis[0].plot(results.history["loss"], color='r', label='train loss')
axis[0].plot(results.history["val_loss"], color='b', label='dev loss')
axis[0].set_title('Loss Comparison')
axis[0].legend()
axis[1].plot(results.history["accuracy"], color='r', label='train accuracy')
axis[1].plot(results.history["val_accuracy"], color='b', label='dev accuracy')
axis[1].set_title('Accuracy Comparison')
axis[1].legend()

# Качество обучения
unet.evaluate(X_valid, y_valid)

# Демонстрация работы сети
def VisualizeResults(index):
    img = X_valid[index]  # изображение
    img = img[np.newaxis, ...]  # добавляем измерение примеров
    pred_y = unet.predict(img)  # расчитываем маску
    pred_mask = tf.argmax(pred_y[0], axis=-1)  # выбираем максимальный класс (канал - последнее измерение)
    pred_mask = pred_mask[..., tf.newaxis]  # канальное измерение добавляем
    # рисуем
    fig, arr = plt.subplots(1, 3, figsize=(15, 15))
    arr[0].imshow(X_valid[index])
    arr[0].set_title('Processed Image')
    arr[1].imshow(y_valid[index, :, :])
    arr[1].set_title('Actual Masked Image ')
    arr[2].imshow(pred_mask[:, :, 0])
    arr[2].set_title('Predicted Masked Image ')

index = 1
VisualizeResults(index)
