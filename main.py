import os
import cv2
import imutils
import numpy as np

mainFolder = 'images'  # Папка в которой хранятся изображения
path = os.listdir(mainFolder)  # Путь до необходимой папки
imgs = [cv2.imread(f'{mainFolder}/{img}') for img in path]  # Формируем список изображений

# Создаем сшивание + получаем статус со сшитой фотографией
stitcher = cv2.Stitcher.create()
status, pan = stitcher.stitch(imgs)

if status == cv2.STITCHER_OK:

    print('[INFO] cropping images...')
    
    # создаем рамку размером 10 пикселей, окружающую сшитое изображение
    pan = cv2.copyMakeBorder(pan, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0))

    # преобразуем сшитое изображение в оттенки серого + задаем порог
    # таким образом, чтобы для всех пикселей, превышающих ноль, было установлено значение 255
    # т.е бинаризуем изображение
    
    gray = cv2.cvtColor(pan, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
    
    # находим все внешние контуры на изображении порога, затем находим
    # самый большой контур, который будет контуром сшитого изображение
    
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    
    # выделяем память для маски, которая будет содержать
    # прямоугольную ограничивающую рамку области сшитого изображения
    mask = np.zeros(thresh.shape, dtype="uint8")
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    
    # создем две копии маски: одну, которая будет служить нашей реальной
    # минимальной прямоугольной областю и другая, служащая счетчиком
    # для того, сколько пикселей необходимо удалить, чтобы сформировать минимальное
    # прямоугольная область
    minRect = mask.copy()
    sub = mask.copy()

    # цикл работает пока в вычитаемом изображении не останется ненулевых пикселей
    while cv2.countNonZero(sub) > 0:
        # размываем минимальную прямоугольную маску, а затем вычетаем
        # пороговое изображение из минимальной прямоугольной маски
        # таким образом, мы можем посчитать, остались ли какие-либо ненулевые пиксели
        minRect = cv2.erode(minRect, None)
        sub = cv2.subtract(minRect, thresh)

    # найходим контуры в минимальной прямоугольной маске, а затем
    # извлекаем ограничивающую рамку (x, y) - координаты
    cnts = cv2.findContours(minRect.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    (x, y, w, h) = cv2.boundingRect(c)
    
    # использем координаты ограничивающей рамки, чтобы извлечь наше окончательное
    # сшитое изображение
    pan = pan[y:y + h, x:x + w]

    # Сохраняем и выводим на экран
    cv2.imwrite('panoram.jpeg', pan)
    cv2.imshow('pan img', pan)
    cv2.waitKey(0)

elif status == cv2.STITCHER_ERR_NEED_MORE_IMGS:
    print('[INFO] Требуется больше входных изображений для построения панорамы')
elif status == cv2.STITCHER_ERR_HOMOGRAPHY_EST_FAIL:
    print('[INFO] На ваших изображениях недостаточно отличительной, уникальной текстуры/объектов для точного '
          'сопоставления ключевых точек')
elif status == cv2.STITCHER_ERR_CAMERA_PARAMS_ADJUST_FAIL:
    print('[INFO] Невозможно правильно оценить внутренние/внешние характеристики камеры по входным изображениям')
else:
    print('[ERROR] Произошла не предвидемая ошибка!!!')
