import cv2
import numpy as np
import os
from matplotlib import pyplot as plt


def video_process_canny(frames, params, masked_first_frame):
    video = []

    cord = find_rect(masked_first_frame)
    img1 = frames[0].copy()  # считываю первый кадр


    dispers1 = np.var(
        img1[cord[0][1]:cord[0][1] + cord[0][3], cord[0][0]:cord[0][0] + cord[0][2]])  # дисперсия для первой области
    dispers2 = np.var(
        img1[cord[1][1]:cord[1][1] + cord[1][3], cord[1][0]:cord[1][0] + cord[1][2]])  # дисперсия для второй области

    for i in range(0, len(frames)):
        img1 = frames[i].copy()
        clahe_img = prepare(img1)  # тут делаем эквализацию и тд
        x1, y1, dispers1 = poisk(clahe_img, cord[0], dispers1,
                                 5)  # получаем положение окна и значение дисперсии для области 1
        x2, y2, dispers2 = poisk(clahe_img, cord[1], dispers2,
                                 5)  # получаем положение окна и значение дисперсии для области 2

        cord = [(x1, y1, cord[0][2], cord[0][3]), (x2, y2, cord[1][2], cord[1][3])]

        mask = canny_segment(clahe_img, cord, img1, params)


        video.append(mask)
    return video

def prepare(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    filtered = cv2.medianBlur(gray, 7)

    cv2.imshow('Rectangles', filtered)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))

    clahe_img = clahe.apply(filtered)

    return clahe_img


def find_rect(masked_frame): # ищет положение размеченных областей

    gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGRA2GRAY)

    ret1, threshold = cv2.threshold(gray, 1, 255, cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rectangles = [cv2.boundingRect(cnt) for cnt in contours]
    coord_rectangles = []

    for rect in rectangles:
        coord_rectangles.append(rect)
        cv2.rectangle(masked_frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 2)

    cv2.imshow('Rectangles', masked_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return coord_rectangles


def canny_segment(clahe_img, cord, img1, params):
    ret1, bin_img = cv2.threshold(clahe_img, 100, 255, cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)


    # dilateCanny = cv2.dilate(outputCanny, kernel, iterations=3)

    # morph_img = cv2.morphologyEx(bin_img, cv2.MORPH_GRADIENT, kernel)

    # morph_img = cv2.morphologyEx(morph_img, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = np.ones_like(img1)

    for i in cord:
        outputCanny = cv2.Canny(bin_img[i[1]:i[1] + i[3], i[0]:i[0] + i[2]], params[0][0], params[0][1])

        outputCanny = cv2.dilate(outputCanny, kernel, iterations=2)
        contours, hierarchy = cv2.findContours( outputCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Замкнуть незамкнутые контуры

        for contour in contours:
            if not cv2.isContourConvex(contour):
                epsilon = params[1] * cv2.arcLength(contour, True) # апроксимация контуров
                approx = cv2.approxPolyDP(contour, epsilon, True)

                cv2.fillPoly(mask[i[1]:i[1] + i[3], i[0]:i[0] + i[2]], [approx], (255, 255, 255))
                #cv2.drawContours(img1[i[1]:i[1] + i[3], i[0]:i[0] + i[2]], [approx], 0, (0, 0, 255), 2)
    return  mask


def poisk(img, h, dispersia, b=5):  # функция ищет положение области интереса для следующего кадра
# параметр b означает на сколько мы расширяем область интереса. И потом по этой расширенной области окном проходим и
# смотрим где больше совпадает дисперсия.

# h[1] - это y
# h[0] - это x
# h[2] - ширина окна
# h[3] - высота


    disp = []  # в этот список будем записывать дисперсию для каждого положения окна и координаты. потом сравним какое
    # значение наиболее близко к значению дисперсии прошлого кадра.
    disp_cord = []



    for y in range(img[h[1]-b:h[1] + h[3]+b, h[0]-b:h[0] + h[2]+b].shape[0]):
        if y + h[3] > img[h[1]-b:h[1] + h[3]+b, h[0]-b:h[0] + h[2]+b].shape[0]:  # проверка на выход за край по вертикали
            break
        else:
            for x in range(img[h[1]-b:h[1] + h[3]+b, h[0]-b:h[0] + h[2]+b].shape[1]):
                if x + h[2] > img[h[1]-b:h[1] + h[3]+b, h[0]-b:h[0] + h[2]+b].shape[1]:  # проверка на выход за край по горизонтали
                    break
                else:
                    disp.append(np.var(img[y+h[1]-b:y+h[1]-b + h[3], x+h[0]-b:x+h[2]+h[0]-b]))
                    disp_cord.append((x, y))

    result_dispers, result_index = find_closest_value(disp, dispersia)


    return disp_cord[result_index][0]-b+h[0], disp_cord[result_index][1]-b+h[1], result_dispers


def find_closest_value(numbers, target):
    closest_value = min(numbers, key=lambda x: abs(x - target))
    closest_index = numbers.index(closest_value)
    return closest_value, closest_index

path_to_frames = 'C:\\Users\\Asus\\Desktop\\education\\masters\\KP_OMI\\test2\\orig2\\' #папка со всеми кадрами
masked_first_frame = cv2.imread('C:\\Users\\Asus\\Desktop\\education\\masters\\KP_OMI\\test2\\masked\\1.png') #первый кадр(который размечен). По нему отслеживаем
path_to_folder = 'C:\\Users\\Asus\\Desktop\\education\\masters\\KP_OMI\\test2\\canny\\' # сохраняю итоговые кадры

params = [(10, 150), 0.001]
frames = []
for i in range(2):
    frame = cv2.imread(f'C:\\Users\\Asus\\Desktop\\education\\masters\\KP_OMI\\test2\\orig2\\frame{i}.jpg')
    frames.append(frame)


video = video_process_canny(frames, params, masked_first_frame)

cv2.imshow('vid', video[0])
cv2.imshow('vid1', video[1])
cv2.waitKey(0)
cv2.destroyAllWindows()

# тут видео собираю

output_video = 'C:\\Users\\Asus\\Desktop\\education\\masters\\KP_OMI\\test2\\canny\\result.mp4'

first_image = cv2.imread(os.path.join(path_to_folder , os.listdir(path_to_folder )[0]))
height, width, _ = first_image.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 30
video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

for image_name in os.listdir(path_to_folder ):
    image_path = os.path.join(path_to_folder , image_name)
    image = cv2.imread(image_path)
    video_writer.write(image)

video_writer.release()