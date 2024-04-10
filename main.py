import face_recognition  # Импорт библиотеки для распознавания лиц
import os  # Импорт библиотеки для работы с операционной системой
import sys  # Импорт библиотеки для работы с системными параметрами
import cv2  # Импорт библиотеки для работы с изображениями
import math  # Импорт библиотеки для выполнения математических операций
import numpy as np  # Импорт библиотеки для работы с массивами данных


#статистика кол ва камер 
#госты 
#важность ии  в современном мире 



def face_confidence(face_distance, face_match_threshold=0.6):
    # Функция для вычисления уверенности в соответствии лиц
    range = (1.0 - face_match_threshold)  # Вычисление диапазона
    linear_value = (1.0 - face_distance) / (range * 2.0)  # Вычисление линейного значения

    # Если расстояние больше порогового значения, возвращаем линейное значение
    if face_distance > face_match_threshold:
        return str(round(linear_value * 100, 2)) + '%'  
    # Иначе, вычисляем уверенность с помощью формулы
    else:
        value = (linear_value + ((1.0 - linear_value) * math.pow((linear_value - 0.5) * 2, 0.2))) * 100  
        return str(round(value, 2)) + '%'  






class FaceRecognition:
    def __init__(self):
        self.known_face_encodings = []  # Список для известных кодировок лиц
        self.known_face_names = []  # Список для имен известных лиц
        self.encode_faces()  # Метод для кодирования известных лиц

    def encode_faces(self):
        # Проходим по изображениям в папке "faces"
        for image in os.listdir('faces'):
            face_image = face_recognition.load_image_file(f'faces/{image}')  # Загрузка изображения
            face_encoding = face_recognition.face_encodings(face_image)[0]  # Кодирование лица

            # Добавляем кодировку и имя в соответствующие списки
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image)

        print(self.known_face_names)  # Вывод списка имен известных лиц



    def run_recognition(self):
        video_capture = cv2.VideoCapture(0)  # Запуск захвата видео с камеры

        # Если видеоисточник не открыт, выводим сообщение об ошибке и завершаем программу
        if not video_capture.isOpened():
            sys.exit('video resource not found...')

        while True:
            ret, frame = video_capture.read()  # Получаем кадр с видеопотока

            if ret:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)  # Уменьшаем размер кадра
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)  # Преобразуем цветовое пространство

                face_locations = face_recognition.face_locations(rgb_small_frame)  # Находим расположение лиц
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)  # Кодируем лица

                face_names = []  # Создаем список для имен лиц

                # Проходим по кодировкам найденных лиц
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)  # Сравниваем лица

                    name = 'unknown'  # Изначально имя устанавливаем как "Неизвестно"
                    confidence = 'unknown'  # Изначально уверенность устанавливаем как "Неизвестно"

                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)  # Вычисляем расстояние между лицами
                    best_match_index = np.argmin(face_distances)  # Находим индекс наилучшего соответствия

                    # Если нашли совпадение, устанавливаем имя и уверенность
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = face_confidence(face_distances[best_match_index])

                    face_names.append(f'{name} ({confidence})')  # Добавляем имя и уверенность в список лиц

                # Отображаем рамки и надписи на изображении
                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)  # Рисуем прямоугольник вокруг лица
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), -1)  # Рисуем прямоугольник для текста
                    cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)  # Добавляем надпись

                cv2.imshow('face recognition', frame)  # Отображаем кадр с лицами

                key = cv2.waitKey(1)  # Ждем нажатия клавиши

                # Если нажата клавиша 'q' или русская 'й', завершаем программу
                if key == ord('q') or key == ord('й'):
                    break

        video_capture.release()  # Освобождаем ресурсы камеры
        cv2.destroyAllWindows()  # Закрываем окно просмотра лиц




if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()
