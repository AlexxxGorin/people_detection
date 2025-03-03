import os
import cv2
from ultralytics import YOLO
from config import classes, input_dir, output_dir


def get_cap_info(cap):
    """
    Возвращает базовую информацию о видео: FPS, ширину и высоту кадра.

    Параметры
    ----------
    cap : cv2.VideoCapture
        Объект видео, из которого извлекается информация.

    Возвращает
    ----------
    tuple
        Кортеж (fps, width, height), где:
        - fps (float): Количество кадров в секунду.
        - width (int): Ширина кадра в пикселях.
        - height (int): Высота кадра в пикселях.
        - frames_total (int): Количество кадров в видео
    """
    return (
        cap.get(cv2.CAP_PROP_FPS),
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    )


def process_video(video_path, filename, model):
    """
    Обрабатывает указанный видеофайл с помощью модели YOLO, прорисовывая
    прямоугольники и подписи для обнаруженных объектов на каждом кадре,
    после чего сохраняет готовое видео в формате MP4.

    Параметры
    ----------
    video_path : str
        Путь к входному видеофайлу.
    filename : str
        Имя видеофайла (используется для формирования названия выходного файла).
    model : YOLO
        Модель детекции, поддерживающая метод .predict() для получения
        результатов на каждом кадре.

    Используемые глобальные переменные
    ----------
    output_dir : str
        Путь к папке, в которую будет сохранён результат.
    classes : dict
        Словарь, где по индексу (ID класса) можно получить название класса
        для подписи.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Не удалось открыть видео {video_path}")
        return

    fps, width, height, frames_total = get_cap_info(cap)

    # Формируем путь для выходного файла
    base_name = os.path.splitext(filename)[0]
    output_video_path = os.path.join(output_dir, base_name + "_processed.mp4")

    # Настраиваем VideoWriter для записи в MP4
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Читаем кадры из входного видео, обрабатываем моделью и сохраняем
    frame_counter = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_counter += 1
        results = model.predict(frame, verbose=False)

        # results[0].boxes содержит координаты, классы и конфиденции
        boxes = results[0].boxes.xyxy
        confs = results[0].boxes.conf
        class_ids = results[0].boxes.cls

        # Отрисовываем детекции на кадре
        if boxes is not None:
            for box, conf, cls_id in zip(boxes, confs, class_ids):
                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{classes[int(cls_id)]} {float(conf):.2f}"

                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    2
                )

        # Записываем кадр в выходное видео
        out.write(frame)
        print(f"Frames processed {frame_counter}/{frames_total}")

    cap.release()
    out.release()
    print(f"Сохранён обработанный файл: {output_video_path}")


def main():
    """
    Запускает процесс обработки видеофайлов в папке input_dir,
    используя модель YOLO, и сохраняет результаты в output_dir.
    """
    # Загрузка модели (весов)
    model = YOLO("src/model/best.pt")

    # Создаём папку для результатов, если она не существует
    os.makedirs(output_dir, exist_ok=True)

    # Перебираем все видеофайлы в папке входных данных
    for filename in os.listdir(input_dir):
        # Проверяем, что файл - это видео (можно расширить список форматов)
        if filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            video_path = os.path.join(input_dir, filename)
            process_video(video_path, filename, model)

    print("Обработка завершена.")


if __name__ == "__main__":
    main()
