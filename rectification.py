import cv2
import numpy as np
import os

def compute_rectification_maps(K1, D1, K2, D2, R, T, img_size):
    """
    Вычисляет карты преобразования для исправления (rectification) стереоизображений.
    """
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, D1, K2, D2, img_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY
    )
    map1x, map1y = cv2.initUndistortRectifyMap(K1, D1, R1, P1, img_size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K2, D2, R2, P2, img_size, cv2.CV_32FC1)
    
    return map1x, map1y, map2x, map2y, Q

def extract_single_frame(video_path, frame_idx, target_img_size):
    """
    Извлекает кадр с заданным индексом из видео и возвращает его, 
    масштабируя до target_img_size.
    """
    cap = cv2.VideoCapture(video_path)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if frame.shape[0] != target_img_size[1] or frame.shape[1] != target_img_size[0]:
        frame = cv2.resize(frame, target_img_size)
    
    return frame

if __name__ == "__main__":
    # Калибровочные параметры камер
    IMG_WIDTH = 1024
    IMG_HEIGHT = 768
    img_size = (IMG_WIDTH, IMG_HEIGHT)

    bining = 2
    # Размер пикселя в мм, учитывая биннинг
    pixel_size_x_mm = 0.00345 * bining 
    pixel_size_y_mm = 0.00345 * bining

    cx1 = IMG_WIDTH / 2 
    cy1 = IMG_HEIGHT / 2 
    cx2 = IMG_WIDTH / 2 
    cy2 = IMG_HEIGHT / 2 

    focal_length_cam1_mm = 6.0 
    focal_length_cam2_mm = 8.0 

    fx1 = focal_length_cam1_mm / pixel_size_x_mm
    fy1 = focal_length_cam1_mm / pixel_size_y_mm
    fx2 = focal_length_cam2_mm / pixel_size_x_mm
    fy2 = focal_length_cam2_mm / pixel_size_y_mm

    K1 = np.array([[fx1, 0, cx1],
                   [0, fy1, cy1],
                   [0, 0, 1.0]])
    
    D1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0]) # Предполагаем отсутствие дисторсии

    K2 = np.array([[fx2, 0, cx2],
                   [0, fy2, cy2],
                   [0, 0, 1.0]])
    D2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0]) # Предполагаем отсутствие дисторсии

    # Внешние параметры
    # Ваша композиция вращений
    theta_degrees = 33
    theta_radians = np.deg2rad(theta_degrees)

    R_x = np.array([[1, 0, 0], 
                    [0, np.cos(theta_radians), -np.sin(theta_radians)],
                    [0, np.sin(theta_radians), np.cos(theta_radians)]])
    
    R_z = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])

    R = R_z.T @ R_x.T

    # T - смещение Камеры 2 относительно Камеры 1.
    T = np.array([[-3000], [0.0], [0.0]]) # Смещение по X на 3000 мм

    # --- 4. Пути к вашим видеофайлам и параметрам сохранения ---
    video_left_path = 'camera_top_far.mkv_20250528T043324.792208.mkv'  # Левая камера
    video_right_path = 'camera_top_near.mkv_20250528T043324.787084.mkv' # Правая камера
    
    frame_index_to_process = 69 # 70-й кадр

    # Настройки сохранения выровненных кадров
    OUTPUT_FOLDER = 'rectified_frames_output_with_grid' # Изменено название папки
    os.makedirs(OUTPUT_FOLDER, exist_ok=True) 

    # Вычисление карт выравнивания
    map1x, map1y, map2x, map2y, _ = compute_rectification_maps(K1, D1, K2, D2, R, T, img_size)

    # Извлечение 70-го кадра из каждого видео

    frame_l_orig = extract_single_frame(video_left_path, frame_index_to_process, img_size)
    frame_r_orig = extract_single_frame(video_right_path, frame_index_to_process, img_size)

    #  Ректификация
    left_rectified = cv2.remap(frame_l_orig, map1x, map1y, cv2.INTER_LINEAR)
    right_rectified = cv2.remap(frame_r_orig, map2x, map2y, cv2.INTER_LINEAR)
    
    print("Кадры успешно ректифицированы (выровнены).")

    print("\n--- Добавление горизонтальных линий для визуализации ---")
    line_color = (0, 0, 255) # Красный цвет в BGR
    line_thickness = 1
    line_step = 40 # Шаг в пикселях между линиями

    # Рисуем линии на левом ректифицированном изображении
    for i in range(0, left_rectified.shape[0], line_step):
        cv2.line(left_rectified, (0, i), (left_rectified.shape[1], i), line_color, line_thickness)
    
    # Рисуем линии на правом ректифицированном изображении
    for i in range(0, right_rectified.shape[0], line_step):
        cv2.line(right_rectified, (0, i), (right_rectified.shape[1], i), line_color, line_thickness)

    # Сохранение выровненных кадров 
    filename_base = f"frame_{frame_index_to_process+1}"

    cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"{filename_base}_left_rectified_with_grid.png"), left_rectified)

    cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"{filename_base}_right_rectified_with_grid.png"), right_rectified)

    # Визуализация
    # Отображаем оригинальные и ректифицированные кадры
    display_img_orig = np.hstack((frame_l_orig, frame_r_orig))
    display_img_rect_with_grid = np.hstack((left_rectified, right_rectified)) 

    # Масштабируем для отображения, если кадры слишком большие
    display_width = display_img_orig.shape[1] 
    display_img_rect_with_grid = cv2.resize(display_img_rect_with_grid, 
                                            (display_width, int(display_img_rect_with_grid.shape[0] * display_width / display_img_rect_with_grid.shape[1])))

    final_display = np.vstack((display_img_orig, display_img_rect_with_grid))
    
    cv2.imshow(f'Кадр {frame_index_to_process+1}: Оригиналы (Сверху) | Ректифицированные с сеткой (Снизу) (Нажмите любую клавишу)', final_display)
    
    filename_base = f"frame_{frame_index_to_process+1}"

    # Сохраняем оригинальные кадры
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"{filename_base}_left_original.png"), frame_l_orig)
    cv2.imwrite(os.path.join(OUTPUT_FOLDER, f"{filename_base}_right_original.png"), frame_r_orig)
    cv2.waitKey(0)
    cv2.destroyAllWindows()