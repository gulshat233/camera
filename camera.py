import cv2
import numpy as np
import os


class FaceAndFingerCounter:
    def __init__(self):
        # Каскады для распознавания
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.owner_img_path = "owner.jpg"
        self.owner_hist = None

        if os.path.exists(self.owner_img_path):
            self.load_owner()
            print("✅ Фото владельца загружено!")
        else:
            print("❌ Файл 'owner.jpg' не найден")
            print("Нажмите 's' чтобы сделать фото")

        # Камера
        self.cap = cv2.VideoCapture(0)

    def load_owner(self):
        img = cv2.imread(self.owner_img_path)
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)

            if len(faces) > 0:
                x, y, w, h = faces[0]
                face = img[y:y + h, x:x + w]
            else:
                face = img

            # Создаём гистограмму
            self.owner_hist = []
            for i in range(3):
                hist = cv2.calcHist([face], [i], None, [256], [0, 256])
                cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
                self.owner_hist.append(hist)

    def detect_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
        )

        if len(faces) > 0:
            return faces[0]  # x, y, w, h
        return None

    def is_owner(self, frame, face_rect):
        if self.owner_hist is None:
            return False

        x, y, w, h = face_rect
        face = frame[y:y + h, x:x + w]

        # Сравниваем гистограммы
        score = 0
        for i in range(3):
            hist = cv2.calcHist([face], [i], None, [256], [0, 256])
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
            score += cv2.compareHist(self.owner_hist[i], hist, cv2.HISTCMP_CORREL)

        similarity = score / 3
        return similarity > 0.5

    def detect_hands(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        hands = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5000:  # Минимальная площадь руки
                hull = cv2.convexHull(contour)
                hull_indices = cv2.convexHull(contour, returnPoints=False)

                if len(hull_indices) > 3:
                    defects = cv2.convexityDefects(contour, hull_indices)
                else:
                    defects = None

                hands.append((contour, hull, defects))

        return hands

    def count_fingers(self, contour, hull, defects):
        if defects is None:
            return 0

        finger_count = 0

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            # Угол между пальцами
            a = np.linalg.norm(np.array(start) - np.array(far))
            b = np.linalg.norm(np.array(end) - np.array(far))
            c = np.linalg.norm(np.array(start) - np.array(end))

            if a * b > 0:  # избегаем деления на ноль
                angle = np.arccos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))
                angle = np.degrees(angle)

                # Если угол острый - это палец
                if angle <= 90 and d > 5000:
                    finger_count += 1

        return min(finger_count + 1, 5)  # +1 для большого пальца

    def run(self):
        print("\n🚀 Программа запущена!")
        print("Нажмите 'q' для выхода")
        print("Нажмите 's' чтобы сделать фото владельца")
        print("Держите руку на расстоянии от лица")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Ошибка камеры")
                break

            frame = cv2.flip(frame, 1)
            frame_copy = frame.copy()

            # Ищем лицо
            face_rect = self.detect_face(frame)
            owner_detected = False

            if face_rect is not None:
                x, y, w, h = face_rect

                # Проверяем владельца
                if self.owner_hist is not None:
                    owner_detected = self.is_owner(frame, face_rect)

                color = (0, 255, 0) if owner_detected else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                label = "OWNER" if owner_detected else "UNKNOWN"
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Статус
            if self.owner_hist is None:
                cv2.putText(frame, "No owner photo! Press 's'", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            elif owner_detected:
                cv2.putText(frame, "OWNER DETECTED ✅", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Searching for owner...", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            # Считаем пальцы только для владельца
            if owner_detected:
                # Убираем область лица для поиска рук
                if face_rect is not None:
                    fx, fy, fw, fh = face_rect
                    cv2.rectangle(frame_copy, (fx - 20, fy - 20),
                                  (fx + fw + 20, fy + fh + 20), (0, 0, 0), -1)

                hands = self.detect_hands(frame_copy)
                total_fingers = 0

                for contour, hull, defects in hands:
                    cv2.drawContours(frame, [contour], -1, (255, 0, 0), 2)
                    cv2.drawContours(frame, [hull], -1, (0, 255, 0), 2)

                    count = self.count_fingers(contour, hull, defects)
                    total_fingers += count

                    # Подпись
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        cv2.putText(frame, str(count), (cx, cy),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 0), 2)

                if total_fingers > 0:
                    cv2.putText(frame, f"Total fingers: {total_fingers}", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)

            cv2.imshow('Face & Finger Counter', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(self.owner_img_path, frame)
                self.load_owner()
                print("✅ Фото владельца сохранено!")

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = FaceAndFingerCounter()
    app.run()