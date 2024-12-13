import cv2
from deepface import DeepFace
from threading import Thread

# متغیر برای ذخیره احساس شناسایی شده
emotion = "Analyzing..."

# تابع جداگانه برای تحلیل احساسات
def analyze_emotion(frame):
    global emotion
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        if result and 'dominant_emotion' in result[0]:
            emotion = result[0]['dominant_emotion']
        else:
            emotion = "No Emotion Detected"
    except Exception as e:
        emotion = "Error"

# راه‌اندازی دوربین
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting...")
        break

    # معکوس کردن فریم برای نمایش صحیح
    frame = cv2.flip(frame, 1)

    # شروع تحلیل در یک Thread جداگانه
    if not Thread(target=analyze_emotion, args=(frame,)).is_alive():
        Thread(target=analyze_emotion, args=(frame,)).start()

    # نمایش احساسات روی تصویر
    cv2.putText(frame, f"Emotion: {emotion}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # نمایش تصویر
    cv2.imshow('Emotion Detection', frame)

    # خروج با فشردن کلید q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# آزاد کردن منابع
cap.release()
cv2.destroyAllWindows()
