import cv2
import mediapipe as mp
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import os

# === Setup ===
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
DATA_PATH = "C:/Users/madhu/Downloads/archive (4)/leapGestRecog"

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

# === Option 1: Collect Gesture Data ===
def collect_data(gesture_name, samples=100):
    print(f"\n Collecting data for: {gesture_name}")
    cap = cv2.VideoCapture(0)
    count = 0
    data = []

    while count < samples:
        ret, frame = cap.read()
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
                landmarks = []
                for lm in handLms.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                data.append(landmarks)
                count += 1

        cv2.putText(frame, f"{gesture_name}: {count}/{samples}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Collecting Gesture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    with open(f"{DATA_PATH}/{gesture_name}.pkl", "wb") as f:
        pickle.dump((data, [gesture_name]*len(data)), f)
    print(f"Saved {len(data)} samples for '{gesture_name}' to {DATA_PATH}/{gesture_name}.pkl")

# === Option 2: Train Model from .pkl Files ===
def train_model():
    print("\n Training model from collected gesture data...")
    X, y = [], []
    for file in os.listdir(DATA_PATH):
        file_path = os.path.join(DATA_PATH, file)
        if os.path.isfile(file_path) and file_path.endswith(".pkl"):
            with open(file_path, "rb") as f:
                data, labels = pickle.load(f)
                X.extend(data)
                y.extend(labels)

    if not X:
        print("No gesture data found. Please collect some first.")
        return

    model = RandomForestClassifier()
    model.fit(X, y)

    with open("gesture_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print(" Model trained and saved as gesture_model.pkl")

# === Option 3: Run Real-Time Gesture Prediction ===
def run_prediction():
    print("\n Starting real-time gesture recognition...")
    if not os.path.exists("gesture_model.pkl"):
        print(" Model not found. Train it first using option 2.")
        return

    with open("gesture_model.pkl", "rb") as f:
        model = pickle.load(f)

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
                landmarks = []
                for lm in handLms.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                if len(landmarks) == 63:
                    pred = model.predict([landmarks])[0]
                    cv2.putText(frame, f"Gesture: {pred}", (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Gesture Prediction", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# === Main Menu ===
if __name__ == "__main__":
    while True:
        print("\n==== Hand Gesture Recognition Menu ====")
        print("1. Collect new gesture data")
        print("2. Train model")
        print("3. Run real-time prediction")
        print("4. Exit")
        choice = input("Enter your choice (1-4): ")

        if choice == '1':
            gesture = input("Enter gesture name (e.g., peace, stop): ")
            collect_data(gesture)
        elif choice == '2':
            train_model()
        elif choice == '3':
            run_prediction()
        elif choice == '4':
            print("Exiting.")
            break
        else:
            print("Invalid choice. Please select 1–4.")

