# PostureAI Web

This project uses:
- MediaPipe Pose Landmarker in the browser
- FastAPI
- The existing `posture_model.pkl`
- The exact 13 feature columns used by the uploaded `train.py`

## 1. Put your model here

Copy your existing file:

    backend/posture_model.pkl

Do not retrain unless you want to change the model.

## 2. Install backend dependencies

From the `backend` folder:

    pip install fastapi uvicorn pandas joblib

## 3. Start the backend

From `backend`:

    uvicorn server:app --host 0.0.0.0 --port 8000 --ssl-keyfile ..\certs\key.pem --ssl-certfile ..\certs\cert.pem

## 4. Install frontend

From `frontend`:

    npm install

## 5. Start frontend

From `frontend`:

    npm run dev

Vite will print an address such as:

    https://localhost:5173/

For another device on the same LAN, use:

    https://YOUR-PC-LAN-IP:5173/

Example:

    https://192.168.1.100:5173/

The frontend automatically sends prediction requests to:

    https://YOUR-PC-LAN-IP:8000/

## Camera note

Browser camera access can be restricted on insecure origins. `localhost` is normally allowed, but another LAN device may require a secure context depending on the browser. If the camera does not work over LAN, use HTTPS for the frontend or test on the PC first.

## Important

The model was trained with these exact features:

    nose_x
    nose_y
    left_shoulder_x
    left_shoulder_y
    right_shoulder_x
    right_shoulder_y
    left_elbow_angle
    right_elbow_angle
    left_knee_angle
    right_knee_angle
    shoulder_angle
    torso_angle
    normalized_head_distance

The new frontend intentionally uses the same feature calculations as the uploaded `main.js`.
