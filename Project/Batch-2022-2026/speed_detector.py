# speed_detector.py
import cv2
import dlib
import time
import math
import os

# Adjust these defaults if needed
WIDTH = 1280
HEIGHT = 720

def estimateSpeed(location1, location2, fps):
    # pixel distance between centers (or top-left as code originally used)
    d_pixels = math.sqrt((location2[0] - location1[0])**2 + (location2[1] - location1[1])**2)
    ppm = 8.8  # pixels per meter (tune for your camera)
    d_meters = d_pixels / ppm
    speed = d_meters * fps * 3.6  # convert m/s to km/h-ish estimate
    return speed

def process_video(input_path, output_path, haar_cascade_path='myhaar.xml', resize_to=(WIDTH, HEIGHT)):
    """
    Process input video and write annotated output video to output_path.
    Returns: dict with summary info (frames_processed, processed_video_path)
    """
    # Ensure cascade exists
    if not os.path.exists(haar_cascade_path):
        raise FileNotFoundError(f"Haar cascade file not found at {haar_cascade_path}")

    carCascade = cv2.CascadeClassifier(haar_cascade_path)
    video = cv2.VideoCapture(input_path)
    if not video.isOpened():
        raise IOError("Could not open input video: " + input_path)

    # Attempt to get fps from video, fallback to 18
    in_fps = video.get(cv2.CAP_PROP_FPS)
    fps = in_fps if in_fps and in_fps > 0 else 18.0

    width, height = resize_to

    # Setup tracker state
    frameCounter = 0
    currentCarID = 0
    carTracker = {}
    carLocation1 = {}
    carLocation2 = {}
    speed = [None] * 10000  # large enough for many objects

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frames_processed = 0
    rectangleColor = (0, 255, 0)

    try:
        while True:
            start_time = time.time()
            rc, image = video.read()
            if not rc or image is None:
                break

            image = cv2.resize(image, (width, height))
            resultImage = image.copy()
            frameCounter += 1

            # Remove low-quality trackers
            carIDtoDelete = []
            for carID in list(carTracker.keys()):
                trackingQuality = carTracker[carID].update(image)
                if trackingQuality < 7:
                    carIDtoDelete.append(carID)
            for carID in carIDtoDelete:
                carTracker.pop(carID, None)
                carLocation1.pop(carID, None)
                carLocation2.pop(carID, None)

            # Every N frames, detect new cars
            if not (frameCounter % 10):
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                cars = carCascade.detectMultiScale(gray, 1.1, 13, 18, (24, 24))
                for (_x, _y, _w, _h) in cars:
                    x, y, w, h = int(_x), int(_y), int(_w), int(_h)
                    x_bar = x + 0.5 * w
                    y_bar = y + 0.5 * h
                    matchCarID = None
                    for carID in carTracker.keys():
                        trackedPosition = carTracker[carID].get_position()
                        t_x = int(trackedPosition.left())
                        t_y = int(trackedPosition.top())
                        t_w = int(trackedPosition.width())
                        t_h = int(trackedPosition.height())
                        t_x_bar = t_x + 0.5 * t_w
                        t_y_bar = t_y + 0.5 * t_h
                        if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h))
                            and (x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))):
                            matchCarID = carID
                    if matchCarID is None:
                        tracker = dlib.correlation_tracker()
                        tracker.start_track(image, dlib.rectangle(x, y, x + w, y + h))
                        carTracker[currentCarID] = tracker
                        carLocation1[currentCarID] = [x, y, w, h]
                        currentCarID += 1

            # Update positions and draw rectangles
            for carID in carTracker.keys():
                trackedPosition = carTracker[carID].get_position()
                t_x = int(trackedPosition.left())
                t_y = int(trackedPosition.top())
                t_w = int(trackedPosition.width())
                t_h = int(trackedPosition.height())
                cv2.rectangle(resultImage, (t_x, t_y), (t_x + t_w, t_y + t_h), rectangleColor, 4)
                carLocation2[carID] = [t_x, t_y, t_w, t_h]

            end_time = time.time()
            proc_fps = 1.0 / (end_time - start_time) if (end_time != start_time) else fps

            # Calculate speed
            for i in list(carLocation1.keys()):
                if frameCounter % 1 == 0 and i in carLocation2:
                    x1, y1, w1, h1 = carLocation1[i]
                    x2, y2, w2, h2 = carLocation2[i]
                    carLocation1[i] = [x2, y2, w2, h2]
                    if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
                        if (speed[i] is None or speed[i] == 0) and 275 <= y1 <= 285:
                            speed[i] = estimateSpeed([x1, y1, w1, h1], [x2, y2, w2, h2], fps)
                        if speed[i] is not None and y1 >= 180:
                            text = f"{int(speed[i])} km/hr"
                            cv2.putText(resultImage, text, (int(x1 + w1/2), int(y1-5)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)

            # write frame to out video
            out.write(resultImage)
            frames_processed += 1

    finally:
        video.release()
        out.release()

    return {
        "frames_processed": frames_processed,
        "output_path": output_path,
        "fps_used": fps
    }
