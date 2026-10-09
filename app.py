
import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from collections import deque
from ultralytics import YOLO

st.set_page_config(
    page_title="SmartRetail AI",
    page_icon="🏬",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #0b1117;
    --panel: #111c24;
    --green: #56f5a6;
    --muted: #94a7b5;
}

.stApp {
    background: var(--bg);
    font-family: 'Inter', sans-serif;
}

[data-testid="stHeader"] {
    background: rgba(11, 17, 23, 0.9);
}

[data-testid="stSidebar"] {
    background: #0e171e;
    border-right: 1px solid #22313b;
}

[data-testid="stSidebar"] * {
    color: #e8f1f5;
}

h1, h2, h3, p, label {
    color: #e8f1f5;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    padding: 26px 30px;
    border: 1px solid #263b43;
    border-radius: 20px;
    background: linear-gradient(120deg, #142a2a, #111c24 65%, #15212a);
    margin-bottom: 24px;
}

.hero-tag {
    display: inline-block;
    padding: 6px 10px;
    border: 1px solid #2b654d;
    border-radius: 30px;
    color: #56f5a6;
    background: #173326;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1px;
}

.hero h1 {
    font-size: 36px;
    font-weight: 800;
    margin: 15px 0 8px 0;
    color: #ffffff;
}

.hero p {
    color: #a6b8c4;
    font-size: 15px;
    margin-bottom: 0;
}

.section-title {
    font-size: 18px;
    font-weight: 700;
    margin: 12px 0 14px 0;
    color: #e8f1f5;
}

[data-testid="stMetric"] {
    background: #111c24;
    border: 1px solid #263b43;
    padding: 18px;
    border-radius: 15px;
}

[data-testid="stMetricLabel"] {
    color: #a6b8c4;
}

[data-testid="stMetricValue"] {
    color: #56f5a6;
}

.stButton > button {
    background: #56f5a6;
    color: #07130d;
    border: none;
    border-radius: 10px;
    min-height: 44px;
    font-weight: 800;
    transition: 0.2s ease;
}

.stButton > button:hover {
    background: #8bffc5;
    color: #07130d;
    border: none;
    box-shadow: 0 0 20px #56f5a633;
}

[data-testid="stFileUploader"] {
    background: #111c24;
    border: 1px dashed #3a5963;
    border-radius: 15px;
    padding: 12px;
}

[data-testid="stFileUploader"] section {
    background: transparent;
}

hr {
    border-color: #263b43;
}

.stAlert {
    border-radius: 12px;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


st.markdown("""
<div class="hero">
    <span class="hero-tag">● AI SYSTEM READY</span>
    <h1>SmartRetail AI</h1>
    <p>Intelligent crowd monitoring · Queue analytics · Computer vision</p>
</div>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="section-title">Video Intelligence Center</div>',
    unsafe_allow_html=True
)

# ---------------- SETTINGS ----------------

MODEL_PATH = "yolo11n.pt"

# Number of recent positions used to estimate whether a person is waiting
HISTORY_LENGTH = 30

# A person is considered relatively stationary if their recent
# positions stay within this pixel distance
STATIONARY_DISTANCE = 35

# Maximum distance between stationary people to group them
GROUP_DISTANCE = 140

# Padding around the automatically estimated queue zone
ZONE_PADDING = 50


with st.sidebar:
    st.markdown("## 🏬 SmartRetail AI")
    st.caption("RETAIL INTELLIGENCE PLATFORM")

    st.divider()

    st.markdown("### ⚙️ Monitoring settings")

    queue_limit = st.slider(
        "Queue alert threshold",
        min_value=2,
        max_value=20,
        value=4,
        help="Show an alert when the estimated queue reaches this number."
    )

    st.markdown("### 🧠 AI configuration")
    st.success("YOLO detection enabled")
    st.success("ByteTrack tracking enabled")

    st.divider()

    st.caption("SMARTRETAIL AI · PROTOTYPE")

uploaded_video = st.file_uploader(
    "Upload a CCTV video",
    type=["mp4", "avi", "mov", "mkv"]
)

# ---------------- HELPER FUNCTIONS ----------------

def find_groups(people):
    """Group nearby stationary people using their center points."""

    groups = []
    remaining = set(people.keys())

    while remaining:
        first = remaining.pop()
        group = {first}
        stack = [first]

        while stack:
            current_id = stack.pop()
            cx, cy = people[current_id]["center"]

            nearby = []

            for other_id in remaining:
                ox, oy = people[other_id]["center"]
                distance = np.hypot(cx - ox, cy - oy)

                if distance <= GROUP_DISTANCE:
                    nearby.append(other_id)

            for other_id in nearby:
                remaining.remove(other_id)
                group.add(other_id)
                stack.append(other_id)

        groups.append(group)

    return groups


def estimate_queue_zone(group, people, frame_width, frame_height):
    """Estimate a rectangle around the detected waiting group."""

    boxes = [people[person_id]["box"] for person_id in group]

    x1 = min(box[0] for box in boxes) - ZONE_PADDING
    y1 = min(box[1] for box in boxes) - ZONE_PADDING
    x2 = max(box[2] for box in boxes) + ZONE_PADDING
    y2 = max(box[3] for box in boxes) + ZONE_PADDING

    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(frame_width - 1, int(x2))
    y2 = min(frame_height - 1, int(y2))

    return x1, y1, x2, y2


# ---------------- VIDEO ANALYSIS ----------------

if uploaded_video is not None:

    st.video(uploaded_video)

    if st.button("Analyze Automatically", type="primary"):

        temp_path = None

        try:
            # Save uploaded video temporarily
            suffix = os.path.splitext(uploaded_video.name)[1] or ".mp4"

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as temp_file:
                temp_file.write(uploaded_video.getvalue())
                temp_path = temp_file.name

            # Load YOLO model
            model = YOLO(MODEL_PATH)

            cap = cv2.VideoCapture(temp_path)

            if not cap.isOpened():
                st.error("Could not open this video.")
                st.stop()

            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0:
                fps = 25

            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Track history for each person
            histories = {}
            latest_people = {}
            frame_number = 0

            preview = st.empty()
            status = st.empty()
            metrics = st.empty()

            st.info(
                "Analyzing people and estimating a queue zone. "
                "The first few seconds may not be enough to identify waiting."
            )

            while True:
                success, frame = cap.read()

                if not success:
                    break

                frame_number += 1

                # Track only people (COCO class 0) with ByteTrack
                results = model.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=[0],
                    conf=0.35,
                    verbose=False
                )

                current_people = {}

                result = results[0]
                boxes = result.boxes

                if boxes is not None and boxes.id is not None:
                    box_coords = boxes.xyxy.cpu().numpy()
                    track_ids = boxes.id.int().cpu().tolist()

                    for coords, track_id in zip(box_coords, track_ids):
                        x1, y1, x2, y2 = map(int, coords)

                        # Use bottom-center as the person's ground position
                        center = (
                            int((x1 + x2) / 2),
                            int(y2)
                        )

                        if track_id not in histories:
                            histories[track_id] = deque(
                                maxlen=HISTORY_LENGTH
                            )

                        histories[track_id].append(center)

                        positions = list(histories[track_id])

                        # Estimate whether the person is waiting
                        stationary = False

                        if len(positions) >= 15:
                            points = np.array(positions)
                            median_point = np.median(points, axis=0)

                            distances = np.linalg.norm(
                                points - median_point,
                                axis=1
                            )

                            stationary = (
                                np.max(distances) <= STATIONARY_DISTANCE
                            )

                        current_people[track_id] = {
                            "center": center,
                            "box": (x1, y1, x2, y2),
                            "stationary": stationary
                        }

                latest_people = current_people

                # Keep people who appear to be waiting
                waiting_people = {
                    person_id: data
                    for person_id, data in current_people.items()
                    if data["stationary"]
                }

                # Group nearby waiting people
                groups = find_groups(waiting_people)

                # Choose the largest group with at least two people
                valid_groups = [
                    group for group in groups if len(group) >= 2
                ]

                queue_group = (
                    max(valid_groups, key=len)
                    if valid_groups else set()
                )

                display_frame = frame.copy()

                # Draw tracked people
                for person_id, data in current_people.items():
                    x1, y1, x2, y2 = data["box"]

                    if person_id in queue_group:
                        color = (0, 200, 0)  # Green: estimated queue
                        label = f"Queue ID {person_id}"
                    elif data["stationary"]:
                        color = (0, 200, 255)  # Yellow: stationary
                        label = f"Waiting? {person_id}"
                    else:
                        color = (255, 150, 0)  # Blue: moving
                        label = f"Person {person_id}"

                    cv2.rectangle(
                        display_frame,
                        (x1, y1),
                        (x2, y2),
                        color,
                        2
                    )

                    cv2.putText(
                        display_frame,
                        label,
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2
                    )

                queue_count = len(queue_group)

                # Estimate and draw the queue zone automatically
                if queue_group:
                    zx1, zy1, zx2, zy2 = estimate_queue_zone(
                        queue_group,
                        waiting_people,
                        frame_width,
                        frame_height
                    )

                    cv2.rectangle(
                        display_frame,
                        (zx1, zy1),
                        (zx2, zy2),
                        (0, 255, 0),
                        3
                    )

                    cv2.putText(
                        display_frame,
                        f"Estimated Queue: {queue_count}",
                        (zx1, max(25, zy1 - 12)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )

                else:
                    cv2.putText(
                        display_frame,
                        "Queue zone not identified yet",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 165, 255),
                        2
                    )

                # Alert when the queue exceeds the configured limit
                if queue_count >= queue_limit:
                    cv2.putText(
                        display_frame,
                        "QUEUE ALERT!",
                        (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3
                    )

                # Display the analyzed frame
                preview.image(
                    cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    use_container_width=True
                )

                status.write(f"Frames analyzed: {frame_number}")

                
            with metrics.container():
                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "👥 Queue Size",
                    queue_count
                )

                col2.metric(
                    "🚶 People Detected",
                    len(current_people)
                )

                col3.metric(
                    "🎯 Queue Threshold",
                    queue_limit
                )

            cap.release()

            st.success("Video analysis completed.")

            st.write(f"Total frames analyzed: {frame_number}")
            st.write(
                "Note: the queue size is an estimate of the largest "
                "group of relatively stationary people, not a confirmed "
                "count of customers waiting at a checkout."
            )

        except Exception as error:
            st.error(f"Analysis failed: {error}")

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)