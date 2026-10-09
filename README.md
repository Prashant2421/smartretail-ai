# SmartRetail AI

SmartRetail AI is a Streamlit application for analyzing uploaded retail
footage with YOLO object detection and tracking. It provides a dashboard for
visualizing tracked objects and retail activity from video.

## Run locally

1. Create and activate a Python virtual environment.
2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Start the Streamlit app:

   ```bash
   streamlit run app.py
   ```

The repository includes the `yolo11n.pt` model and a sample
`uploaded_video.mp4` input.
