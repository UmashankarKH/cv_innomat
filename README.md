# Face Attendance Demo

YOLOv8 (face detection) + ArcFace (recognition) + OpenCV + Streamlit.

## Run locally
```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
streamlit run app.py
```
First launch downloads the model weights (~300 MB total), so keep internet on.

## Using saved images
Create `data/known_faces/<person_name>/` with 3-5 photos each, then click
"Rebuild database from folder" in the Register tab.

## Files
- face_engine.py  - detection, embeddings, matching, bounding boxes
- attendance.py   - CSV logging, one entry per person per day
- app.py          - Streamlit UI
