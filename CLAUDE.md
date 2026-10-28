# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
python -m venv venv
venv\Scripts\activate            # Windows (source venv/bin/activate on Mac/Linux)
pip install -r requirements.txt
streamlit run app.py
```

There are no tests, linter, or build step. The first launch downloads model weights (~300 MB: YOLOv8 face from Hugging Face `arnabdhar/YOLOv8-Face-Detection`, InsightFace `buffalo_l`), so it needs internet. `venv/` is checked into the project directory; ignore it when searching.

## Architecture

Three flat modules, all paths relative to the **current working directory** (run from the project root):

- [face_engine.py](face_engine.py) — `FaceEngine`: two-stage pipeline. YOLOv8 detects face boxes; each box is cropped with 30% padding and re-run through InsightFace (detection+recognition modules only, CPU) to get a 512-d normalized ArcFace embedding. Matching is cosine similarity (dot product of normalized vectors) taking the max over each person's stored embeddings; below the threshold → "Unknown".
- Liveness (passive anti-spoofing) runs per detected box *before* embedding, using `models/liveness.onnx` (auto-downloaded if missing; 128×128 RGB, logits `[real, spoof]`, score = real − spoof). Faces scoring below the sidebar "Liveness threshold" are labelled `Spoof` and never matched or logged.
- [attendance.py](attendance.py) — CSV log at `data/attendance.csv` (`Name,Date,Time`), enforcing one row per person per day.
- [app.py](app.py) — Streamlit UI with three tabs (Mark Attendance, Register Faces, Records). `FaceEngine` is created once via `@st.cache_resource`; detection confidence and match threshold come from sidebar sliders.

### Data layout (created at import of face_engine)
- `data/embeddings.pkl` — pickled `{name: [embedding, ...]}`; this is the live recognition database, loaded at engine init and rewritten on enroll/delete.
- `data/known_faces/<person_name>/*.jpg` — raw images. The UI saves uploads here on registration so the DB can be regenerated; `build_from_folder()` **wipes and rebuilds** `embeddings.pkl` entirely from this folder.

### Gotchas
- `engine.enroll` uses only the largest detected face per image and detects with the default conf (0.5), ignoring the sidebar slider.
- Engine state is cached across Streamlit reruns, so changes to `face_engine.py` require restarting the app (or clearing the cache) to take effect.
