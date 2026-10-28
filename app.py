import cv2
import numpy as np
import streamlit as st

import attendance
from face_engine import FaceEngine, KNOWN_DIR

st.set_page_config(page_title="Face Attendance", page_icon="🧑‍💼", layout="wide")


@st.cache_resource(show_spinner="Loading YOLO + ArcFace models (first run downloads weights)...")
def get_engine():
    return FaceEngine()


engine = get_engine()


def decode(uploaded):
    arr = np.frombuffer(uploaded.getvalue(), np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def show(img_bgr, caption=None):
    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), caption=caption, use_container_width=True)


st.title("🧑‍💼 Face Recognition Attendance")
st.caption("YOLOv8 face detection + ArcFace recognition · OpenCV · Streamlit")

with st.sidebar:
    st.header("Settings")
    det_conf = st.slider("Detection confidence (YOLO)", 0.1, 0.9, 0.5, 0.05)
    threshold = st.slider("Match threshold (cosine similarity)", 0.2, 0.8, 0.4, 0.05,
                          help="Higher = stricter. Raise it if wrong people get matched.")
    live_thr = st.slider("Liveness threshold", -3.0, 5.0, 0.0, 0.25,
                         help="Higher = stricter. Raise it if photos/screens still pass; "
                              "lower it if real faces are rejected.")
    st.divider()
    st.subheader("Registered people")
    people = engine.people()
    if people:
        for n, c in people.items():
            st.write(f"• **{n}** ({c} images)")
    else:
        st.info("No one registered yet.")

tab1, tab2, tab3 = st.tabs(["✅ Mark Attendance", "➕ Register Faces", "📋 Records"])

# ---------------- Mark attendance ----------------
with tab1:
    mode = st.radio("Input", ["Webcam", "Upload photo"], horizontal=True)
    img = None
    if mode == "Webcam":
        shot = st.camera_input("Look at the camera and take a photo")
        if shot:
            img = decode(shot)
    else:
        up = st.file_uploader("Upload a photo", type=["jpg", "jpeg", "png"], key="att_up")
        if up:
            img = decode(up)

    if img is not None:
        if not engine.db:
            st.warning("Register at least one person first.")
        else:
            results = engine.identify(img, det_conf, threshold, live_thr)
            left, right = st.columns([3, 2])
            with left:
                show(engine.annotate(img, results), "Detection result")
            with right:
                if not results:
                    st.error("No face detected.")
                for r in results:
                    if r["name"] == "Spoof":
                        st.error(f"🚫 Spoof attempt blocked – photo/screen detected "
                                 f"(liveness {r['live']:.2f})")
                    elif r["name"] == "Unknown":
                        st.error("Unknown face – not in database")
                    else:
                        new, t = attendance.mark(r["name"])
                        if new:
                            st.success(f"✅ {r['name']} marked present at {t} "
                                       f"(match {r['score']:.2f})")
                        else:
                            st.info(f"ℹ️ {r['name']} already marked today at {t}")

# ---------------- Register ----------------
with tab2:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Register a new person")
        name = st.text_input("Full name")
        src = st.radio("Source", ["Upload images", "Webcam"], horizontal=True)
        imgs = []
        if src == "Upload images":
            files = st.file_uploader("Upload 3–5 clear face photos", type=["jpg", "jpeg", "png"],
                                     accept_multiple_files=True)
            imgs = [decode(f) for f in files] if files else []
        else:
            shot = st.camera_input("Capture", key="reg_cam")
            imgs = [decode(shot)] if shot else []

        if st.button("Save to database", type="primary"):
            if not name.strip():
                st.error("Enter a name.")
            elif not imgs:
                st.error("Provide at least one image.")
            else:
                n = engine.enroll(name.strip(), imgs)
                # keep the raw images too, so the database can be rebuilt later
                folder = KNOWN_DIR / name.strip()
                folder.mkdir(parents=True, exist_ok=True)
                for i, im in enumerate(imgs):
                    cv2.imwrite(str(folder / f"{len(list(folder.iterdir())) + i}.jpg"), im)
                if n:
                    st.success(f"Added {n} face(s) for {name}.")
                    st.rerun()
                else:
                    st.error("No face found in the image(s). Try a clearer, front-facing photo.")
    with c2:
        st.subheader("Index saved images folder")
        st.write(f"Put photos in `{KNOWN_DIR}/<person_name>/` then rebuild:")
        if st.button("Rebuild database from folder"):
            with st.spinner("Indexing..."):
                rep = engine.build_from_folder()
            st.json(rep)
            st.rerun()
        if people:
            who = st.selectbox("Remove a person", list(people))
            if st.button("Delete"):
                engine.delete_person(who)
                st.rerun()

# ---------------- Records ----------------
with tab3:
    df = attendance.load()
    if df.empty:
        st.info("No attendance recorded yet.")
    else:
        day = st.selectbox("Date", ["All"] + sorted(df["Date"].unique(), reverse=True))
        view = df if day == "All" else df[df["Date"] == day]
        st.dataframe(view, use_container_width=True, hide_index=True)
        st.download_button("Download CSV", view.to_csv(index=False), "attendance.csv", "text/csv")
