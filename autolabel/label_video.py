"""Extract frames from a local video and register them in the project."""
from pathlib import Path
from uuid import uuid4

from label_project import Project


def extract_frames(project_path, video_path, fps, cancelled, progress):
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("Install OpenCV from requirements-ml.txt to import video.") from error
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise ValueError("Could not open this video.")
        native_fps = capture.get(cv2.CAP_PROP_FPS)
        if native_fps <= 0:
            raise ValueError("Could not determine the video's frame rate.")
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        rate = min(fps, native_fps)
        destination = Path(project_path).parent / "images" / (Path(video_path).stem + "-" + uuid4().hex[:8])
        destination.mkdir(parents=True)
        paths = []
        index = 0
        next_time = 0.0
        while not cancelled.is_set():
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = index / native_fps
            if timestamp + 0.5 / native_fps >= next_time:
                path = destination / f"frame_{index:08d}.jpg"
                ok, encoded = cv2.imencode(".jpg", frame)
                if not ok:
                    raise ValueError(f"Could not encode frame {index}.")
                path.write_bytes(encoded.tobytes())
                paths.append(path)
                next_time += 1 / rate
            index += 1
            if index % 25 == 0:
                progress(index, max(index, total))
    finally:
        capture.release()
    project = Project(project_path)
    try:
        return project.import_images(paths)
    finally:
        project.close()

