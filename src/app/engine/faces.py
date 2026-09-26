"""Finding the faces in footage, and saying where they are.

Faces, not people. A detector that looks for faces does not see somebody
turned away from the camera, and the difference matters enough that it is
worth saying plainly wherever this is used: `faces: 0` means no face was
found, not that the shot is empty.

What comes out is one record per second, which is the granularity reframing
needs. A vertical crop that follows a speaker has to know where they were at
each moment, and a single position averaged over a whole shot is the one
answer that is wrong for every part of it.

The frames themselves are not decoded here. The analysis pass already had the
file open, so it writes small stills as it goes and this reads them back —
adding a face measurement must not add a second pass over the footage.
"""

import os
import re
import shutil
import statistics
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from app.engine import models
from app.engine.ffmpeg import OperationCancelled
from app.models.media import FaceMeasurement

# Stills are written this often, and named by their position in that sequence.
# Provisional: twice a second is also how often the reframing can react.
FRAME_FPS = 2
FRAME_WIDTH = 480
FRAME_PATTERN = "face_%06d.jpg"
FRAME_NAME = re.compile(r"face_(\d+)\.jpg$")
# Below this the detector is guessing. Raising it loses small faces in the background,
# which is the right trade for "is there somebody in this shot and where are they".
# Provisional.
SCORE_THRESHOLD = 0.6
NMS_THRESHOLD = 0.3
TOP_K = 50

# A face's box as a share of the frame: (share of area, centre x, centre y).
Face = Tuple[float, float, float]

def frames_dir(work_dir: str) -> str:
    """Directory the analysis pass writes its stills to.

    Args:
        work_dir: The job's working directory.

    Returns:
        The path. It is not created here; the pass that writes into it does that.
    """
    return os.path.join(work_dir, "faces")

def discard_frames(work_dir: str) -> None:
    """Throw away the stills once they have been looked at.

    They are scaffolding: two a second for the length of the file, which is a
    couple of hundred megabytes an hour, and nothing reads them after the
    detector has. Nothing else clears a job's working directory, so leaving
    them would mean a workspace that grows with every analysis and never
    shrinks.

    Args:
        work_dir: The job's working directory.
    """
    shutil.rmtree(frames_dir(work_dir), ignore_errors=True)

def _detector(width: int, height: int):
    """Load the face detector for a frame size.

    Args:
        width: Frame width in pixels.
        height: Frame height in pixels.

    Returns:
        An OpenCV `FaceDetectorYN` set up for that size.

    Raises:
        RuntimeError: If the model cannot be downloaded.
    """
    import cv2

    return cv2.FaceDetectorYN.create(
        models.ensure(models.FACE_DETECTION), "", (width, height),
        score_threshold=SCORE_THRESHOLD, nms_threshold=NMS_THRESHOLD, top_k=TOP_K,
    )

def _faces_in(detector, image) -> List[Face]:
    """Find the faces in one frame, in shares of the frame rather than pixels.

    Measuring in shares is what makes two files comparable when they were shot
    at different sizes, and it is also what a crop wants: a reframe is
    expressed as a fraction of the picture, not as a pixel column.

    Args:
        detector: The detector, already set to this frame's size.
        image: The frame, as OpenCV loaded it.

    Returns:
        Every face found, largest first.
    """
    height, width = image.shape[:2]
    _, found = detector.detect(image)
    if found is None:
        return []
    # The detector hands back 32-bit floats. Kept as they are, they reach the stored
    # analysis as a number with a long tail of noise on the end of it, so each one
    # becomes an ordinary float here rather than somewhere further downstream.
    faces = [
        (
            min(1.0, float(box[2]) * float(box[3]) / (width * height)),
            min(1.0, max(0.0, (float(box[0]) + float(box[2]) / 2) / width)),
            min(1.0, max(0.0, (float(box[1]) + float(box[3]) / 2) / height)),
        )
        for box in found
    ]
    return sorted(faces, reverse=True)

def _per_second(frames: Sequence[Tuple[float, List[Face]]], duration: float) -> List[FaceMeasurement]:
    """Fold the sampled frames into one record per second.

    Args:
        frames: `(seconds, faces)` in time order.
        duration: Length of the file, which the last record is cut to.

    Returns:
        One record per second that had a frame sampled in it. `faces` is the
        most seen at once rather than an average: a face the detector missed
        in one frame of two did not leave the room. The position is the
        largest face's, taken from the frame where it was largest, because
        averaging two positions puts the crop between two people and on
        neither.
    """
    grouped: Dict[int, List[Tuple[float, List[Face]]]] = {}
    for seconds, faces in frames:
        grouped.setdefault(int(seconds), []).append((seconds, faces))

    measurements: List[FaceMeasurement] = []
    for second in sorted(grouped):
        sampled = [faces for _, faces in grouped[second]]
        biggest = max((faces[0] for faces in sampled if faces), default=None)
        measurements.append(FaceMeasurement(
            start=float(second),
            end=round(min(second + 1.0, duration) if duration else second + 1.0, 3),
            faces=max(len(faces) for faces in sampled),
            face_share=None if biggest is None else round(biggest[0], 4),
            face_x=None if biggest is None else round(biggest[1], 4),
            face_y=None if biggest is None else round(biggest[2], 4),
        ))
    return measurements

def detect_faces(
    work_dir: str,
    duration: float,
    on_progress: Callable[[float], None],
    is_cancelled: Callable[[], bool],
    fps: int = FRAME_FPS,
) -> List[FaceMeasurement]:
    """Find the faces in the stills the analysis pass left behind.

    Args:
        work_dir: The job's working directory, holding the stills.
        duration: Length of the file in seconds.
        on_progress: Called with the completed fraction.
        is_cancelled: Polled between frames to stop early.
        fps: Rate the stills were written at, which is what turns a frame's
            number into its position in the file.

    Returns:
        One record per second. Empty for a file with no picture, which wrote
        no stills.

    Raises:
        OperationCancelled: If cancellation was requested.
        RuntimeError: If the face detection model cannot be downloaded.
    """
    folder = frames_dir(work_dir)
    if not os.path.isdir(folder):
        return []
    names = sorted(
        (name for name in os.listdir(folder) if FRAME_NAME.search(name)),
        key=lambda name: int(FRAME_NAME.search(name).group(1)),
    )
    if not names:
        return []

    import cv2

    detector: Optional[object] = None
    size: Optional[Tuple[int, int]] = None
    frames: List[Tuple[float, List[Face]]] = []
    for index, name in enumerate(names):
        if is_cancelled():
            raise OperationCancelled()
        image = cv2.imread(os.path.join(folder, name))
        if image is None:
            continue
        height, width = image.shape[:2]
        if detector is None or size != (width, height):
            detector, size = _detector(width, height), (width, height)
            detector.setInputSize((width, height))
        number = int(FRAME_NAME.search(name).group(1))
        frames.append(((number - 1) / fps, _faces_in(detector, image)))
        on_progress((index + 1) / len(names))
    return _per_second(frames, duration)

def framing_note(measurements: Sequence[FaceMeasurement]) -> Optional[dict]:
    """Sum up where the faces sat across a stretch, for deciding a reframe.

    Args:
        measurements: The per-second records over the stretch.

    Returns:
        The share of the stretch that had anybody on screen, how many faces
        were usually up, and the middle and spread of where the largest face
        sat across the frame. `None` when nothing was measured. The spread is
        what says whether one crop can hold the whole stretch or whether it
        has to move.
    """
    if not measurements:
        return None
    placed = [record for record in measurements if record.face_x is not None]
    note = {
        "seconds": len(measurements),
        "with_a_face": round(len(placed) / len(measurements), 3),
        "faces_median": statistics.median(record.faces for record in measurements),
    }
    if placed:
        across = [record.face_x for record in placed]
        note["face_x_median"] = round(statistics.median(across), 3)
        note["face_x_range"] = [round(min(across), 3), round(max(across), 3)]
        note["face_share_median"] = round(statistics.median(record.face_share for record in placed), 4)
    return note
