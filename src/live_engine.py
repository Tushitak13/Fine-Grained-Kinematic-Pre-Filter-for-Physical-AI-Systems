"""
live_engine.py

Handles webcam capture + MediaPipe hand tracking + live scoring via
DeviationFilter. Used by live_view.py to drive the Live Demo tab.
"""

import cv2
import mediapipe as mp
import numpy as np

import data_loader
from filter_logic import DeviationFilter

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


class LiveEngine:
    def __init__(self):
        ref_mean, ref_std, ref_is_real = data_loader.load_reference_envelope()
        self.reference_is_real = ref_is_real
        self.scorer = DeviationFilter(ref_mean, ref_std)
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.cap = None
        self._frame_index = 0
        self._ref_length = len(ref_mean)

    def start(self):
        self.cap = cv2.VideoCapture(0)
        self._frame_index = 0
        self.scorer.reset()

    def stop(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def read_and_score(self):
        """
        Reads one frame from the webcam, runs hand tracking, and scores it.
        Returns: (frame_bgr_with_overlay, score_dict_or_None)
        score_dict_or_None is None if no hand was detected this frame.
        """
        if self.cap is None:
            return None, None

        success, frame = self.cap.read()
        if not success:
            return None, None

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        score = None
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            # Draw skeleton overlay directly on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            coords = []
            for lm in hand_landmarks.landmark:
                coords.extend([lm.x, lm.y, lm.z])
            frame_vec = np.array(coords, dtype=np.float32)

            ref_index = min(self._frame_index, self._ref_length - 1)
            score = self.scorer.score_frame(frame_vec, ref_index)
            score["frame"] = self._frame_index

        self._frame_index += 1

        # Convert back to RGB for Streamlit display
        display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return display_frame, score