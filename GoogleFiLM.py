from google_film import interpolator
import tensorflow as tf
import numpy as np


class GoogleFiLM:
    def __init__(self) -> None:
        model_path = 'model_weights/google_film/saved_model'
        self.interpolator = interpolator.Interpolator(model_path=model_path)

    def interpolate_frame(self, frame1, frame2):
            
        # First batched image.
        frame1 = frame1 / float(np.iinfo(np.uint8).max) # normalize to [0,1]
        frame1 = np.expand_dims(frame1, axis=0)

        # Second batched image.
        frame2 = frame2 / float(np.iinfo(np.uint8).max) # normalize to [0,1]
        frame2 = np.expand_dims(frame2, axis=0)

        # Batched time.
        batch_dt = np.full(shape=(1,), fill_value=0.5, dtype=np.float32)

        # Invoke the model for one mid-frame interpolation.
        mid_frame = interpolator.interpolate(frame1, frame2, batch_dt)[0]

        # return mid_frame
        return mid_frame