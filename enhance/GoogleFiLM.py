from .google_film import interpolator as interpolator_lib
import tensorflow as tf
import numpy as np


class google_film_model:
    def __init__(self) -> None:
        self.model_path = 'enhance/model_weights/google_film/saved_model'
        self.interpolator = interpolator_lib.Interpolator(model_path=self.model_path)

    def interpolate_frame(self, frame1, frame2):

        # print(f"frame1.shape: {frame1.shape}, frame2.shape: {frame2.shape}")
            
        # First batched image.
        frame1 = tf.cast(frame1, dtype=tf.float32).numpy()
        frame1 = frame1 / float(np.iinfo(np.uint8).max) # normalize to [0,1]
        frame1 = np.expand_dims(frame1, axis=0)

        # Second batched image.
        frame2 = tf.cast(frame2, dtype=tf.float32).numpy()
        frame2 = frame2 / float(np.iinfo(np.uint8).max) # normalize to [0,1]
        frame2 = np.expand_dims(frame2, axis=0)

        # Batched time.
        batch_dt = np.full(shape=(1,), fill_value=0.5, dtype=np.float32)

        # Invoke the model for one mid-frame interpolation.
        mid_frame = self.interpolator.interpolate(frame1, frame2, batch_dt)[0]

        mid_frame = np.clip(mid_frame * float(np.iinfo(np.uint8).max), 0.0, float(np.iinfo(np.uint8).max))
        mid_frame = (mid_frame + 0.5).astype(np.uint8)

        # return mid_frame
        return mid_frame