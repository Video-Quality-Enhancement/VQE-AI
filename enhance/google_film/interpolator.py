# Copyright 2022 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A wrapper class for running a frame interpolation TF2 saved model.

Usage:
  model_path='/tmp/saved_model/'
  it = Interpolator(model_path)
  result_batch = it.interpolate(image_batch_0, image_batch_1, batch_dt)

  Where image_batch_1 and image_batch_2 are numpy tensors with TF standard
  (B,H,W,C) layout, batch_dt is the sub-frame time in range [0,1], (B,) layout.
"""
from typing import Optional
import numpy as np
import tensorflow as tf
import onnxruntime as ort


def _pad_to_align(x, align):
  """Pad image batch x so width and height divide by align.

  Args:
    x: Image batch to align.
    align: Number to align to.

  Returns:
    1) An image padded so width % align == 0 and height % align == 0.
    2) A bounding box that can be fed readily to tf.image.crop_to_bounding_box
      to undo the padding.
  """
  # Input checking.
  assert np.ndim(x) == 4
  assert align > 0, 'align must be a positive number.'

  height, width = x.shape[-3:-1]
  height_to_pad = (align - height % align) if height % align != 0 else 0
  width_to_pad = (align - width % align) if width % align != 0 else 0

  bbox_to_pad = {
      'offset_height': height_to_pad // 2,
      'offset_width': width_to_pad // 2,
      'target_height': height + height_to_pad,
      'target_width': width + width_to_pad
  }
  padded_x = tf.image.pad_to_bounding_box(x, **bbox_to_pad)
  bbox_to_crop = {
      'offset_height': height_to_pad // 2,
      'offset_width': width_to_pad // 2,
      'target_height': height,
      'target_width': width
  }
  return padded_x, bbox_to_crop


class Interpolator:
  """A class for generating interpolated frames between two input frames.

  Uses TF2 saved model format.
  """

  def __init__(self, model_path: str,
               align: Optional[int] = None) -> None:
    """Loads a saved model.

    Args:
      model_path: Path to the saved model. If none are provided, uses the
        default model.
      align: 'If >1, pad the input size so it divides with this before
        inference.'
    """
    self._model = tf.keras.models.load_model(model_path)
    self._align = align

  def interpolate(self, x0: np.ndarray, x1: np.ndarray,
                  dt: np.ndarray) -> np.ndarray:
    """Generates an interpolated frame between given two batches of frames.

    All input tensors should be np.float32 datatype.

    Args:
      x0: First image batch. Dimensions: (batch_size, height, width, channels)
      x1: Second image batch. Dimensions: (batch_size, height, width, channels)
      dt: Sub-frame time. Range [0,1]. Dimensions: (batch_size,)

    Returns:
      The result with dimensions (batch_size, height, width, channels).
    """
    if self._align is not None:
      x0, bbox_to_crop = _pad_to_align(x0, self._align)
      x1, _ = _pad_to_align(x1, self._align)

    inputs = {'x0': x0, 'x1': x1, 'time': dt[..., np.newaxis]}
    result = self._model(inputs, training=False)
    image = result['image'].numpy()

    if self._align is not None:
      return tf.image.crop_to_bounding_box(image, **bbox_to_crop)
    return image
  

class Interpolator_onnx:
    """A class for generating interpolated frames between two input frames.

    Uses ONNX model format.
    """

    def __init__(self, model_path: str, align: Optional[int] = None) -> None:
        """Loads an ONNX model.

        Args:
          model_path: Path to the ONNX model. If none are provided, uses the
            default model.
          align: 'If >1, pad the input size so it divides with this before
            inference.'
        """
        providers = [("CUDAExecutionProvider", {"cudnn_conv_use_max_workspace": '1'})]
        sess_options = ort.SessionOptions()
        self._model = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
        self._input_names = [input.name for input in self._model.get_inputs()]
        self._align = align

    def interpolate(self, x0: np.ndarray, x1: np.ndarray,
                    dt: np.ndarray) -> np.ndarray:
        """Generates an interpolated frame between given two batches of frames.

        All input tensors should be np.float32 datatype.

        Args:
          x0: First image batch. Dimensions: (batch_size, height, width, channels)
          x1: Second image batch. Dimensions: (batch_size, height, width, channels)
          dt: Sub-frame time. Range [0,1]. Dimensions: (batch_size,)

        Returns:
          The result with dimensions (batch_size, height, width, channels).
        """
        if self._align is not None:
            x0, bbox_to_crop = _pad_to_align(x0, self._align)
            x1, _ = _pad_to_align(x1, self._align)

        # Run inference
        output = self._model.run(None,
                                 {self._input_names[0]: dt[..., np.newaxis],
                                  self._input_names[1]: x1,
                                  self._input_names[2]: x0}
                                )

        # Process the output as needed
        # for i in range(len(output)):
        #     print(f'output[{i}].shape: {output[i].shape}')
        image = output[17]

        if self._align is not None:
            return tf.image.crop_to_bounding_box(image, **bbox_to_crop)
        return image


class Interpolator_tflite:
    """A class for generating interpolated frames between two input frames.

    Uses TensorFlow Lite model format.
    """

    def __init__(self, model_path: str,
                 align: Optional[int] = None) -> None:
        """Loads a TensorFlow Lite model.

        Args:
          model_path: Path to the TensorFlow Lite model. If none are provided, uses the
            default model.
          align: 'If >1, pad the input size so it divides with this before
            inference.'
        """
        self._interpreter = tf.lite.Interpreter(model_path=model_path)
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

        # Resize input tensors
        self._interpreter.resize_tensor_input(self._input_details[1]['index'], [1, 360, 640, 3])
        self._interpreter.resize_tensor_input(self._input_details[2]['index'], [1, 360, 640, 3])

        self._interpreter.allocate_tensors()

        
        
        self._align = align

    def interpolate(self, x0: np.ndarray, x1: np.ndarray,
                    dt: np.ndarray) -> np.ndarray:
        """Generates an interpolated frame between given two batches of frames.

        All input tensors should be np.float32 datatype.

        Args:
          x0: First image batch. Dimensions: (batch_size, height, width, channels)
          x1: Second image batch. Dimensions: (batch_size, height, width, channels)
          dt: Sub-frame time. Range [0,1]. Dimensions: (batch_size,)

        Returns:
          The result with dimensions (batch_size, height, width, channels).
        """
        if self._align is not None:
            x0, bbox_to_crop = _pad_to_align_tflite(x0, self._align)
            x1, _ = _pad_to_align_tflite(x1, self._align)

        print(f"x0.shape: {x0.shape}, x1.shape: {x1.shape}, dt[..., np.newaxis].shape: {dt[..., np.newaxis].shape}")

        self._interpreter.set_tensor(self._input_details[0]['index'], dt[..., np.newaxis])
        self._interpreter.set_tensor(self._input_details[1]['index'], x1)
        self._interpreter.set_tensor(self._input_details[2]['index'], x0)

        # Invoke inference
        self._interpreter.invoke()

        # Read output tensor values
        image = self._interpreter.get_tensor(self._output_details[1]['index'])

        if self._align is not None:
            return tf.image.crop_to_bounding_box(image, **bbox_to_crop)
        return image


def _pad_to_align_tflite(image: np.ndarray, align: int):
    """Pads an image to the specified alignment.

    Args:
      image: The image to pad.
      align: The alignment to pad to.

    Returns:
      The padded image and a dictionary containing the bounding box of the unpadded image.
    """
    height, width, channels = image.shape
    pad_top = (align - height % align) // 2
    pad_bottom = align - height % align - pad_top
    pad_left = (align - width % align) // 2
    pad_right = align - width % align - pad_left
    image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 'constant')
    bbox = {
        'top': pad_top,
        'bottom': height + pad_top,
        'left': pad_left,
        'right': width + pad_left,
    }
    return image, bbox
