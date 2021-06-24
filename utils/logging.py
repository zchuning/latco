import imageio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

from utils import tools


class TBLogger:
  def __init__(self, dir):
    writer = tf.summary.create_file_writer(str(dir), max_queue=1000, flush_millis=20000)
    writer.set_as_default()
    self.dir = dir
    self.writer = writer

  def log_graph(self, name, curves):
    for curve_name, curve in curves.items():
      for i, v in enumerate(curve):
        tf.summary.scalar(curve_name, v, i)
    self.writer.flush()

  def log_hist(self, name, data):
    tf.summary.histogram(name, data)
    self.writer.flush()

  def log_graph_hist(self, name, curves):
    for curve_name, curve in curves.items():
      self.log_graph(None, {curve_name: curve})
      self.log_hist('h' + curve_name, curve)
    self.writer.flush()

  def log_image(self, name, image):
    """

    :param name:
    :param image: HxWxC
    :return:
    """
    if np.issubdtype(image.dtype, np.floating):
      image = image + 0.5
    tools.image_summary(name, image[None])
    # tools.graph_summary(self.writer, tools.image_summary, name, image)
    self.writer.flush()

  def log_video(self, name, video):
    """

    :param name:
    :param video: (B x) T x H x W x C
    :return:
    """
    if isinstance(video, list):
      video = np.array(video)
    if np.issubdtype(video.dtype, np.floating):
      video = video + 0.5
    tools.video_summary(name, video)
    # tools.graph_summary(self.writer, tools.video_summary, name, video)
    self.writer.flush()

  def log_scatter(self, name, data):
    height=400
    width=400
    dpi=10
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    plt.xticks(fontsize=100)
    plt.yticks(fontsize=100)
    plt.scatter(data[0], data[1], lw=30)
    plt.grid()
    plt.tight_layout()
    fig_img = fig2img(fig)
    plt.close(fig)
    self.log_image(name, fig_img - 0.5)


def fig2img(fig):
  """Converts a given figure handle to a 3-channel numpy image array."""
  fig.canvas.draw()
  w, h = fig.canvas.get_width_height()
  buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
  buf.shape = (w, h, 4)
  buf = np.roll(buf, 3, axis=2)
  w, h, d = buf.shape
  return np.array(Image.frombytes("RGBA", (w, h), buf.tostring()), dtype=np.float32)[:, :, :3] / 255.
