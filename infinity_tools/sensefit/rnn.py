import os
from infinity_tools.common.ml.rnn import BaseModel
from infinity_tools.common.vis.notebook import display_video_as_gif
from infinity_tools.common.vis.videos import overlay_repcount_pred
from infinity_tools.sensefit.datagen import SenseFitGenerator
from infinity_tools.sensefit.vis import plot_predictions


class SenseFitModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.generator = SenseFitGenerator

    def display_predictions(self, data_path, pred_count, output_tag):
        """Overlays rep count predictions onto video."""

        video_path = data_path.replace(".csv", ".mp4")
        if os.path.exists(video_path):
            if output_tag is not None:
                _output_tag = f"_{output_tag}"
            else:
                _output_tag = ""
            pred_path = data_path.replace(".csv", f"_pred{_output_tag}.mp4")
            overlay_repcount_pred(video_path, pred_count, pred_path)
            display_video_as_gif(
                pred_path, downsample_resolution=3, downsample_frames=3
            )
        else:
            plot_predictions(data_path, pred_count)
