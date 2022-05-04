from infinity_tools.common.ml.rnn import BaseModel
from infinity_tools.visionfit.datagen import VisionFitGenerator
from infinity_tools.common.vis.videos import overlay_repcount_pred
from infinity_tools.common.vis.notebook import display_video_as_gif
from infinity_tools.visionfit.vis import overlay_movenet_on_video


class VisionFitModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.generator = VisionFitGenerator

    def display_predictions(self, data_path, pred_count, output_tag: str = ""):
        """Overlays rep count predictions onto video."""

        if output_tag != "":
            _output_tag = f"_{output_tag}"
        pred_path = data_path.replace(".mp4", f"_pred{_output_tag}.mp4")
        movenet_path = data_path.replace(".mp4", f"_movenet{_output_tag}.mp4")
        overlay_movenet_on_video(data_path, movenet_path)
        overlay_repcount_pred(movenet_path, pred_count, pred_path, font_scale=0.75)
        display_video_as_gif(pred_path)
