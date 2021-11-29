"""ResNet50, 1x schedule."""

_base_ = [
    "../_base_/models/res50.py",
    "../_base_/datasets/bdd100k.py",
    "../_base_/schedules/schedule_1x.py",
    "../_base_/default_runtime.py",
]
load_from = (
    "https://dl.cv.ethz.ch/bdd100k/pose/models/res50_256x192_pose_bdd100k.pth"
)