from enum import Enum


class LaneModelType(Enum):
    UFLD_TUSIMPLE = 0
    UFLD_CULANE = 1
    UFLDV2_TUSIMPLE = 2
    UFLDV2_CULANE = 3
    UFLDV2_CURVELANES = 4


class OffsetType(Enum):
    UNKNOWN = "检测中 ..."
    RIGHT = "请保持右侧行驶"
    LEFT = "请保持左侧行驶"
    CENTER = "车道保持良好"  # Good Lane Keeping


class CurvatureType(Enum):
    UNKNOWN = "检测中 ..."
    STRAIGHT = "直行"
    EASY_LEFT = "左转"
    HARD_LEFT = "左转"
    EASY_RIGHT = "右转"
    HARD_RIGHT = "右转"


lane_colors = [(255, 0, 0), (46, 139, 87), (50, 205, 50), (0, 255, 255)]
