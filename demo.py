import cv2
import time
import numpy as np
import logging
from PIL import Image, ImageDraw, ImageFont
# import pycuda.driver as drv

from ObjectTracker import BYTETracker
from taskConditions import TaskConditions, Logger
from ObjectDetector import YoloDetector, EfficientdetDetector
from ObjectDetector.utils import ObjectModelType,  CollisionType
from ObjectDetector.distanceMeasure import SingleCamDistanceMeasure

from TrafficLaneDetector import UltrafastLaneDetector, UltrafastLaneDetectorV2
from TrafficLaneDetector.ufldDetector.perspectiveTransformation import PerspectiveTransformation
from TrafficLaneDetector.ufldDetector.utils import LaneModelType, OffsetType, CurvatureType
LOGGER = Logger(None, logging.INFO, logging.INFO)


video_path = "./models/video.mp4"

lane_config = {
    "model_path": "./models/ufldv2_culane_res18_320x1600.onnx",
    "model_type": LaneModelType.UFLDV2_CULANE
}

object_config = {
    "model_path": './models/yolov8l.onnx',
    "model_type": ObjectModelType.YOLOV8,
    "classes_path": './ObjectDetector/models/coco_label.txt',
    "box_score": 0.4,
    "box_nms_iou": 0.5
}

# 全局字体变量
_chinese_font = None


def load_chinese_font(size=20):
    """加载中文字体，只在第一次调用时加载"""
    global _chinese_font
    if _chinese_font is None:
        font_paths = [
            '/System/Library/Fonts/Supplemental/Songti.ttc',
            '/System/Library/Fonts/STHeiti Medium.ttc',
            '/System/Library/Fonts/Hiragino Sans GB.ttc',
            '/System/Library/Fonts/STHeiti Light.ttc',
            '/System/Library/Fonts/Supplemental/Songti.ttc',
            '/System/Library/Fonts/CJKSymbolsFallback.ttc'
        ]

        for font_path in font_paths:
            try:
                _chinese_font = ImageFont.truetype(font_path, size)
                print(f"成功加载字体: {font_path}")
                break
            except (OSError, IOError):
                continue

        if _chinese_font is None:
            _chinese_font = ImageFont.load_default()
            print("使用默认字体")

    return _chinese_font


def put_chinese_text(img, text, position, font_size=20, color=(255, 255, 255)):
    """
    在图像上显示中文文字

    Args:
        img: OpenCV图像
        text: 要显示的文字
        position: 文字位置 (x, y)
        font_size: 字体大小
        color: 文字颜色 (B, G, R)

    Returns:
        img: 添加文字后的图像
    """
    try:
        # 使用预加载的字体
        font = load_chinese_font(font_size)

        # 将OpenCV图像转换为PIL图像
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)

        # 转换颜色格式 (BGR -> RGB)
        color_rgb = (color[2], color[1], color[0])

        # 绘制文字
        draw.text(position, text, font=font, fill=color_rgb)

        # 转换回OpenCV格式
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return img_cv

    except Exception as e:
        print(f"中文文字绘制失败: {e}")
        # 备选方案：使用cv2.putText
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2, cv2.LINE_AA)
        return img

# Priority : FCWS > LDWS > LKAS


class ControlPanel(object):
	CollisionDict = {
						CollisionType.UNKNOWN : (0, 255, 255),
						CollisionType.NORMAL : (0, 255, 0),
						CollisionType.PROMPT : (0, 102, 255),
						CollisionType.WARNING : (0, 0, 255)
	 				}

	OffsetDict = { 
					OffsetType.UNKNOWN : (0, 255, 255), 
					OffsetType.RIGHT :  (0, 0, 255), 
					OffsetType.LEFT : (0, 0, 255), 
					OffsetType.CENTER : (0, 255, 0)
				 }

	CurvatureDict = { 
						CurvatureType.UNKNOWN : (0, 255, 255),
						CurvatureType.STRAIGHT : (0, 255, 0),
						CurvatureType.EASY_LEFT : (0, 102, 255),
						CurvatureType.EASY_RIGHT : (0, 102, 255),
						CurvatureType.HARD_LEFT : (0, 0, 255),
						CurvatureType.HARD_RIGHT : (0, 0, 255)
					}

    def __init__(self):
        # 初始化中文字体
        print("初始化中文字体...")
        load_chinese_font(20)

        collision_warning_img = cv2.imread(
            './assets/FCWS-warning.png', cv2.IMREAD_UNCHANGED)
        self.collision_warning_img = cv2.resize(
            collision_warning_img, (100, 100))
        collision_prompt_img = cv2.imread(
            './assets/FCWS-prompt.png', cv2.IMREAD_UNCHANGED)
        self.collision_prompt_img = cv2.resize(
            collision_prompt_img, (100, 100))
        collision_normal_img = cv2.imread(
            './assets/FCWS-normal.png', cv2.IMREAD_UNCHANGED)
        self.collision_normal_img = cv2.resize(
            collision_normal_img, (100, 100))
        left_curve_img = cv2.imread(
            './assets/left_turn.png', cv2.IMREAD_UNCHANGED)
        self.left_curve_img = cv2.resize(left_curve_img, (200, 200))
        right_curve_img = cv2.imread(
            './assets/right_turn.png', cv2.IMREAD_UNCHANGED)
        self.right_curve_img = cv2.resize(right_curve_img, (200, 200))
        keep_straight_img = cv2.imread(
            './assets/straight.png', cv2.IMREAD_UNCHANGED)
        self.keep_straight_img = cv2.resize(keep_straight_img, (200, 200))
        determined_img = cv2.imread('./assets/warn.png', cv2.IMREAD_UNCHANGED)
        self.determined_img = cv2.resize(determined_img, (200, 200))
        left_lanes_img = cv2.imread(
            './assets/LTA-left_lanes.png', cv2.IMREAD_UNCHANGED)
        self.left_lanes_img = cv2.resize(left_lanes_img, (300, 200))
        right_lanes_img = cv2.imread(
            './assets/LTA-right_lanes.png', cv2.IMREAD_UNCHANGED)
        self.right_lanes_img = cv2.resize(right_lanes_img, (300, 200))


		# FPS
		self.fps = 0
		self.frame_count = 0
		self.start = time.time()

		self.curve_status = None

	def _updateFPS(self) :
		"""
		Update FPS.

		Args:
			None

		Returns:
			None
		"""
		self.frame_count += 1
		if self.frame_count >= 30:
			self.end = time.time()
			self.fps = self.frame_count / (self.end - self.start)
			self.frame_count = 0
			self.start = time.time()

	def DisplayBirdViewPanel(self, main_show, min_show, show_ratio=0.25) :
		"""
		Display BirdView Panel on image.

		Args:
			main_show: video image.
			min_show: bird view image.
			show_ratio: display scale of bird view image.

		Returns:
			main_show: Draw bird view on frame.
		"""
		W = int(main_show.shape[1]* show_ratio)
		H = int(main_show.shape[0]* show_ratio)

		min_birdview_show = cv2.resize(min_show, (W, H))
		min_birdview_show = cv2.copyMakeBorder(min_birdview_show, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0, 0, 0]) # 添加边框
		main_show[0:min_birdview_show.shape[0], -min_birdview_show.shape[1]: ] = min_birdview_show

	def DisplaySignsPanel(self, main_show, offset_type, curvature_type) :
		"""
		Display Signs Panel on image.

		Args:
			main_show: image.
			offset_type: offset status by OffsetType. (UNKNOWN/CENTER/RIGHT/LEFT)
			curvature_type: curature status by CurvatureType. (UNKNOWN/STRAIGHT/HARD_LEFT/EASY_LEFT/HARD_RIGHT/EASY_RIGHT)

		Returns:
			main_show: Draw sings info on frame.
		"""

		W = 400
		H = 365
		widget = np.copy(main_show[:H, :W])
		widget //= 2
		widget[0:3,:] = [0, 0, 255]  # top
		widget[-3:-1,:] = [0, 0, 255] # bottom
		widget[:,0:3] = [0, 0, 255]  #left
		widget[:,-3:-1] = [0, 0, 255] # right
		main_show[:H, :W] = widget

		if curvature_type == CurvatureType.UNKNOWN and offset_type in { OffsetType.UNKNOWN, OffsetType.CENTER } :
			y, x = self.determined_img[:,:,3].nonzero()
			main_show[y+10, x-100+W//2] = self.determined_img[y, x, :3]
			self.curve_status = None

		elif (curvature_type == CurvatureType.HARD_LEFT or self.curve_status== "Left") and \
			(curvature_type not in { CurvatureType.EASY_RIGHT, CurvatureType.HARD_RIGHT }) :
			y, x = self.left_curve_img[:,:,3].nonzero()
			main_show[y+10, x-100+W//2] = self.left_curve_img[y, x, :3]
			self.curve_status = "Left"

		elif (curvature_type == CurvatureType.HARD_RIGHT or self.curve_status== "Right") and \
			(curvature_type not in { CurvatureType.EASY_LEFT, CurvatureType.HARD_LEFT }) :
			y, x = self.right_curve_img[:,:,3].nonzero()
			main_show[y+10, x-100+W//2] = self.right_curve_img[y, x, :3]
			self.curve_status = "Right"
		
		
		if ( offset_type == OffsetType.RIGHT ) :
			y, x = self.left_lanes_img[:,:,2].nonzero()
			main_show[y+10, x-150+W//2] = self.left_lanes_img[y, x, :3]
		elif ( offset_type == OffsetType.LEFT ) :
			y, x = self.right_lanes_img[:,:,2].nonzero()
			main_show[y+10, x-150+W//2] = self.right_lanes_img[y, x, :3]
		elif curvature_type == CurvatureType.STRAIGHT or self.curve_status == "Straight" :
			y, x = self.keep_straight_img[:,:,3].nonzero()
			main_show[y+10, x-100+W//2] = self.keep_straight_img[y, x, :3]
			self.curve_status = "Straight"

        self._updateFPS()
        main_show = put_chinese_text(main_show, f"车道偏离预警: {offset_type.value}", (10, 240),
                                     font_size=16, color=self.OffsetDict[offset_type])
        main_show = put_chinese_text(main_show, f"车道保持辅助: {curvature_type.value}", (10, 270),
                                     font_size=16, color=self.CurvatureDict[curvature_type])
        cv2.putText(main_show, "FPS  : %.2f" % self.fps, (10,
                    widget.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        return main_show

    def DisplayCollisionPanel(self, main_show, collision_type, obect_infer_time, lane_infer_time, show_ratio=0.25):
        """
        Display Collision Panel on image.

		Args:
			main_show: image.
			collision_type: collision status by CollisionType. (WARNING/PROMPT/NORMAL)
			obect_infer_time: object detection time -> float.
			lane_infer_time:  lane detection time -> float.

		Returns:
			main_show: Draw collision info on frame.
		"""

		W = int(main_show.shape[1]* show_ratio)
		H = int(main_show.shape[0]* show_ratio)

		widget = np.copy(main_show[H+20:2*H, -W-20:])
		widget //= 2
		widget[0:3,:] = [0, 0, 255]  # top
		widget[-3:-1,:] = [0, 0, 255] # bottom
		widget[:,-3:-1] = [0, 0, 255] #left
		widget[:,0:3] = [0, 0, 255]  # right
		main_show[H+20:2*H, -W-20:] = widget

		if (collision_type == CollisionType.WARNING) :
			y, x = self.collision_warning_img[:,:,3].nonzero()
			main_show[H+y+50, (x-W-5)] = self.collision_warning_img[y, x, :3]
		elif (collision_type == CollisionType.PROMPT) :
			y, x =self.collision_prompt_img[:,:,3].nonzero()
			main_show[H+y+50, (x-W-5)] = self.collision_prompt_img[y, x, :3]
		elif (collision_type == CollisionType.NORMAL) :
			y, x = self.collision_normal_img[:,:,3].nonzero()
			main_show[H+y+50, (x-W-5)] = self.collision_normal_img[y, x, :3]

        main_show = put_chinese_text(main_show, f"碰撞预警: {collision_type.value}",
                                     (main_show.shape[1] - int(W) + 100, 240),
                                     font_size=14, color=self.CollisionDict[collision_type])
        main_show = put_chinese_text(main_show, f"对象检测: {obect_infer_time:.2f}s",
                                     (main_show.shape[1] - int(W) + 100, 300),
                                     font_size=14, color=(230, 230, 230))
        main_show = put_chinese_text(main_show, f"车道检测: {lane_infer_time:.2f}s",
                                     (main_show.shape[1] - int(W) + 100, 325),
                                     font_size=14, color=(230, 230, 230))

        return main_show


if __name__ == "__main__":

    # Initialize read and save video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("video path is error. please check it.")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    vout = cv2.VideoWriter(
        video_path[:-4]+'_out.mp4', fourcc, 30.0, (width, height))
    cv2.namedWindow("ADAS Simulation", cv2.WINDOW_NORMAL)

    # ==========================================================
    #                   Initialize Class
    # ==========================================================
    # LOGGER.info("[Pycuda] Cuda Version: {}".format(drv.get_version()))
    # LOGGER.info("[Driver] Cuda Version: {}".format(drv.get_driver_version()))
    LOGGER.info("-"*40)

    # lane detection model
    LOGGER.info("Detector Model Type : {}".format(
        lane_config["model_type"].name))
    if "UFLDV2" in lane_config["model_type"].name:
        UltrafastLaneDetectorV2.set_defaults(lane_config)
        laneDetector = UltrafastLaneDetectorV2(logger=LOGGER)
    else:
        UltrafastLaneDetector.set_defaults(lane_config)
        laneDetector = UltrafastLaneDetector(logger=LOGGER)
    transformView = PerspectiveTransformation((width, height), logger=LOGGER)

    # object detection model
    LOGGER.info("ObjectDetector Model Type : {}".format(
        object_config["model_type"].name))
    if ObjectModelType.EfficientDet == object_config["model_type"]:
        EfficientdetDetector.set_defaults(object_config)
        objectDetector = EfficientdetDetector(logger=LOGGER)
    else:
        YoloDetector.set_defaults(object_config)
        objectDetector = YoloDetector(logger=LOGGER)
    distanceDetector = SingleCamDistanceMeasure()
    objectTracker = BYTETracker(names=objectDetector.colors_dict)

    # display panel
    displayPanel = ControlPanel()
    analyzeMsg = TaskConditions()

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    LOGGER.info(f"开始处理视频，总帧数: {total_frames}")

    while cap.isOpened():

        ret, frame = cap.read()  # Read frame from the video
        if ret:
            frame_show = frame.copy()

            # ========================== Detect Model =========================
            obect_time = time.time()
            objectDetector.DetectFrame(frame)
            obect_infer_time = round(time.time() - obect_time, 2)

            if objectTracker:
                box = [obj.tolist(format_type="xyxy")
                       for obj in objectDetector.object_info]
                score = [obj.conf for obj in objectDetector.object_info]
                id = [obj.label for obj in objectDetector.object_info]
                # id = [objectDetector.class_names.index(obj.label) for obj in objectDetector.object_info]
                objectTracker.update(box, score, id, frame)

            lane_time = time.time()
            laneDetector.DetectFrame(frame)
            lane_infer_time = round(time.time() - lane_time, 4)

            # ========================= Analyze Status ========================
            distanceDetector.updateDistance(objectDetector.object_info)
            vehicle_distance = distanceDetector.calcCollisionPoint(
                laneDetector.lane_info.area_points)

            if (analyzeMsg.CheckStatus() and laneDetector.lane_info.area_status):
                transformView.updateTransformParams(
                    *laneDetector.lane_info.lanes_points[1:3], analyzeMsg.transform_status)
            birdview_show = transformView.transformToBirdView(frame_show)

            birdview_lanes_points = [transformView.transformToBirdViewPoints(
                lanes_point) for lanes_point in laneDetector.lane_info.lanes_points]
            (vehicle_direction, vehicle_curvature), vehicle_offset = transformView.calcCurveAndOffset(
                birdview_show, *birdview_lanes_points[1:3])

            analyzeMsg.UpdateCollisionStatus(
                vehicle_distance, laneDetector.lane_info.area_status)
            analyzeMsg.UpdateOffsetStatus(vehicle_offset)
            analyzeMsg.UpdateRouteStatus(vehicle_direction, vehicle_curvature)

            # ========================== Draw Results =========================
            transformView.DrawDetectedOnBirdView(
                birdview_show, birdview_lanes_points, analyzeMsg.offset_msg)
            if LOGGER.clevel == logging.DEBUG:
                transformView.DrawTransformFrontalViewArea(frame_show)
            laneDetector.DrawDetectedOnFrame(frame_show, analyzeMsg.offset_msg)
            laneDetector.DrawAreaOnFrame(
                frame_show, displayPanel.CollisionDict[analyzeMsg.collision_msg])
            objectDetector.DrawDetectedOnFrame(frame_show)
            objectTracker.DrawTrackedOnFrame(frame_show, False)
            distanceDetector.DrawDetectedOnFrame(frame_show)

            displayPanel.DisplayBirdViewPanel(frame_show, birdview_show)
            frame_show = displayPanel.DisplaySignsPanel(
                frame_show, analyzeMsg.offset_msg, analyzeMsg.curvature_msg)
            frame_show = displayPanel.DisplayCollisionPanel(
                frame_show, analyzeMsg.collision_msg, obect_infer_time, lane_infer_time)

            # 显示处理后的帧
            cv2.imshow("ADAS Simulation", frame_show)

            # 同时保存到输出视频文件
            vout.write(frame_show)

            frame_count += 1
            if frame_count % 30 == 0:  # 每30帧打印一次进度
                progress = (frame_count / total_frames) * 100
                LOGGER.info(
                    f"处理进度: {frame_count}/{total_frames} ({progress:.1f}%)")

        else:
            break

        # 按 'q' 键退出，按 'p' 键暂停/继续
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            cv2.waitKey(0)  # 暂停直到按任意键

    vout.release()
    cap.release()
    cv2.destroyAllWindows()
    LOGGER.info(f"视频处理完成，输出文件: {video_path[:-4]}_out.mp4")
