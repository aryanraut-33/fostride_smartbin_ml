#!/usr/bin/python3
import rclpy
import cv2
import os
import json
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms.functional as F
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from datetime import datetime

# ================================
# DEVICE SETUP
# ================================
device = torch.device("cuda") if torch.cuda.is_available() else \
         torch.device("mps") if torch.backends.mps.is_available() else \
         torch.device("cpu")

# ================================
# BUILD MASK R-CNN MODEL
# ================================
def get_mask_rcnn_model(num_classes: int):
    # Load an instance segmentation model pre-trained on COCO
    # Using weights=None since we will load our own state_dict
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
    
    # 1. Replace the Bounding Box Predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # 2. Replace the Mask Predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
    
    return model

# ================================
# LOAD MODEL WEIGHTS
# ================================
# 4 classes: Metal (1), Paper (2), Plastic (3) + Background (0)
NUM_CLASSES = 4
model = get_mask_rcnn_model(num_classes=NUM_CLASSES)

MODEL_PATH = "/home/wise/ros2_ws/src/r3bin/r3bin/nodes/best_m1_baseline.pt"
if os.path.exists(MODEL_PATH):
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
else:
    print(f"WARNING: Model not found at {MODEL_PATH}")

model.to(device)
model.eval() # Set to evaluation mode

# ================================
# SAVE DIRECTORY (ROOT)
# ================================
SAVE_ROOT = "/home/wise/capturedImages"

# ================================
# CLASS MAP
# ================================
# Mask R-CNN outputs indices. We map them internally.
# Background=0, Metal=1, Paper=2, Plastic=3
ID_TO_CLASS = {
    1: "metal",
    2: "paper",
    3: "plastic"
}
VALID_CLASSES = ["paper", "plastic", "metal", "mixed"]


class CameraToServo(Node):
    def __init__(self):
        super().__init__("camera_to_servo")

        self.bridge_ = CvBridge()
        self.mixedId = 0

        self.subscriber_ = self.create_subscription(
            Image,
            "bboxImage",
            self.image_callback,
            10
        )

        self.publisher_ = self.create_publisher(String, "fromCamera", 10)
        self.pub_ = self.create_publisher(String, "toCamera", 10)

        self.get_logger().info(f"Camera to Servo node started! Using device: {device}")

    # ================================
    # CLASS EXTRACTION
    # ================================
    def extractClass(self, outputs):
        """
        Extracts the primary class from Mask R-CNN outputs.
        outputs is a dictionary containing 'boxes', 'labels', 'scores', 'masks'
        """
        if len(outputs["boxes"]) == 0:
            return "mixed"

        detected = set()

        for i in range(len(outputs["scores"])):
            conf = outputs["scores"][i].item()
            if conf < 0.35:
                continue

            classId = outputs["labels"][i].item()
            if classId in ID_TO_CLASS:
                className = ID_TO_CLASS[classId]
                detected.add(className)

        if len(detected) == 0:
            return "mixed"

        if len(detected) > 1:
            return "mixed"

        return list(detected)[0]

    # ================================
    # CREATE FOLDER PATH
    # ================================
    def get_save_path(self, wasteType):
        today = datetime.now().strftime("%Y-%m-%d")

        wasteType = wasteType.lower()
        if wasteType not in VALID_CLASSES:
            wasteType = "mixed"

        path = os.path.join(SAVE_ROOT, today, wasteType)

        # 🔥 AUTO-CREATE FOLDERS ON FIRST IMAGE
        os.makedirs(path, exist_ok=True)

        return path

    # ================================
    # SAVE RAW IMAGE
    # ================================
    def saveRawImage(self, frame, wasteType):
        folder = self.get_save_path(wasteType)

        timestamp = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
        filename = os.path.join(folder, f"{timestamp}.jpg")

        cv2.imwrite(filename, frame)
        self.get_logger().info(f"Saved raw image: {filename}")

        return timestamp

    # ================================
    # IMAGE CALLBACK
    # ================================
    def image_callback(self, msg):
        # 1. Convert ROS Image msg to OpenCV BGR
        frame = self.bridge_.imgmsg_to_cv2(msg, "bgr8")
        
        # 2. Convert BGR to RGB (Mask R-CNN expects RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 3. Convert to PyTorch Tensor [C, H, W] in range [0.0, 1.0]
        tensor_img = F.to_tensor(frame_rgb).to(device)

        # 4. Run Inference (Mask R-CNN takes a list of tensors)
        with torch.no_grad():
            outputs = model([tensor_img])[0]  # Get dict for the first (and only) image

        # 5. Extract class based on scores > 0.35
        wasteType = self.extractClass(outputs)

        # 6. Save BGR image (OpenCV expects BGR for writing)
        imageId = self.saveRawImage(frame, wasteType)

        payload = {
            "id": imageId,
            "error": "none",
            "type": wasteType
        }

        outMsg = String()
        outMsg.data = json.dumps(payload)
        self.publisher_.publish(outMsg)

        self.get_logger().info(f"Published JSON: {outMsg.data}")


# ================================
# MAIN
# ================================
def main():
    rclpy.init()
    node = CameraToServo()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node stopped by user")
    except Exception as e:
        node.get_logger().error(f"Unexpected error: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

#*/5 * * * * /usr/bin/rclone copy /home/wise/capturedImages dropbox:capturedImages --create-empty-src-dirs >> /home/wise/rclone.log 2>&1.