import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def get_mask_rcnn_model(num_classes: int):
    """
    Builds a Mask R-CNN model with a ResNet50-FPN backbone and replaces the
    pre-trained classification and mask heads with ones matching the dataset.
    
    Args:
        num_classes (int): Number of object classes + 1 (for Mask R-CNN's internal background).
                           For Fostride: Metal (1), Paper (2), Plastic (3) -> num_classes = 4.
                           
    Returns:
        torchvision.models.detection.MaskRCNN: The PyTorch model.
    """
    # Load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT")
    
    # 1. Replace the Bounding Box Predictor
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # 2. Replace the Mask Predictor
    # Get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    
    # Replace the pre-trained mask predictor
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)
                                                       
    return model
