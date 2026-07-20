import cv2
import torch
import kornia as K
from kornia.image import image_to_tensor


class FeatureMatcher:
    """
    Algorithm for matching cross-season satellite imagery.
    Combines deep feature extraction (LoFTR) with geometric verification (RANSAC).
    """

    def __init__(self, weights_path=None):
        # Load the algorithm's deep learning backbone
        self.matcher = K.feature.LoFTR(pretrained='outdoor' if not weights_path else None)

        if weights_path:
            self.matcher.load_state_dict(torch.load(weights_path)['state_dict'])

        self.matcher = self.matcher.eval()


    def extract_and_match(self, img1, img2, confidence_threshold=0.8):
        """
        Extracts and maps features from two images.
        """
        tensor1 = self._preprocess(img1)
        tensor2 = self._preprocess(img2)

        input_dict = {
            'image0': K.color.rgb_to_grayscale(tensor1),
            'image1': K.color.rgb_to_grayscale(tensor2)
        }

        # Feature Matching
        with torch.inference_mode():
            corresp = self.matcher(input_dict)

        # Confidence Filtering
        mask = corresp['confidence'] > confidence_threshold
        indices = torch.nonzero(mask, as_tuple=True)

        kp0 = corresp['keypoints0'][indices].cpu().numpy()
        kp1 = corresp['keypoints1'][indices].cpu().numpy()

        # Geometric Verification (RANSAC)
        inliers = None
        if len(kp0) >= 8:  # Minimum points required for Fundamental Matrix
            try:
                _, inliers = cv2.findFundamentalMat(kp0, kp1, cv2.USAC_ACCURATE, 1, 0.99, 100000)
                inliers = inliers > 0
            except Exception as e:
                pass

        return kp0, kp1, inliers, tensor1, tensor2

    def _preprocess(self, image):
        # Normalizes RGB arrays to float32 tensors
        img_tensor = image_to_tensor(image).float() / 255.0
        return img_tensor.unsqueeze(0)