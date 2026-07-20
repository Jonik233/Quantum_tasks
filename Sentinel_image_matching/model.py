import cv2
import torch
import kornia as K
from kornia_moons.viz import draw_LAF_matches
from kornia.image import image_to_tensor


class LoFTR_Matcher:
    def __init__(self, image_size=None):
        self.image_size = image_size
        self.matcher = K.feature.LoFTR(pretrained='outdoor').eval()

    def __call__(self, image0, image1, confidence_min=0.8, accurate=True):
        tensor0 = self._convert_image(image0)
        tensor1 = self._convert_image(image1)

        input_dict = {
            'image0': K.color.rgb_to_grayscale(tensor0),
            'image1': K.color.rgb_to_grayscale(tensor1)
        }
        with torch.inference_mode():
            corresp = self.matcher(input_dict)

        mask = corresp['confidence'] > confidence_min
        indices = torch.nonzero(mask, as_tuple=True)
        keypoints0 = corresp['keypoints0'][indices].cpu().numpy()
        keypoints1 = corresp['keypoints1'][indices].cpu().numpy()
        confidence = corresp['confidence'][indices].cpu().numpy()

        fmat_method = cv2.USAC_ACCURATE if accurate else cv2.USAC_MAGSAC
        try:
            fmat, inliers = cv2.findFundamentalMat(keypoints0, keypoints1, fmat_method, 1, 0.99, 100000)
            inliers = inliers > 0
        except:
            inliers = None

        return {
            'image0': tensor0,  # [ADDED] Required for drawing matches later
            'image1': tensor1,  # [ADDED] Required for drawing matches later
            'keypoints0': keypoints0,
            'keypoints1': keypoints1,
            'confidence': confidence,
            'inliers': inliers
        }

    @staticmethod
    def draw_matches(match_dict):
        """Uses kornia_moons to plot the keypoints and inlier lines"""
        output_fig = draw_LAF_matches(
            K.feature.laf_from_center_scale_ori(
                torch.from_numpy(match_dict['keypoints0']).view(1, -1, 2),
                torch.ones(match_dict['keypoints0'].shape[0]).view(1, -1, 1, 1),
                torch.ones(match_dict['keypoints0'].shape[0]).view(1, -1, 1),
            ),
            K.feature.laf_from_center_scale_ori(
                torch.from_numpy(match_dict['keypoints1']).view(1, -1, 2),
                torch.ones(match_dict['keypoints1'].shape[0]).view(1, -1, 1, 1),
                torch.ones(match_dict['keypoints1'].shape[0]).view(1, -1, 1),
            ),
            torch.arange(match_dict['keypoints0'].shape[0]).view(-1, 1).repeat(1, 2),
            K.tensor_to_image(match_dict['image0']),
            K.tensor_to_image(match_dict['image1']),
            match_dict['inliers'],
            draw_dict={
                'inlier_color': (0.2, 1, 0.2),  # Green for good matches
                'tentative_color': (1, 0.1, 0.1),
                'feature_color': (0.2, 0.5, 1),
                'vertical': False}
        )
        return output_fig

    def _convert_image(self, image):
        image = image_to_tensor(image)
        image = image.float().unsqueeze(dim=0) / 255.0
        return image