import os
import cv2
import torch
import argparse
import kornia as K
from kornia.image import image_to_tensor
from kornia_moons.viz import draw_LAF_matches
import matplotlib.pyplot as plt


class LoFTR_Inference:
    def __init__(self, weights_path=None):
        """
        Initializes the LoFTR model and loads custom weights if provided.
        """

        # Load the base model
        self.matcher = K.feature.LoFTR(pretrained='outdoor')

        # Load fine-tuned weights if available
        if weights_path and os.path.exists(weights_path):
            print(f"Loading custom weights from {weights_path}...")
            self.matcher.load_state_dict(torch.load(weights_path))
        else:
            print("Warning: Custom weights not found. Using default pre-trained weights.")

        self.matcher.eval()

    def match(self, img_path1, img_path2, confidence_min=0.8):
        """
        Reads two images from disk, processes them, and extracts keypoints.
        """
        # Read images
        img1 = cv2.imread(img_path1)
        img2 = cv2.imread(img_path2)

        if img1 is None or img2 is None:
            raise FileNotFoundError("Could not read one or both image paths.")

        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

        # Convert to tensors
        tensor1 = self._prepare_tensor(img1)
        tensor2 = self._prepare_tensor(img2)

        input_dict = {
            'image0': K.color.rgb_to_grayscale(tensor1),
            'image1': K.color.rgb_to_grayscale(tensor2)
        }

        # Run model
        with torch.inference_mode():
            corresp = self.matcher(input_dict)

        # Filter by confidence
        mask = corresp['confidence'] > confidence_min
        indices = torch.nonzero(mask, as_tuple=True)

        keypoints0 = corresp['keypoints0'][indices].cpu().numpy()
        keypoints1 = corresp['keypoints1'][indices].cpu().numpy()

        # Geometric verification (RANSAC)
        try:
            fmat, inliers = cv2.findFundamentalMat(keypoints0, keypoints1, cv2.USAC_ACCURATE, 1, 0.99, 100000)
            inliers = inliers > 0
        except:
            inliers = None

        return {
            'image0': tensor1,
            'image1': tensor2,
            'keypoints0': keypoints0,
            'keypoints1': keypoints1,
            'inliers': inliers,
            'total_matches': len(keypoints0),
            'valid_inliers': int(sum(inliers)[0]) if inliers is not None else 0
        }

    def _prepare_tensor(self, image):
        image_tensor = image_to_tensor(image)
        image_tensor = image_tensor.float().unsqueeze(dim=0) / 255.0
        return image_tensor

    @staticmethod
    def save_visualization(match_dict, output_path):
        """Generates the side-by-side plot and saves it to disk."""
        fig = draw_LAF_matches(
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
            draw_dict={'inlier_color': (0.2, 1, 0.2), 'tentative_color': (1, 0.1, 0.1), 'vertical': False}
        )
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        print(f"Visualization saved successfully to: {output_path}")
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-Season Satellite Image Matching Inference")
    parser.add_argument("--img1", type=str, required=True, help="Path to first image (e.g., Summer)")
    parser.add_argument("--img2", type=str, required=True, help="Path to second image (e.g., Winter)")
    parser.add_argument("--weights", type=str, default="loftr_satellite_weights.pt",
                        help="Path to custom model weights")
    parser.add_argument("--output", type=str, default="match_result.png", help="Path to save the output visualization")
    parser.add_argument("--conf", type=float, default=0.8, help="Minimum confidence threshold (0.0 to 1.0)")

    args = parser.parse_args()

    # Initialize model
    engine = LoFTR_Inference(weights_path=args.weights)

    # Run matching
    print(f"Comparing {args.img1} and {args.img2}...")
    results = engine.match(args.img1, args.img2, confidence_min=args.conf)

    print(f"Total Keypoints Detected: {results['total_matches']}")
    print(f"Robust Inlier Matches: {results['valid_inliers']}")

    # Save the output visualization
    if results['valid_inliers'] > 0:
        engine.save_visualization(results, args.output)
    else:
        print("Not enough reliable matches found to generate a visualization.")