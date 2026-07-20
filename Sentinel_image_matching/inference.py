import cv2
import os
import torch
import argparse
import kornia as K
import pandas as pd
import matplotlib.pyplot as plt
from algorithm import FeatureMatcher
from kornia_moons.viz import draw_LAF_matches


def save_matches(kp0, kp1, inliers, img1_tensor, img2_tensor, output_path=None, close_fig=True):
    """Saves keypoint match visualisations using Kornia Moons."""

    if inliers is None or len(inliers) == 0:
        print("No valid inliers to plot.")
        return None

    fig = draw_LAF_matches(
        K.feature.laf_from_center_scale_ori(
            torch.from_numpy(kp0).view(1, -1, 2),
            torch.ones(kp0.shape[0]).view(1, -1, 1, 1),
            torch.ones(kp0.shape[0]).view(1, -1, 1),
        ),
        K.feature.laf_from_center_scale_ori(
            torch.from_numpy(kp1).view(1, -1, 2),
            torch.ones(kp1.shape[0]).view(1, -1, 1, 1),
            torch.ones(kp1.shape[0]).view(1, -1, 1),
        ),
        torch.arange(kp0.shape[0]).view(-1, 1).repeat(1, 2),
        K.image.tensor_to_image(img1_tensor),
        K.image.tensor_to_image(img2_tensor),
        inliers,
        draw_dict={
            'inlier_color': (0.2, 1.0, 0.2),
            'tentative_color': (1.0, 0.1, 0.1),
            'vertical': False
        }
    )

    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', dpi=300)

    if close_fig:
        plt.close(fig)



def match_image_pair(matcher, img1_path, img2_path, out_path=None):
    """Runs inference on a single cross-season image pair and optionally creates a visualisation."""

    img1 = cv2.cvtColor(cv2.imread(img1_path), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread(img2_path), cv2.COLOR_BGR2RGB)

    kp0, kp1, inliers, tensor1, tensor2 = matcher.extract_and_match(img1, img2)

    total_detected = len(kp0)
    num_inliers = int(sum(inliers)[0]) if inliers is not None else 0
    inlier_ratio = (num_inliers / total_detected) if total_detected > 0 else 0.0

    if out_path:
        save_matches(
            kp0, kp1, inliers, tensor1, tensor2,
            output_path=out_path
        )

    return {
        'kp0': kp0,
        'kp1': kp1,
        'inliers': inliers,
        'num_raw_matches': total_detected,
        'num_inliers': num_inliers,
        'inlier_ratio': inlier_ratio,
        'tensor1': tensor1,
        'tensor2': tensor2
    }


def run_batch_evaluation(matcher, csv_path, dataset_dir, out_dir=None, min_inlier_thresh=15):
    """Executes inference across all pairs in dataset_labels.csv and returns metrics DataFrame."""

    df = pd.read_csv(csv_path)
    results = []

    for idx, row in df.iterrows():
        patch_id = row['patch_id']
        path_a = os.path.join(dataset_dir, row['image_A_path'])
        path_b = os.path.join(dataset_dir, row['image_B_path'])

        save_path = os.path.join(out_dir, f"{patch_id}_match.png") if out_dir else None

        pair_res = match_image_pair(matcher, path_a, path_b, out_path=save_path)

        results.append({
            'patch_id': patch_id,
            'image_A_path': row['image_A_path'],
            'image_B_path': row['image_B_path'],
            'raw_matches': pair_res['num_raw_matches'],
            'inliers': pair_res['num_inliers'],
            'inlier_ratio': pair_res['inlier_ratio'],
            'success': pair_res['num_inliers'] >= min_inlier_thresh
        })

    results_df = pd.DataFrame(results)

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        results_df.to_csv(os.path.join(out_dir, "batch_evaluation_metrics.csv"), index=False)

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-Season Inference Script")
    parser.add_argument("--img1", default=None, help="Path to Season A image")
    parser.add_argument("--img2", default=None, help="Path to Season B image")
    parser.add_argument("--csv", default=None, help="Path to dataset_labels.csv")
    parser.add_argument("--dataset_dir", default="image_matching_dataset", help="Dataset directory")
    parser.add_argument("--weights", default=None, help="Model weights path")
    parser.add_argument("--out", default="inference_output", help="Output directory or file path")

    args = parser.parse_args()
    matcher = FeatureMatcher(weights_path=args.weights)

    if args.img1 and args.img2:
        res = match_image_pair(matcher, args.img1, args.img2, out_path=args.out)
        print(f"Single pair inference finished: {res['num_inliers']} inliers found.")
    elif args.csv:
        res_df = run_batch_evaluation(matcher, args.csv, args.dataset_dir, out_dir=args.out)
        print(f"Batch evaluation finished for {len(res_df)} pairs.")
        print(f"Mean Inliers: {res_df['inliers'].mean():.2f} | Success Rate: {res_df['success'].mean():.2%}")
    else:
        print("Please provide --img1 and --img2 for single mode OR --csv for batch mode.")