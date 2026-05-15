#!/usr/bin/env python
# coding=utf-8

import argparse
import json
import torch
import yaml
import numpy as np
from pathlib import Path

from PIL import Image

from models.scoring_model import ScoringModelRunner
from models.multimodal_encoder.siglip_encoder import SiglipVisionTower
from models.multimodal_encoder.t5_encoder import T5Embedder
from data.bson_vla_dataset import BsonVLADataset
from data.lerobot_vla_dataset import LeRobotVLADataset


def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for scoring model.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/scoring.yaml",
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained scoring model checkpoint.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/ours/true",
        help="Path to the dataset for evaluation.",
    )
    parser.add_argument(
        "--load_from",
        type=str,
        default="bson",
        choices=["bson", "lerobot"],
        help="Type of dataset to load.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="scoring_results.json",
        help="Output file to save scoring results.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples to evaluate. -1 for all samples.",
    )
    parser.add_argument(
        "--episode_aggregation",
        type=str,
        default="mean",
        choices=["mean", "median", "min", "max"],
        help="How to aggregate per-clip scores into a single d(τ) per episode "
             "(Dexora §III-C: K-sub-clip aggregation).",
    )
    parser.add_argument(
        "--pretrained_text_encoder_name_or_path",
        type=str,
        default="google/t5-v1_1-xxl",
        help="Path or HF id of the T5 text encoder used at discriminator training time.",
    )
    parser.add_argument(
        "--pretrained_vision_encoder_name_or_path",
        type=str,
        default="google/siglip-so400m-patch14-384",
        help="Path or HF id of the SigLip vision encoder used at discriminator training time.",
    )
    parser.add_argument(
        "--logpi_file",
        type=str,
        default=None,
        help="Optional path to a JSON of precomputed log-pi values, as produced by compute_logpi.py.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    with open(args.config_path, "r") as fp:
        config = yaml.safe_load(fp)

    # Initialize model
    scoring_runner = ScoringModelRunner(config)
    scoring_runner.load_pretrained(args.model_path)
    model = scoring_runner.model
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Encoders to keep train/inference conditioning consistent.
    # (The discriminator was trained with text+image features as the condition
    # stream; running it with `lang_cond=img_cond=None` at inference is a strict
    # distribution shift and would silently change behaviour.)
    text_embedder = T5Embedder(
        from_pretrained=args.pretrained_text_encoder_name_or_path,
        model_max_length=config['dataset']['tokenizer_max_length'],
        local_files_only=False,
        device=device,
    )
    vision_encoder = SiglipVisionTower(
        vision_tower=args.pretrained_vision_encoder_name_or_path,
        args=None,
        delay_load=False,
    )
    vision_encoder.vision_tower.to(device).eval()
    
    # Load dataset
    if args.load_from == "bson":
        dataset = BsonVLADataset(
            bson_dir=args.dataset_path,
            sub_sample=1.0,
            normalize_mode="min_max",
            stats_file="v5_bson_stats/dataset_statistics.json"
        )
    elif args.load_from == "lerobot":
        dataset = LeRobotVLADataset(config=config)
    else:
        raise ValueError(f"Unsupported dataset type: {args.load_from}")
    
    # Load valid episodes for comparison
    with open("episode_quality_analysis/complete_analysis_results.json", 'r') as f:
        analysis_results = json.load(f)
    valid_episodes = set(analysis_results["filtering_thresholds"]["valid_episodes"])
    
    print(f"Loaded dataset with {len(dataset)} samples")
    print(f"Valid episodes: {len(valid_episodes)}")
    
    # Inference
    results = []
    num_samples = len(dataset) if args.num_samples == -1 else min(args.num_samples, len(dataset))
    
    with torch.no_grad():
        for i in range(0, num_samples, args.batch_size):
            batch_end = min(i + args.batch_size, num_samples)
            batch_data = []
            
            # Collect batch
            for j in range(i, batch_end):
                try:
                    data = dataset[j]
                    batch_data.append(data)
                except Exception as e:
                    print(f"Error loading sample {j}: {e}")
                    continue
            
            if not batch_data:
                continue
            
            # Prepare batch tensors. Always feed the *last* proprio frame so
            # the discriminator receives the same `state` shape it was trained
            # with ([B, state_dim]).
            states_full = torch.stack(
                [torch.tensor(data['state'], dtype=torch.float32) for data in batch_data]
            ).to(device)
            if states_full.ndim == 3:
                states = states_full[:, -1, :]
            else:
                states = states_full
            actions = torch.stack(
                [torch.tensor(data['actions'], dtype=torch.float32) for data in batch_data]
            ).to(device)
            B = actions.shape[0]

            # ---- Language conditioning ----
            instructions = []
            for data in batch_data:
                instr = ""
                if isinstance(data.get('meta'), dict):
                    instr = data['meta'].get('instruction', '')
                if not instr:
                    instr = data.get('instruction', '')
                instructions.append(instr or "")
            tok = text_embedder.tokenizer(
                instructions,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=config['dataset']['tokenizer_max_length'],
            )
            input_ids = tok['input_ids'].to(device)
            lang_attn_mask = tok['attention_mask'].to(device).bool()
            lang_embeds = text_embedder.model(
                input_ids=input_ids, attention_mask=lang_attn_mask
            )["last_hidden_state"].detach()

            # ---- Image conditioning ----
            image_tensors = []
            for data in batch_data:
                if 'images' in data and isinstance(data['images'], list) and data['images']:
                    # already preprocessed in dataset's loader
                    image_tensors.append(torch.stack(
                        [t if torch.is_tensor(t) else torch.from_numpy(t) for t in data['images']],
                        dim=0,
                    ))
                else:
                    # fall back to zeros: behaviour matches "missing camera" masking
                    H = vision_encoder.image_processor.size["height"]
                    W = vision_encoder.image_processor.size["width"]
                    image_tensors.append(torch.zeros(
                        config['common']['num_cameras'] * config['common']['img_history_size'],
                        3, H, W,
                    ))
            images = torch.stack(image_tensors, dim=0).to(device).to(next(vision_encoder.vision_tower.parameters()).dtype)
            B_, N, C_img, H_img, W_img = images.shape
            image_embeds = vision_encoder(images.reshape(-1, C_img, H_img, W_img)).detach()
            image_embeds = image_embeds.reshape(B_, -1, vision_encoder.hidden_size)

            # ---- log-pi proxy ----
            logpi_chunk = torch.zeros(B, 1, device=device)
            if args.logpi_file is not None:
                # Best-effort lookup keyed by (episode, frame); falls back to zeros.
                with open(args.logpi_file, 'r') as f_lp:
                    logpi_dict = json.load(f_lp)
                for k, data in enumerate(batch_data):
                    ep = data.get('meta', {}).get('episode_idx',
                          data.get('meta', {}).get('episode_id', None))
                    fr = data.get('meta', {}).get('step_id', None)
                    if ep is None or fr is None:
                        continue
                    ep_key = str(ep)
                    entry = logpi_dict.get(ep_key)
                    if isinstance(entry, dict):
                        v = entry.get(str(fr))
                        if v is not None:
                            logpi_chunk[k, 0] = float(v)
                    elif entry is not None:
                        logpi_chunk[k, 0] = float(entry)

            # ---- Forward pass with full conditioning ----
            scores = model(
                state=states,
                action_chunk=actions,
                logpi_chunk=logpi_chunk,
                lang_cond=lang_embeds,
                img_cond=image_embeds,
            )
            
            # Store results
            for k, data in enumerate(batch_data):
                # Extract episode ID
                episode_id = None
                if 'meta' in data and isinstance(data['meta'], dict):
                    episode_id = data['meta'].get('episode_id', None)
                
                # If no episode_id, try to extract from dataset
                if episode_id is None and hasattr(dataset, 'episodes') and (i + k) < len(dataset.episodes):
                    episode_path = str(dataset.episodes[i + k])
                    import re
                    match = re.search(r'episode_(\d+)', episode_path)
                    if match:
                        episode_id = int(match.group(1))
                
                is_expert = episode_id in valid_episodes if episode_id is not None else False
                
                result = {
                    "sample_idx": i + k,
                    "episode_id": episode_id,
                    "score": scores[k].item(),
                    "is_expert": is_expert,
                    "predicted_expert": scores[k].item() > 0.5
                }
                results.append(result)
            
            if (i + args.batch_size) % (args.batch_size * 10) == 0:
                print(f"Processed {i + args.batch_size}/{num_samples} samples")
    
    # Calculate metrics
    expert_scores = [r['score'] for r in results if r['is_expert']]
    non_expert_scores = [r['score'] for r in results if not r['is_expert']]
    
    # Accuracy metrics
    correct_predictions = sum(1 for r in results if r['is_expert'] == r['predicted_expert'])
    accuracy = correct_predictions / len(results) if results else 0.0
    
    # Expert detection metrics
    true_positives = sum(1 for r in results if r['is_expert'] and r['predicted_expert'])
    false_positives = sum(1 for r in results if not r['is_expert'] and r['predicted_expert'])
    false_negatives = sum(1 for r in results if r['is_expert'] and not r['predicted_expert'])
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Summary
    summary = {
        "total_samples": len(results),
        "expert_samples": len(expert_scores),
        "non_expert_samples": len(non_expert_scores),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "expert_score_mean": np.mean(expert_scores) if expert_scores else 0.0,
        "expert_score_std": np.std(expert_scores) if expert_scores else 0.0,
        "non_expert_score_mean": np.mean(non_expert_scores) if non_expert_scores else 0.0,
        "non_expert_score_std": np.std(non_expert_scores) if non_expert_scores else 0.0,
    }
    
    # Aggregate to per-episode scores (Dexora §III-C: K sub-clip aggregation).
    from collections import defaultdict
    per_episode_scores = defaultdict(list)
    for r in results:
        if r["episode_id"] is not None:
            per_episode_scores[r["episode_id"]].append(r["score"])
    if args.episode_aggregation == "mean":
        agg_fn = np.mean
    elif args.episode_aggregation == "median":
        agg_fn = np.median
    elif args.episode_aggregation == "min":
        agg_fn = np.min
    elif args.episode_aggregation == "max":
        agg_fn = np.max
    else:  # pragma: no cover
        raise ValueError(args.episode_aggregation)
    episode_scores = {
        int(ep): float(agg_fn(np.asarray(s, dtype=np.float64)))
        for ep, s in per_episode_scores.items()
    }
    summary["episode_aggregation"] = args.episode_aggregation
    summary["num_episodes_scored"] = len(episode_scores)

    output_data = {
        "summary": summary,
        "detailed_results": results,
        "episode_scores": episode_scores,
    }
    
    with open(args.output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print("\n=== Scoring Model Evaluation Results ===")
    print(f"Total samples: {summary['total_samples']}")
    print(f"Expert samples: {summary['expert_samples']}")
    print(f"Non-expert samples: {summary['non_expert_samples']}")
    print(f"Accuracy: {summary['accuracy']:.4f}")
    print(f"Precision: {summary['precision']:.4f}")
    print(f"Recall: {summary['recall']:.4f}")
    print(f"F1 Score: {summary['f1_score']:.4f}")
    print(f"Expert score: {summary['expert_score_mean']:.4f} ± {summary['expert_score_std']:.4f}")
    print(f"Non-expert score: {summary['non_expert_score_mean']:.4f} ± {summary['non_expert_score_std']:.4f}")
    print(f"Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
