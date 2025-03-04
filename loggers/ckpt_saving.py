import os
import torch

def save_checkpoint(model, args, best_val_map, best_val_acc, current_val_map, current_val_acc):
    # Base checkpoint directory
    ckpt_dir = f'/scratch/project_465001389/chandler_scratch/Projects/FrameMixer/checkpoints'
    
    # If ablation is enabled, create a subdirectory for ablations
    if args.ablation:
        save_dir = os.path.join(ckpt_dir, f'ablation/{args.spec_aug}')
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = ckpt_dir

    os.makedirs(ckpt_dir, exist_ok=True)

    # Construct a formatted string for additional arguments to include in the filename
    params_str = (
        f"dur-{args.target_duration}_sr-{args.sample_rate}_win-{args.window_size}_hop-{args.hop_size}_"
        f"mel-{args.mel_bins}_fmin-{args.fmin}_fmax-{args.fmax or 'none'}_"
        f"cls-{args.num_classes}_seed-{args.seed}_bs-{args.batch_size}_"
        f"epoch-{args.max_epoch}_loss-{args.loss}"
    )

    # Add ablation parameters if applicable
    if args.ablation:
        if args.spec_aug == 'specaugment':
            ablation_params = args.specaugment_params
        elif args.spec_aug == 'diffres':
            ablation_params = args.diffres_params
        elif args.spec_aug == 'specmix':
            ablation_params = args.specmix_params
        elif args.spec_aug == 'fma':
            ablation_params = args.framemixer_params
        else:
            # if using fma or none - not yet implemented
            ablation_params = "unknown"

        params_str += f"_abl-{args.spec_aug}_{ablation_params}"

    # Add audiomentations parameters if applicable
    if hasattr(args, 'audiomentations') and args.audiomentations:
        audiomentations_str = "-".join(args.audiomentations)
        params_str += f"_audioment-{audiomentations_str}"

    # Add noise toggle note if both ablation and noise are True
    if args.ablation and args.noise:
        params_str += f"_withnoise_seg-{args.noise_segment_ratio}"

    # Save the best model based on validation mAP
    if current_val_map > best_val_map:
        best_val_map = current_val_map
        print(f"Best validation map updated to {best_val_map:.4f}")
        torch.save(
            model.state_dict(),
            f'{save_dir}/{args.dataset}_{args.frontend}_{args.model_name}_{args.spec_aug}_best_map_{params_str}.pth'
        )

    # Save the best model based on validation accuracy
    if current_val_acc > best_val_acc:
        best_val_acc = current_val_acc
        print(f"Best validation acc updated to {best_val_acc:.4f}")
        torch.save(
            model.state_dict(),
            f'{save_dir}/{args.dataset}_{args.frontend}_{args.model_name}_{args.spec_aug}_best_acc_{params_str}.pth'
        )

    return best_val_map, best_val_acc
