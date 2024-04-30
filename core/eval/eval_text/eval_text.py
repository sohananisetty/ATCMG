import torch
import math
from core.eval.eval_text.helpers import (
    calculate_activation_statistics,
    calculate_diversity,
    calculate_R_precision,
    calculate_frechet_distance,
)
from core.models.generation.muse2 import generate_animation
from core.datasets.base_dataset import BaseMotionDataset
from core import MotionRep
from tqdm import tqdm

base_dset = BaseMotionDataset(motion_rep=MotionRep.BODY, hml_rep="gpvc")


def get_latents(inputs, conditions, tmr, normalize=True):
    text_conds = conditions["text"]
    text_x_dict = {"x": text_conds[0], "mask": text_conds[1].to(torch.bool)}
    motion_x_dict = {"x": inputs[0], "mask": inputs[1].to(torch.bool)}
    motion_mask = motion_x_dict["mask"]
    text_mask = text_x_dict["mask"]
    t_latents = tmr.encode(text_x_dict, sample_mean=True)

    # motion -> motion
    m_latents = tmr.encode(motion_x_dict, sample_mean=True)

    if normalize:
        t_latents = torch.nn.functional.normalize(t_latents, dim=-1)
        m_latents = torch.nn.functional.normalize(m_latents, dim=-1)

    return t_latents, m_latents


@torch.no_grad()
def evaluation_vqvae(
    val_loader,
    motion_vqvae,
    tmr,
    normalize=True,
    eval_bs=256,
):
    motion_vqvae.eval()
    tmr.eval()
    nb_sample = 0
    # base_dset = BaseMotionDataset(motion_rep=MotionRep.BODY, hml_rep="gpvc")

    nb_sample = 0
    motion_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    for inputs, conditions in tqdm(val_loader):
        torch.cuda.empty_cache()

        with torch.no_grad():
            bs = inputs["motion"][0].shape[0]

            if inputs["motion"][0].shape[-1] != tmr.motion_encoder.nfeats:
                input_eval = torch.zeros(
                    inputs["motion"][0].shape[:1] + (tmr.motion_encoder.nfeats,)
                ).to(inputs["motion"][0].device)

                for i in range(bs):
                    lenn = inputs["motion"][1].sum(-1)
                    body_M = base_dset.toMotion(
                        inputs["motion"][i, :lenn],
                        motion_rep=MotionRep("body"),
                        hml_rep=val_loader.dataset.datasets[0].hml_rep,
                    )
                    input_eval[i, :lenn] = body_M()
                t_latents, m_latents = get_latents(
                    input_eval, conditions, tmr, normalize=normalize
                )

            else:

                t_latents, m_latents = get_latents(
                    inputs["motion"], conditions, tmr, normalize=normalize
                )

            pred_pose_eval = torch.zeros(
                inputs["motion"][0].shape[:1] + (tmr.motion_encoder.nfeats,)
            ).to(inputs["motion"][0].device)

            # for k in range(bs):
            #     lenn = int(inputs["lens"][k])
            #     vqvae_output = motion_vqvae(
            #         motion=inputs["motion"][0][k : k + 1, :lenn],
            #     )
            #     pred_pose_eval[k : k + 1, :lenn] = vqvae_output.decoded_motion
            decoded_motion = (
                motion_vqvae(inputs["motion"][0]).decoded_motion
                * inputs["motion"][1][..., None]
            )

            ### need to assert tmr hml_rep

            if decoded_motion.shape[-1] != tmr.motion_encoder.nfeats:

                for i in range(bs):
                    lenn = inputs["motion"][1].sum(-1)
                    body_M = base_dset.toMotion(
                        decoded_motion[i, :lenn],
                        motion_rep=MotionRep("body"),
                        hml_rep=val_loader.dataset.datasets[0].hml_rep,
                    )
                    pred_pose_eval[i, :lenn] = body_M()

            else:
                pred_pose_eval = decoded_motion

            t_latents_pred, m_latents_pred = get_latents(
                (pred_pose_eval, inputs["motion"][1]),
                conditions,
                tmr,
                normalize=normalize,
            )

            motion_list.append(m_latents)
            motion_pred_list.append(m_latents_pred)

            temp_R, temp_match = calculate_R_precision(
                t_latents.cpu().numpy(), m_latents.cpu().numpy(), top_k=3, sum_all=True
            )
            R_precision_real += temp_R
            matching_score_real += temp_match
            temp_R, temp_match = calculate_R_precision(
                t_latents_pred.cpu().numpy(),
                m_latents_pred.cpu().numpy(),
                top_k=3,
                sum_all=True,
            )
            R_precision += temp_R
            matching_score_pred += temp_match
            nb_sample += bs

    motion_annotation_np = torch.cat(motion_list).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list).cpu().numpy()
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(
        motion_annotation_np, 300 if nb_sample > 300 else 100
    )
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"-->  FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    print(msg)

    real_metrics = (0.0, diversity_real, R_precision_real, matching_score_real)
    pred_metrics = (fid, diversity, R_precision, matching_score_pred)

    return real_metrics, pred_metrics


@torch.no_grad()
def evaluation_transformer(
    val_loader, condition_provider, bkn_to_motion, motion_generator, tmr, normalize=True
):
    motion_generator.eval()
    tmr.eval()
    nb_sample = 0

    motion_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    nb_sample = 0
    for inputs, conditions in val_loader:

        with torch.no_grad():
            print(inputs["motion"][0].shape, tmr.motion_encoder.nfeats)
            bs = inputs["motion"][0].shape[0]

            if inputs["motion"][0].shape[-1] != tmr.motion_encoder.nfeats:
                input_eval = torch.zeros(
                    inputs["motion"][0].shape[:1] + (tmr.motion_encoder.nfeats,)
                ).to(inputs["motion"][0].device)

                for i in range(bs):
                    lenn = inputs["motion"][1].sum(-1)
                    body_M = base_dset.toMotion(
                        inputs["motion"][i, :lenn],
                        motion_rep=MotionRep("body"),
                        hml_rep=val_loader.dataset.datasets[0].hml_rep,
                    )
                    input_eval[i, :lenn] = body_M()
                t_latents, m_latents = get_latents(
                    input_eval, conditions, tmr, normalize=normalize
                )

            else:

                t_latents, m_latents = get_latents(
                    inputs["motion"], conditions, tmr, normalize=normalize
                )

            print(t_latents.shape, m_latents.shape)

            pred_pose_eval = torch.zeros(
                inputs["motion"][0].shape[:-1] + (tmr.motion_encoder.nfeats,)
            ).to(inputs["motion"][0].device)

            print(pred_pose_eval.shape)

            for k in range(bs):
                lenn = int(inputs["motion"][1][k].sum())
                text_ = inputs["texts"][k]
                duration_s = math.ceil(lenn / 30)
                all_ids = generate_animation(
                    motion_generator,
                    condition_provider,
                    temperature=0.4,
                    overlap=10,
                    duration_s=duration_s,
                    text=text_,
                    aud_file=(conditions["audio_files"][k]),
                    use_token_critic=True,
                    timesteps=8,
                )
                gen_motion = bkn_to_motion(all_ids[:, :1])()[:lenn][None]

                print(gen_motion.shape)

                pred_pose_eval[k : k + 1, :lenn] = gen_motion

            t_latents_pred, m_latents_pred = get_latents(
                (pred_pose_eval, inputs["motion"][1]), conditions, tmr, normalize
            )

            motion_list.append(m_latents)
            motion_pred_list.append(m_latents_pred)

            temp_R, temp_match = calculate_R_precision(
                t_latents.cpu().numpy(), m_latents.cpu().numpy(), top_k=3, sum_all=True
            )
            R_precision_real += temp_R
            matching_score_real += temp_match
            temp_R, temp_match = calculate_R_precision(
                t_latents_pred.cpu().numpy(),
                m_latents_pred.cpu().numpy(),
                top_k=3,
                sum_all=True,
            )
            R_precision += temp_R
            matching_score_pred += temp_match
            nb_sample += bs

    motion_annotation_np = torch.cat(motion_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()

    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(
        motion_annotation_np, 300 if nb_sample > 300 else 8
    )
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 8)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)  ## assert scipy 1.11.1

    msg = f"-->  FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    print(msg)

    real_metrics = (0.0, diversity_real, R_precision_real, matching_score_real)
    pred_metrics = (fid, diversity, R_precision, matching_score_pred)

    return real_metrics, pred_metrics
