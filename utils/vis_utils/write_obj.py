import argparse
import os.path as osp
from pathlib import Path

import numpy as np
import pyrender
import torch
import trimesh
from core.models.smpl.body_model import BodyModel
from tqdm.auto import tqdm, trange


def main(
    model_folder,
    motion_file,
    output_folder,
    model_type="smplh",
    gender="neutral",
    num_betas=10,
):
    output_folder = Path(output_folder)
    assert output_folder.exists()

    # open motion file
    motion = np.load(motion_file, allow_pickle=True)
    for k, v in motion.items():
        if type(v) is float:
            print(k, v)
        else:
            print(k, v.shape)

    if "betas" in motion:
        betas = motion["betas"]
    else:
        betas = np.zeros((num_betas,))
    num_betas = len(betas)
    # don't know where this is documented but it's from this part of amass
    # https://github.com/nghorbani/amass/blob/master/src/amass/data/prepare_data.py#L39-L40
    # gdr2num = {'male':-1, 'neutral':0, 'female':1}
    # gdr2num_rev = {v:k for k,v in gdr2num.items()}
    if "gender" in motion:
        gender = str(motion["gender"])
    else:
        gender = gender

    print(gender)

    print(num_betas)

    model = BodyModel(
        model_folder,
        model_type=model_type,
        num_betas=num_betas,
    )

    betas, expression = torch.tensor(betas).float(), None
    betas = betas.unsqueeze(0)[:, : model.num_betas]
    if "poses" in motion:
        poses = torch.tensor(motion["poses"]).float()

    global_orient = poses[:, :3]
    if model_type == "smplh":
        body_pose = poses[:, 3:66]
        left_hand_pose = poses[:, 66:111]
        right_hand_pose = poses[:, 111:156]
    else:
        n = poses.shape[0]
        body_pose = poses[:, 3:]
        left_hand_pose = np.zeros((n, 3))
        right_hand_pose = np.zeros((n, 3))

    for pose_idx in trange(body_pose.size(0)):
        pose_idx = [pose_idx]
        # output = model(betas=betas, # expression=expression,
        #                return_verts=True)
        output = model(
            betas=betas,
            global_orient=global_orient[pose_idx],
            body_pose=body_pose[pose_idx],
            left_hand_pose=left_hand_pose[pose_idx],
            right_hand_pose=right_hand_pose[pose_idx],
            # expression=expression,
            return_verts=True,
        )
        vertices = output.v.detach().cpu().numpy().squeeze()
        joints = output.Jtr.detach().cpu().numpy().squeeze()

        vertex_colors = np.ones([vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
        # process=False to avoid creating a new mesh
        tri_mesh = trimesh.Trimesh(
            vertices, model.f, vertex_colors=vertex_colors, process=False
        )

        output_path = output_folder / "{0:04d}.obj".format(pose_idx[0])
        tri_mesh.export(str(output_path))

        # if pose_idx[0] == 0:
        #     print("displaying first pose, exit window to continue processing")
        #     mesh = pyrender.Mesh.from_trimesh(tri_mesh)

        #     scene = pyrender.Scene()
        #     scene.add(mesh)

        #     if plot_joints:
        #         sm = trimesh.creation.uv_sphere(radius=0.005)
        #         sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
        #         tfs = np.tile(np.eye(4), (len(joints), 1, 1))
        #         tfs[:, :3, 3] = joints
        #         joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
        #         scene.add(joints_pcl)

        # pyrender.Viewer(scene, use_raymond_lighting=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMPL-X Demo")

    parser.add_argument(
        "--model-folder", required=True, type=str, help="The path to the model folder"
    )
    parser.add_argument(
        "--motion-file",
        required=True,
        type=str,
        help="The path to the motion file to process",
    )
    parser.add_argument(
        "--output-folder", required=True, type=str, help="The path to the output folder"
    )
    parser.add_argument(
        "--model-type",
        default="smplh",
        type=str,
        choices=["smpl", "smplh", "smplx", "mano", "flame"],
        help="The type of model to load",
    )
    parser.add_argument(
        "--num-expression-coeffs",
        default=10,
        type=int,
        dest="num_expression_coeffs",
        help="Number of expression coefficients.",
    )
    parser.add_argument(
        "--ext", type=str, default="npz", help="Which extension to use for loading"
    )
    parser.add_argument(
        "--sample-expression",
        default=True,
        dest="sample_expression",
        type=lambda arg: arg.lower() in ["true", "1"],
        help="Sample a random expression",
    )
    parser.add_argument(
        "--use-face-contour",
        default=False,
        type=lambda arg: arg.lower() in ["true", "1"],
        help="Compute the contour of the face",
    )

    args = parser.parse_args()

    def resolve(path):
        return osp.expanduser(osp.expandvars(path))

    model_folder = resolve(args.model_folder)
    motion_file = resolve(args.motion_file)
    output_folder = resolve(args.output_folder)
    model_type = args.model_type
    ext = args.ext
    num_expression_coeffs = args.num_expression_coeffs
    sample_expression = args.sample_expression

    main(
        model_folder,
        motion_file,
        output_folder,
        model_type,
        ext=ext,
        sample_expression=sample_expression,
        use_face_contour=args.use_face_contour,
    )
