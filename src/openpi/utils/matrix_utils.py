import numpy as onp
from scipy.spatial.transform import Rotation


def gram_schmidt(vectors: onp.ndarray) -> onp.ndarray:
    """
    Apply Gram-Schmidt process to a set of vectors
    vectors are indexed by rows

    vectors: batchsize, N, D

    return: batchsize, N, D
    """
    if len(vectors.shape) == 2:
        vectors = vectors[None]

    basis = onp.zeros_like(vectors)
    basis[:, 0] = vectors[:, 0] / onp.linalg.norm(vectors[:, 0], axis=-1, keepdims=True)
    for i in range(1, vectors.shape[1]):
        v = vectors[:, i]
        for j in range(i):
            v -= onp.sum(v * basis[:, j], axis=-1, keepdims=True) * basis[:, j]
        basis[:, i] = v / onp.linalg.norm(v, axis=-1, keepdims=True)
    return basis


def rot_6d_to_rot_mat(rot_6d: onp.ndarray) -> onp.ndarray:
    """
    Convert a 6d representation to rotation matrix
    rot_6d: N, 6

    return: N, 3, 3
    """
    rot_6d = rot_6d.reshape(-1, 2, 3)
    # assert the first two vectors are orthogonal
    if not onp.allclose(onp.sum(rot_6d[:, 0] * rot_6d[:, 1], axis=-1), 0):
        rot_6d = gram_schmidt(rot_6d)

    rot_mat = onp.zeros((rot_6d.shape[0], 3, 3))
    rot_mat[:, :2, :] = rot_6d
    rot_mat[:, 2, :] = onp.cross(rot_6d[:, 0], rot_6d[:, 1])
    return rot_mat


def rot_mat_to_rot_6d(rot_mat: onp.ndarray) -> onp.ndarray:
    """
    Convert a rotation matrix to 6d representation
    rot_mat: N, 3, 3

    return: N, 6
    """
    rot_6d = rot_mat[:, :2, :]  # N, 2, 3
    return rot_6d.reshape(-1, 6)  # N, 6


def quat_to_rot_6d(quat: onp.ndarray, scalar_first: bool = True) -> onp.ndarray:
    """
    Convert a quaternion to 6d representation
    quat: N, 4
    scalar_first: bool # If True, the quaternion is in the format [w, x, y, z], otherwise [x, y, z, w]
    """
    rot_mat = Rotation.from_quat(quat, scalar_first=scalar_first).as_matrix()
    return rot_mat_to_rot_6d(rot_mat)


def convert_abs_action(action, proprio):
    """
    Calculate the next state from the delta action and the current proprioception
    action: S, T, action_dim
    proprio: S, T, proprio_dim
    """
    delta_trans = action[:, :, :3].reshape(-1, 3)
    delta_rot = action[:, :, 3:9].reshape(-1, 6)
    delta_rot = Rotation.from_matrix(rot_6d_to_rot_mat(delta_rot))

    current_state = onp.repeat(proprio[:, 0:1], action.shape[1], 1)
    current_trans = current_state[:, :, :3].reshape(-1, 3)
    current_rot = Rotation.from_matrix(rot_6d_to_rot_mat(current_state[:, :, 3:9].reshape(-1, 6)))

    trans = onp.einsum("ijk,ik->ij", current_rot.as_matrix(), delta_trans) + current_trans
    rot = (current_rot * delta_rot).as_matrix()

    rot = rot_mat_to_rot_6d(rot).reshape(-1, action.shape[1], 6)
    trans = trans.reshape(-1, action.shape[1], 3)

    if action.shape[-1] == proprio.shape[-1]:
        # no eos
        desired_mat = onp.concatenate([trans, rot, action[:, :, -1:]], axis=-1)
    else:
        # with eos
        desired_mat = onp.concatenate([trans, rot, action[:, :, -2:]], axis=-1)
    return desired_mat


def rot_6d_to_euler(rot_6d: onp.ndarray, format="XYZ"):
    """
    Convert 6d representation to euler angles
    rot_6d: N, 6
    """
    rot_mat = rot_6d_to_rot_mat(rot_6d)
    return Rotation.from_matrix(rot_mat).as_euler(format, degrees=False)


def rot_6d_to_quat(rot_6d: onp.ndarray) -> onp.ndarray:
    """
    Convert 6d representation to quaternion
    rot_6d: N, 6
    """
    rot_mat = rot_6d_to_rot_mat(rot_6d)
    return Rotation.from_matrix(rot_mat).as_quat(scalar_first=True)


def action_10d_to_8d(action: onp.ndarray) -> onp.ndarray:
    """
    Convert a 10d action to a 8d action
    - 3d translation, 6d rotation, 1d gripper
    to - 3d translation, 4d euler angles, 1d gripper
    """
    return onp.concatenate([action[:3], rot_6d_to_quat(action[3:-1]).squeeze(), action[-1:]], axis=-1)
