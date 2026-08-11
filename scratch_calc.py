import numpy as np
from scipy.spatial.transform import Rotation as R

# Old values for FL_thigh / RL_thigh
m_old = 7.4554
m_new = 7.9

L_old = 0.35
L_new = 0.354

k_m = m_new / m_old
k_L = L_new / L_old

def compute_new_inertial(pos_old, quat_old_wxyz, diag_old):
    # pos_old is CoM in body frame
    # quat_old is w, x, y, z
    # diag_old is principal moments of inertia around CoM
    
    # 1. New CoM
    pos_new = np.array([pos_old[0], pos_old[1], pos_old[2] * k_L])
    
    # 2. Inertia tensor around old CoM in body frame
    # scipy Rotation uses x, y, z, w
    quat_xyzw = [quat_old_wxyz[1], quat_old_wxyz[2], quat_old_wxyz[3], quat_old_wxyz[0]]
    rot = R.from_quat(quat_xyzw)
    R_mat = rot.as_matrix()
    
    I_diag = np.diag(diag_old)
    # I_body_com = R * I_diag * R^T
    I_body_com = R_mat @ I_diag @ R_mat.T
    
    # To scale, we need moments \int x^2 dm, etc. around CoM.
    # I_xx = \int (y^2 + z^2) dm
    # I_yy = \int (x^2 + z^2) dm
    # I_zz = \int (x^2 + y^2) dm
    # I_xy = -\int xy dm
    # ...
    # This requires diagonalizing a slightly different matrix or solving linear equations.
    # Let J = \int r r^T dm.
    # I = tr(J)*I_3x3 - J  =>  J = (tr(I)/2)*I_3x3 - I
    
    J_old = (np.trace(I_body_com) / 2.0) * np.eye(3) - I_body_com
    
    # J_old has elements:
    # J_xx = \int x^2 dm, J_yy = \int y^2 dm, J_zz = \int z^2 dm
    # J_xy = \int xy dm
    
    # Scale integrals:
    # x and y coords don't scale (cross section constant). z scales by k_L.
    # mass scales by k_m.
    # \int x^2 dm_new = k_m * \int x^2 dm_old
    # \int z^2 dm_new = k_m * k_L^2 * \int z^2 dm_old
    # \int x z dm_new = k_m * k_L * \int x z dm_old
    
    S = np.diag([1.0, 1.0, k_L])
    J_new = k_m * (S @ J_old @ S)
    
    # Reconstruct new inertia tensor around new CoM:
    I_body_com_new = np.trace(J_new) * np.eye(3) - J_new
    
    # 3. Find new principal moments and rotation
    evals, evecs = np.linalg.eigh(I_body_com_new)
    
    # Sort eigenvalues
    idx = np.argsort(evals)
    diag_new = evals[idx]
    R_new_mat = evecs[:, idx]
    
    # Ensure rotation matrix is proper (det = 1)
    if np.linalg.det(R_new_mat) < 0:
        R_new_mat[:, -1] *= -1
        
    rot_new = R.from_matrix(R_new_mat)
    quat_new_xyzw = rot_new.as_quat()
    quat_new_wxyz = [quat_new_xyzw[3], quat_new_xyzw[0], quat_new_xyzw[1], quat_new_xyzw[2]]
    
    # Ensure scalar part is positive
    if quat_new_wxyz[0] < 0:
        quat_new_wxyz = [-x for x in quat_new_wxyz]
        
    return pos_new, quat_new_wxyz, diag_new

fl_pos_old = [-0.00418663, -0.0366068, -0.0432737]
fl_quat_old = [0.884916, 0.0880602, -0.00896728, 0.457262]
fl_diag_old = [0.085057, 0.0842843, 0.0112538]

fr_pos_old = [-0.00418663, 0.0366068, -0.0432737]
fr_quat_old = [0.457262, -0.00896728, 0.0880602, 0.884916]
fr_diag_old = [0.085057, 0.0842843, 0.0112538]

pos_fl, quat_fl, diag_fl = compute_new_inertial(fl_pos_old, fl_quat_old, fl_diag_old)
pos_fr, quat_fr, diag_fr = compute_new_inertial(fr_pos_old, fr_quat_old, fr_diag_old)

print("FL/RL:")
print(f'pos="{pos_fl[0]:.6g} {pos_fl[1]:.6g} {pos_fl[2]:.6g}" quat="{quat_fl[0]:.6g} {quat_fl[1]:.6g} {quat_fl[2]:.6g} {quat_fl[3]:.6g}" mass="7.9" diaginertia="{diag_fl[0]:.6g} {diag_fl[1]:.6g} {diag_fl[2]:.6g}"')

print("FR/RR:")
print(f'pos="{pos_fr[0]:.6g} {pos_fr[1]:.6g} {pos_fr[2]:.6g}" quat="{quat_fr[0]:.6g} {quat_fr[1]:.6g} {quat_fr[2]:.6g} {quat_fr[3]:.6g}" mass="7.9" diaginertia="{diag_fr[0]:.6g} {diag_fr[1]:.6g} {diag_fr[2]:.6g}"')

