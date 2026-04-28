import numpy as np
from scipy.spatial.transform import Rotation as R

m_old = 7.4554
m_new = 7.9
L_old = 0.35
L_new = 0.354

k_m = m_new / m_old
k_L = L_new / L_old

def compute_new_inertial_v2(pos_old, quat_old_wxyz, diag_old):
    pos_new = np.array([pos_old[0], pos_old[1], pos_old[2] * k_L])
    quat_xyzw = [quat_old_wxyz[1], quat_old_wxyz[2], quat_old_wxyz[3], quat_old_wxyz[0]]
    rot = R.from_quat(quat_xyzw)
    R_mat = rot.as_matrix()
    
    I_diag = np.diag(diag_old)
    I_body_com = R_mat @ I_diag @ R_mat.T
    
    J_old = (np.trace(I_body_com) / 2.0) * np.eye(3) - I_body_com
    S = np.diag([1.0, 1.0, k_L])
    J_new = k_m * (S @ J_old @ S)
    I_body_com_new = np.trace(J_new) * np.eye(3) - J_new
    
    # Evaluate new I along the original principal axes:
    # Since we are just stretching a little, the principal axes won't change much.
    # Let's project I_body_com_new onto the old principal axes:
    I_diag_new_approx = R_mat.T @ I_body_com_new @ R_mat
    
    # Get exact eigenvalues
    evals, evecs = np.linalg.eigh(I_body_com_new)
    
    # We want to match each new eigenvector with the old one
    # evecs are columns. R_mat columns are the old eigenvectors.
    idx = []
    for i in range(3):
        old_vec = R_mat[:, i]
        # find the evec that has highest dot product with old_vec
        dots = np.abs(evecs.T @ old_vec)
        best_j = np.argmax(dots)
        idx.append(best_j)
        
    diag_new = evals[idx]
    R_new_mat = evecs[:, idx]
    
    # Ensure dot products are positive to avoid 180 deg flips
    for i in range(3):
        if np.dot(R_new_mat[:, i], R_mat[:, i]) < 0:
            R_new_mat[:, i] *= -1
            
    if np.linalg.det(R_new_mat) < 0:
        # this shouldn't happen if we matched properly and didn't flip parity, but just in case
        pass # we can't just flip one column if we matched them. It must be +1 because R_mat was +1
        
    rot_new = R.from_matrix(R_new_mat)
    quat_new_xyzw = rot_new.as_quat()
    quat_new_wxyz = [quat_new_xyzw[3], quat_new_xyzw[0], quat_new_xyzw[1], quat_new_xyzw[2]]
    
    if quat_new_wxyz[0] < 0:
        quat_new_wxyz = [-x for x in quat_new_wxyz]
        
    return pos_new, quat_new_wxyz, diag_new

fl_pos_old = [-0.00418663, -0.0366068, -0.0432737]
fl_quat_old = [0.884916, 0.0880602, -0.00896728, 0.457262]
fl_diag_old = [0.085057, 0.0842843, 0.0112538]

fr_pos_old = [-0.00418663, 0.0366068, -0.0432737]
fr_quat_old = [0.457262, -0.00896728, 0.0880602, 0.884916]
fr_diag_old = [0.085057, 0.0842843, 0.0112538]

pos_fl, quat_fl, diag_fl = compute_new_inertial_v2(fl_pos_old, fl_quat_old, fl_diag_old)
pos_fr, quat_fr, diag_fr = compute_new_inertial_v2(fr_pos_old, fr_quat_old, fr_diag_old)

print("FL/RL:")
print(f'pos="{pos_fl[0]:.6g} {pos_fl[1]:.6g} {pos_fl[2]:.6g}" quat="{quat_fl[0]:.6g} {quat_fl[1]:.6g} {quat_fl[2]:.6g} {quat_fl[3]:.6g}" mass="7.9" diaginertia="{diag_fl[0]:.6g} {diag_fl[1]:.6g} {diag_fl[2]:.6g}"')

print("FR/RR:")
print(f'pos="{pos_fr[0]:.6g} {pos_fr[1]:.6g} {pos_fr[2]:.6g}" quat="{quat_fr[0]:.6g} {quat_fr[1]:.6g} {quat_fr[2]:.6g} {quat_fr[3]:.6g}" mass="7.9" diaginertia="{diag_fr[0]:.6g} {diag_fr[1]:.6g} {diag_fr[2]:.6g}"')

