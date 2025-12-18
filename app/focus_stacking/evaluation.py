"""Evaluation metrics for focus stacking (Q_AB/F)."""

import numpy as np
import cv2

def compute_sobel_gradients(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute Sobel edge strength and orientation.
    
    Args:
        img: Input image (H, W) or (H, W, C).
        
    Returns:
        g: Edge strength (H, W).
        alpha: Edge orientation in radians (H, W).
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    gray = gray.astype(np.float32)
    
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    
    g = np.sqrt(gx**2 + gy**2)
    alpha = np.arctan2(gy, gx) # -pi to pi
    
    return g, alpha

def compute_Q_preservation(g_src: np.ndarray, a_src: np.ndarray, g_fus: np.ndarray, a_fus: np.ndarray) -> np.ndarray:
    """Compute edge preservation measure Q(src, fus)."""
    # Constants from Xydeas & Petrovic (2000)
    Tg, kg, Dg = 0.9994, -15.0, 0.5
    Ta, ka, Da = 0.9879, -22.0, 0.8
    
    # --- Edge strength preservation (Qg) ---
    # Ratio g_fus/g_src or g_src/g_fus, whichever is <= 1
    g_ratio = np.zeros_like(g_src)
    
    # Avoid div by zero
    eps = 1e-6
    g_src_safe = g_src.copy()
    g_src_safe[g_src_safe < eps] = eps
    g_fus_safe = g_fus.copy()
    g_fus_safe[g_fus_safe < eps] = eps
    
    mask = g_src > g_fus
    g_ratio[mask] = g_fus[mask] / g_src_safe[mask]
    g_ratio[~mask] = g_src[~mask] / g_fus_safe[~mask]
    
    Qg = Tg / (1 + np.exp(kg * (g_ratio - Dg)))
    
    # --- Orientation preservation (Qa) ---
    # Calculate difference in orientation (mod pi)
    diff = np.abs(a_src - a_fus)
    
    # Wrap to [0, pi]
    diff = np.where(diff > np.pi, 2*np.pi - diff, diff)
    # Wrap to [0, pi/2] (since edge orientation is symmetric)
    diff = np.where(diff > np.pi/2, np.pi - diff, diff)
    
    # Normalize to [0, 1] where 1 is perfect match
    A_measure = 1.0 - diff / (np.pi / 2.0)
    
    Qa = Ta / (1 + np.exp(ka * (A_measure - Da)))
    
    return Qg * Qa

def compute_q_abf(fused_img: np.ndarray, source_images: list[np.ndarray], sharpness_maps: list[np.ndarray]) -> float:
    """Compute Q_AB/F metric for multi-focus fusion.
    
    For N > 2 images, we select the top-2 sources per pixel based on sharpness maps.
    
    Args:
        fused_img: Fused result (H, W, 3) or (H, W).
        source_images: List of source images.
        sharpness_maps: List of sharpness maps (finest level) corresponding to sources.
        
    Returns:
        Scalar Q_AB/F score (0.0 to 1.0).
    """
    if not source_images:
        return 0.0
        
    # 1. Compute gradients for Fused image
    g_F, a_F = compute_sobel_gradients(fused_img)
    
    # 2. Compute gradients for all Source images
    g_srcs = []
    a_srcs = []
    for src in source_images:
        g, a = compute_sobel_gradients(src)
        g_srcs.append(g)
        a_srcs.append(a)
        
    g_srcs = np.stack(g_srcs, axis=0) # (N, H, W)
    a_srcs = np.stack(a_srcs, axis=0) # (N, H, W)
    
    # 3. Determine Top-2 sources per pixel
    
    # Handle multi-channel sharpness maps (reduce to single channel)
    processed_sharpness = []
    for s_map in sharpness_maps:
        s_curr = s_map
        if s_curr.ndim == 3:
            # Use mean across channels to get a single sharpness value per pixel
            s_curr = np.mean(s_curr, axis=2)
        processed_sharpness.append(s_curr)

    # Ensure sharpness maps match image size (they should if they are level 0)
    # Resize if necessary before stacking to avoid mismatch errors
    target_h, target_w = g_F.shape
    final_sharpness = []
    for s_map in processed_sharpness:
        if s_map.shape != (target_h, target_w):
             final_sharpness.append(cv2.resize(s_map, (target_w, target_h)))
        else:
             final_sharpness.append(s_map)
             
    s_stack = np.stack(final_sharpness, axis=0) # (N, H, W)

    # Get indices of top 2 sharpness values
    # argsort sorts ascending, so take last 2
    indices = np.argsort(s_stack, axis=0) # (N, H, W)
    
    # If we have < 2 images, handle gracefully
    if len(source_images) < 2:
        # Fallback: just use the single image as both A and B (score will be 1.0 if identical)
        idx1 = np.zeros_like(g_F, dtype=int)
        idx2 = np.zeros_like(g_F, dtype=int)
    else:
        idx1 = indices[-1] # Best
        idx2 = indices[-2] # Second best
    
    # 4. Extract A and B gradients using advanced indexing
    idx1_exp = idx1[np.newaxis, :, :]
    idx2_exp = idx2[np.newaxis, :, :]
    
    g_A = np.take_along_axis(g_srcs, idx1_exp, axis=0).squeeze(0)
    a_A = np.take_along_axis(a_srcs, idx1_exp, axis=0).squeeze(0)
    
    g_B = np.take_along_axis(g_srcs, idx2_exp, axis=0).squeeze(0)
    a_B = np.take_along_axis(a_srcs, idx2_exp, axis=0).squeeze(0)
    
    # 5. Compute Q_AF and Q_BF
    Q_AF = compute_Q_preservation(g_A, a_A, g_F, a_F)
    Q_BF = compute_Q_preservation(g_B, a_B, g_F, a_F)
    
    # 6. Weighted sum (Global score)
    # Weight is the edge strength
    numerator = Q_AF * g_A + Q_BF * g_B
    denominator = g_A + g_B
    
    total_num = np.sum(numerator)
    total_den = np.sum(denominator)
    
    if total_den == 0:
        return 0.0
        
    return float(total_num / total_den)
