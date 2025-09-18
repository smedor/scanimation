def view_scanimation(output_path,
                     W_px: int, H_px: int, b_px: int, s_px: int,
                     horizontal_motion=True, reverse_motion=True,
                     fps=12, cycles=2,
                     gif_name="preview.gif", mov_name="preview.mov"):
    import os, cv2, numpy as np, imageio.v2 as imageio

    inter_path = os.path.join(output_path, "interlaced.png")
    bar_path   = os.path.join(output_path, "barrier.png")
    inter = cv2.imread(inter_path, cv2.IMREAD_COLOR)
    bar   = cv2.imread(bar_path, cv2.IMREAD_UNCHANGED)  # keep alpha
    if inter is None or bar is None:
        raise FileNotFoundError("Missing interlaced.png or barrier.png in output_path.")
    if inter.shape[1] != W_px or inter.shape[0] != H_px or bar.shape[1] != W_px or bar.shape[0] != H_px:
        raise ValueError("Barrier/interlaced dimensions don’t match W_px×H_px.")

    N = int((b_px // s_px) + 1)  # since b_px=(N-1)*s_px -> N = (b/s)+1

    alpha = bar[:, :, 3] if bar.shape[2] == 4 else np.full((H_px, W_px), 255, np.uint8)
    alpha = (alpha >= 128).astype(np.uint8) * 255  # 0 slit, 255 bar
    bar_bgr = np.zeros_like(inter)  # black bars for preview

    def compose(offset):
        if horizontal_motion:
            off = (-offset if reverse_motion else offset) % W_px
            a = np.concatenate([alpha[:, off:], alpha[:, :off]], axis=1)
        else:
            off = (-offset if reverse_motion else offset) % H_px
            a = np.concatenate([alpha[off:, :], alpha[:off, :]], axis=0)
        a3 = a[..., None]
        return np.where(a3 == 0, inter, bar_bgr)

    frames = [compose(k * s_px) for k in range(N)]
    frames = frames * max(1, int(cycles))

    # GIF
    gif_path = os.path.join(output_path, gif_name)
    imageio.mimsave(gif_path, [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames],
                    duration=1.0/max(1,fps), loop=0)

    # MOV
    mov_path = os.path.join(output_path, mov_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(mov_path, fourcc, float(fps), (W_px, H_px))
    if not vw.isOpened():
        raise RuntimeError("Failed to open VideoWriter. Try 'avc1' or 'MJPG'.")
    for f in frames: vw.write(f)
    vw.release()

    print(f"[view] N={N} s_px={s_px} b_px={b_px} W%={W_px % s_px} H%={H_px % s_px}")
    print("Saved:", gif_path, mov_path)
    return gif_path, mov_path