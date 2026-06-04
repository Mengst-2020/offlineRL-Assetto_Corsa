import os
import time
import numpy as np
import torch
import deepdish as dd

from my_image import (
    VisionStudentPolicy,
    VisionPolicyFastRunner,
    preprocess_image_bgr,
)

# ========= 配置区 =========
DATA_PATH = "mydata/monza_image/sac-v1/data/"   # 改成你的数据路径
HDF5_NAME = "main_data.hdf5"
EPISODE_KEY = "episode_0"                       # 可改 episode_1 / episode_2...
MODEL_PATH = "model_image/vision_student_policy.pth"        # 你的模型权重路径
HISTORY_LEN = 32

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 测试多少步（从 episode 开头开始取）
N_STEPS = 2000

# 是否对比慢版（每一步重算 32 帧 CNN）
COMPARE_SLOW = True

# warmup 次数（避免首次调用开销影响统计）
WARMUP_STEPS = 50


def cuda_sync_if_needed(device: str):
    if "cuda" in str(device) and torch.cuda.is_available():
        torch.cuda.synchronize()


def percentile_ms(times_s, ps=(50, 90, 95, 99)):
    arr = np.array(times_s) * 1000.0
    return {f"p{p}": float(np.percentile(arr, p)) for p in ps}


@torch.no_grad()
def slow_infer_one_step(model: VisionStudentPolicy, images_bgr_seq, device: str):
    """
    慢版：每个 step 重新构造 (1,H,3,240,320) 并跑完整 forward（相当于你原 act 的核心成本）
    images_bgr_seq: list[np.ndarray] length=H, each is BGR uint8 (240,320,3)
    """
    imgs = [preprocess_image_bgr(im) for im in images_bgr_seq]  # each (3,240,320)
    imgs = np.stack(imgs, axis=0)                               # (H,3,240,320)
    x = torch.from_numpy(imgs).unsqueeze(0).to(device)          # (1,H,3,240,320)
    out = model(x)[0]                                           # (3,)
    out = out.cpu().numpy()
    return out


def main():
    print(f"[Bench] DEVICE={DEVICE}")

    # ---- 1) 读取一个 episode 的图片序列 ----
    h5_path = os.path.join(DATA_PATH, HDF5_NAME)
    dataset_all = dd.io.load(h5_path)
    ep = dataset_all[EPISODE_KEY]
    images = ep["observations"]["image"]    # shape: (T,240,320,3) BGR uint8
    terms = ep["terminations"]
    T = images.shape[0]
    print(f"[Bench] Loaded {EPISODE_KEY} with T={T}")

    # ---- 2) 加载模型 ----
    model = VisionStudentPolicy(history_len=HISTORY_LEN).to(DEVICE)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()

    runner = VisionPolicyFastRunner(model=model, history_len=HISTORY_LEN, device=DEVICE)

    # ---- 3) 冷启动：用第一帧 reset（填满 embedding buffer）----
    first_img = images[0]
    runner.reset(first_img)

    # ---- 4) warmup（不计时）----
    print(f"[Bench] Warmup {WARMUP_STEPS} steps...")
    for i in range(min(WARMUP_STEPS, T - 1)):
        _ = runner.act(images[i])

    cuda_sync_if_needed(DEVICE)

    # ---- 5) 正式计时：fast runner ----
    times_fast = []
    steps = 0

    print(f"[Bench] Timing FAST runner for up to {N_STEPS} steps...")
    for t in range(0, min(N_STEPS, T)):
        if terms[t]:
            break

        cuda_sync_if_needed(DEVICE)
        t0 = time.perf_counter()

        _ = runner.act(images[t])

        cuda_sync_if_needed(DEVICE)
        t1 = time.perf_counter()

        times_fast.append(t1 - t0)
        steps += 1

    avg_ms = (np.mean(times_fast) * 1000.0) if steps > 0 else float("nan")
    fps = (steps / np.sum(times_fast)) if steps > 0 else 0.0
    print(f"[FAST] steps={steps}  avg={avg_ms:.3f} ms  fps={fps:.2f}")
    print(f"[FAST] percentiles(ms) = {percentile_ms(times_fast)}")

    # ---- 6) 可选：慢版对比（每步重算 H 帧 CNN）----
    if COMPARE_SLOW:
        print(f"[Bench] Timing SLOW baseline (recompute {HISTORY_LEN} CNN embeddings each step)...")
        # 准备一个 image_buffer（list）作为历史序列
        # 冷启动：用第0帧填满 H
        image_buffer = [images[0] for _ in range(HISTORY_LEN)]

        # warmup
        for i in range(min(WARMUP_STEPS, T - 1)):
            image_buffer.pop(0)
            image_buffer.append(images[i])
            _ = slow_infer_one_step(model, image_buffer, DEVICE)

        cuda_sync_if_needed(DEVICE)

        times_slow = []
        steps2 = 0
        for t in range(0, min(N_STEPS, T)):
            if terms[t]:
                break

            # 更新历史窗口（最后一帧为当前 images[t]）
            image_buffer.pop(0)
            image_buffer.append(images[t])

            cuda_sync_if_needed(DEVICE)
            t0 = time.perf_counter()

            _ = slow_infer_one_step(model, image_buffer, DEVICE)

            cuda_sync_if_needed(DEVICE)
            t1 = time.perf_counter()

            times_slow.append(t1 - t0)
            steps2 += 1

        avg_ms2 = (np.mean(times_slow) * 1000.0) if steps2 > 0 else float("nan")
        fps2 = (steps2 / np.sum(times_slow)) if steps2 > 0 else 0.0
        print(f"[SLOW] steps={steps2}  avg={avg_ms2:.3f} ms  fps={fps2:.2f}")
        print(f"[SLOW] percentiles(ms) = {percentile_ms(times_slow)}")

        if steps > 0 and steps2 > 0:
            speedup = np.mean(times_slow) / np.mean(times_fast)
            print(f"[Speedup] SLOW/FAST = {speedup:.1f}x")

    print("[Bench] Done.")


if __name__ == "__main__":
    main()
