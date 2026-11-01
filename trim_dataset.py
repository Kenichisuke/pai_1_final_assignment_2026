from lerobot.datasets.lerobot_dataset import LeRobotDataset

# 収集したDatasetから、前後の不要なデータを削除する。
# 先頭から削る秒数、どこまで残すのかを秒数で配列内に指定する。

# ===== 設定 =====
SRC = "repo_id/dataset"
DST = "repo_id/dataset_trim"

# エピソードごとの削る秒数。リストの N 番目が episode_index = N に対応します。
# 先頭から削る秒数
TRIM_START_S = [5, 3, 3, 4, 3, 3, 4, 3, 3, 3]
# TRIM_END_S: エピソードの先頭から何秒目まで残すか（それ以降を削る）
TRIM_END_S = [18, 16, 17, 18, 20, 19, 17, 17, 19, 19]
 
# 配列の要素数がエピソード数と合わないときに使う既定値（秒）
DEFAULT_START_S = 0
DEFAULT_END_S = None  # None なら最後まで残す
# ================


def get_value(values, ep, default):
    return values[ep] if ep < len(values) else default

def main():
    src = LeRobotDataset(SRC)
    fps = src.fps
    n_eps = src.num_episodes
    if len(TRIM_START_S) != n_eps or len(TRIM_END_S) != n_eps:
        print(
            f"[注意] エピソード数は {n_eps} ですが、TRIM_START_S は {len(TRIM_START_S)} 個、"
            f"TRIM_END_S は {len(TRIM_END_S)} 個です。足りない分は既定値を使います。"
        )

    skip = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
    features = {k: v for k, v in src.features.items() if k not in skip}

    dst = LeRobotDataset.create(
        DST, fps=fps, features=features,
        robot_type=src.meta.robot_type, use_videos=True,
    )

    for ep in range(src.num_episodes):
        info = src.meta.episodes[ep]
        ep_from = info["dataset_from_index"]
        ep_to = info["dataset_to_index"]
        start_s = get_value(TRIM_START_S, ep, DEFAULT_START_S)
        end_s = get_value(TRIM_END_S, ep, DEFAULT_END_S)

        s = ep_from + start_s * fps
        e = ep_to if end_s is None else min(ep_from + end_s * fps, ep_to)
        s, e = int(s), int(e)
        if s >= e:
            print(f"episode {ep}: 切りすぎでフレームが残らないためスキップします (start={start_s}, end={end_s})")
            continue

        for i in range(s, e):
            item = src[i]
            frame = {k: item[k] for k in features}
            for k, v in features.items():
                if v["dtype"] in ("image", "video"):   # CHW float -> HWC uint8
                    frame[k] = (item[k].permute(1, 2, 0) * 255).byte().numpy()
            frame["task"] = item["task"]
            dst.add_frame(frame)
        dst.save_episode()
        print(
            f"episode {ep + 1}/{n_eps} done "
            f"(start={start_s}s, end={end_s}s, frames={e - s})",
            flush=True,
        )

    if hasattr(dst, "finalize"):
        dst.finalize()

if __name__ == "__main__":
    main()
