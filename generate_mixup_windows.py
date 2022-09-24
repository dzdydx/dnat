import json
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser

def generate_mixup_windows(train_strong_json_path, num_events_in_window=1, window_length_threshold=0.5):
    print(f"Generating mixup windows with {num_events_in_window} event(s) inside each window.")

    def get_available_window(mix_item):
        end_points = {}
        for label in mix_item["labels"]:
            _, start_time, end_time = label.values()
            end_points.setdefault(start_time, list())
            end_points.setdefault(end_time, list())

        for label in mix_item["labels"]:
            event, start_time, end_time = label.values()
            for k in end_points.keys():
                if start_time <= k and k < end_time:
                    end_points[k].append(event)

        end_points_keys = list(end_points.keys())
        end_points_keys.sort()

        windows = list(filter(lambda x: len(x[1]) == num_events_in_window, end_points.items()))
        avail_windows = []
        exist_class = set()
        for i, window in enumerate(windows):
            start_time, events = window
            exist_class = exist_class | set(events)
            idx = end_points_keys.index(start_time)
            window_end_time = end_points_keys[idx + 1]
            if window_end_time - start_time < window_length_threshold:
                windows.pop(i)
                continue
            else:
                avail_windows.append({
                    "start_time": start_time,
                    "end_time": window_end_time,
                    "events": events
                })

        if len(avail_windows) == 0:
            return None, None

        return avail_windows, exist_class

    
    with open(train_strong_json_path, "r") as fp:
        train_strong_dict = json.load(fp)
    train_strong_data = train_strong_dict["data"]

    mixup_windows = {}
    used_classes = set()
    for mix_sample in tqdm(train_strong_data):
        avail_windows, exist_classes = get_available_window(mix_sample)
        if avail_windows != None:
            used_classes = used_classes | exist_classes
            mixup_windows.setdefault(
                mix_sample["filepath"],
                avail_windows)
    
    p1 = Path(train_strong_json_path).parent / "mixup_windows.json"
    with open(p1, "w") as fp:
        json.dump(mixup_windows, fp)
    
    p2 = Path(train_strong_json_path).parent / "used_classes.json"
    with open(p2, "w") as fp:
        json.dump(list(used_classes), fp)
    
    print(f"{len(mixup_windows.keys())} available windows generated in {p1}")


if __name__ == '__main__':
    parser = ArgumentParser()
    # Basic Training Control
    parser.add_argument('train_strong_data_json', type=str)
    parser.add_argument('--num_events_in_window', default=1, type=int)
    parser.add_argument('--window_length_threshold', default=0.5, type=float)

    args = parser.parse_args()
    generate_mixup_windows(args.train_strong_data_json,
                            num_events_in_window=args.num_events_in_window,
                            window_length_threshold=args.window_length_threshold)