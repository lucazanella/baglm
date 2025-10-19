import math


def read_task_info(path):
    titles = {}
    urls = {}
    n_steps = {}
    steps = {}
    with open(path) as f:
        idx = f.readline()
        while idx != "":
            idx = idx.strip()
            titles[idx] = f.readline().strip()
            urls[idx] = f.readline().strip()
            n_steps[idx] = int(f.readline().strip())
            steps[idx] = f.readline().strip().split(",")
            next(f)
            idx = f.readline()
    return {"title": titles, "url": urls, "n_steps": n_steps, "steps": steps}


def get_vids(path):
    task_vids = {}
    with open(path) as f:
        for line in f:
            task, vid, url = line.strip().split(",")
            if task not in task_vids:
                task_vids[task] = []
            task_vids[task].append(vid)
    return task_vids


def get_language_queries(path, task_steps):
    language_queries = []
    with open(path) as f:
        for line in f:
            step, start, end = line.strip().split(",")
            start = int(math.floor(float(start)))
            end = int(math.ceil(float(end)))
            step = int(step) - 1
            step_description = task_steps[step]
            query = {
                "clip_start_sec": start,
                "clip_end_sec": end,
                "query": step_description,
                "step": step,
            }
            language_queries.append(query)
    return language_queries


def filter_nested_dict_by_video_uids(Y, allowed_video_uids):
    Y_filtered = {}

    for task, task_dict in Y.items():
        filtered_task_dict = {
            video_uid: pred
            for video_uid, pred in task_dict.items()
            if video_uid in allowed_video_uids
        }
        if filtered_task_dict:
            Y_filtered[task] = filtered_task_dict

    return Y_filtered
