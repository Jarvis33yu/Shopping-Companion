import sys

import ujson as json

rollout_file = sys.argv[1]

question_ids = set()
lines = []
with open(rollout_file, "r") as fin:
    for line in fin:
        item = json.loads(line.strip())
        if not item:
            continue
        question_id = item[0]["extra_info"]["question_id"]
        if not question_id:
            continue
        if question_id in question_ids:
            continue

        last_content = item[-1]["reward_model"]["ground_truth"]
        if not last_content:
            continue
        if last_content.count("<answer>") != 1 or last_content.count("</answer>") != 1:
            continue

        question_ids.add(question_id)
        lines.append(line)

with open(rollout_file, "w") as fout:
    for line in lines:
        fout.write(line)
