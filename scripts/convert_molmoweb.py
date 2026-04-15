import json
import base64
from datasets import load_dataset

def convert_molmoweb_to_fireworks(subset="from_template", output_file="fireworks_train.jsonl", max_samples=1000):
    print(f"Loading dataset: allenai/MolmoWeb-SyntheticTrajs / {subset} ...")
    ds = load_dataset("allenai/MolmoWeb-SyntheticTrajs", subset, split="train", streaming=True)
    if max_samples:
        ds = ds.take(max_samples)
    print(f"Streaming dataset, will take up to {max_samples} samples")

    SYSTEM_PROMPT = (
        "You are an autonomous web navigation agent. "
        "Given a task and a screenshot of the current browser state, "
        "decide the next action to take. "
        "Output your reasoning and then the action."
    )

    written = 0
    skipped = 0

    with open(output_file, "w") as fout:
        for idx, sample in enumerate(ds):
            try:
                instruction = sample["instruction"]
                if isinstance(instruction, str):
                    try:
                        instruction = json.loads(instruction)
                    except Exception:
                        pass
                task_text = instruction if isinstance(instruction, str) else instruction.get("intent", str(instruction))

                trajectory = sample["trajectory"]
                if isinstance(trajectory, str):
                    trajectory = json.loads(trajectory)

                # Build image lookup: path -> base64
                image_map = {}
                for img in sample["images"]:
                    b64 = base64.b64encode(img["bytes"]).decode("utf-8")
                    image_map[img["path"]] = b64

                messages = [
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}
                ]

                steps = sorted(trajectory.keys(), key=lambda k: int(k))
                valid_steps = 0
                for i, step_id in enumerate(steps):
                    step = trajectory[step_id]
                    screenshot_name = step.get("screenshot")
                    action = step.get("action", {})
                    action_str = action.get("action_str", "")
                    action_desc = action.get("action_description", "")

                    if not screenshot_name or screenshot_name not in image_map:
                        continue

                    b64_img = image_map[screenshot_name]

                    user_content = [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}},
                    ]
                    if i == 0:
                        user_content.append({"type": "text", "text": f"Task: {task_text}"})
                    else:
                        user_content.append({"type": "text", "text": "What is the next action?"})

                    messages.append({"role": "user", "content": user_content})

                    assistant_text = f"{action_desc}\n{action_str}".strip() if action_desc else action_str
                    messages.append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": assistant_text}]
                    })
                    valid_steps += 1

                if valid_steps > 0:
                    fout.write(json.dumps({"messages": messages}) + "\n")
                    written += 1
                else:
                    skipped += 1

                if (idx + 1) % 100 == 0:
                    print(f"  Processed {idx+1} | written={written}, skipped={skipped}")

            except Exception as e:
                skipped += 1
                if skipped <= 3:
                    print(f"  [skip] sample {idx}: {e}")
                continue

    print(f"\nDone. Written: {written}, Skipped: {skipped} -> {output_file}")
    return written, skipped

if __name__ == "__main__":
    convert_molmoweb_to_fireworks(
        subset="from_template",
        output_file="results/molmoweb_fireworks_1k.jsonl",
        max_samples=1000
    )
