import json
import base64
import io
from datasets import load_dataset
from PIL import Image


def compress_image(pil_img, max_side=768, quality=70):
    """Resize to max_side keeping aspect ratio, convert to JPEG q70, return base64."""
    pil_img = pil_img.convert("RGB")
    pil_img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def convert_qa_to_fireworks(output_file="results/molmoweb_qa_fireworks_1k.jsonl", max_samples=1000):
    print(f"Loading dataset: allenai/MolmoWeb-SyntheticQA (streaming) ...")
    ds = load_dataset("allenai/MolmoWeb-SyntheticQA", split="train", streaming=True)
    if max_samples:
        ds = ds.take(max_samples)

    SYSTEM_PROMPT = (
        "You are a web browsing assistant. "
        "Given a screenshot of a webpage, answer questions about its content, "
        "layout, and interactive elements."
    )

    written = 0
    skipped = 0

    with open(output_file, "w") as fout:
        for idx, sample in enumerate(ds):
            try:
                pil_img = sample["image"]
                b64_img = compress_image(pil_img)

                qa_list = sample["messages"]
                if not qa_list:
                    skipped += 1
                    continue

                messages = [
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}
                ]

                for i, qa in enumerate(qa_list):
                    question = qa.get("question", "")
                    answer = qa.get("answer", "")
                    if not question or not answer:
                        continue

                    user_content = []
                    # Only include image in the first QA turn to save tokens
                    if i == 0:
                        user_content.append(
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
                        )
                    user_content.append({"type": "text", "text": question})

                    messages.append({"role": "user", "content": user_content})
                    messages.append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": answer}]
                    })

                # Need at least system + 1 QA pair
                if len(messages) >= 3:
                    fout.write(json.dumps({"messages": messages}) + "\n")
                    written += 1
                else:
                    skipped += 1

                if (idx + 1) % 100 == 0:
                    print(f"  Processed {idx+1} | written={written}, skipped={skipped}")

            except Exception as e:
                skipped += 1
                if skipped <= 5:
                    print(f"  [skip] sample {idx}: {e}")
                continue

    print(f"\nDone. Written: {written}, Skipped: {skipped} -> {output_file}")
    return written, skipped


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert MolmoWeb-SyntheticQA to Fireworks VLM SFT format")
    parser.add_argument("--max-samples", type=int, default=1000, help="Number of samples to convert")
    parser.add_argument("--output", type=str, default="results/molmoweb_qa_fireworks.jsonl", help="Output JSONL path")
    args = parser.parse_args()

    convert_qa_to_fireworks(
        output_file=args.output,
        max_samples=args.max_samples,
    )
