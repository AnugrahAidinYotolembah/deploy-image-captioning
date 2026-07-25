from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch


def strip_prefix(key: str) -> str:
    prefixes = ("_orig_mod.", "module.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key


def clean_state_dict(state_dict):
    return {strip_prefix(key): value.detach().cpu() for key, value in state_dict.items()}


def extract_image_fc(image_state_dict):
    cleaned = clean_state_dict(image_state_dict)
    fc_state = {key: value for key, value in cleaned.items() if key.startswith("fc.")}
    required = {"fc.weight", "fc.bias"}
    missing = required - set(fc_state)
    if missing:
        raise KeyError(f"Missing image encoder fc keys: {sorted(missing)}")
    return fc_state


def infer_province(image_path: str) -> str:
    folder = Path(image_path).parent.name
    for prefix in ("Pakaian_Adat_", "pakaian_adat_", "Pakaian Adat ", "pakaian adat "):
        folder = folder.replace(prefix, "")
    return folder.replace("_", " ").strip()


def read_dataset_records(csv_path: Path):
    records = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"image_path", "description"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV wajib punya kolom: {sorted(missing)}")
        for row in reader:
            image_path = str(row.get("image_path", "")).strip()
            description = str(row.get("description", "")).strip()
            if not image_path or not description:
                continue
            records.append({
                "image_path": image_path,
                "basename": Path(image_path).name,
                "description": description,
                "province": row.get("province") or infer_province(image_path),
            })
    return records


def load_bert_vocab(tokenizer_name: str):
    try:
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        vocab = tokenizer.get_vocab()
        id_to_token = [None] * len(vocab)
        for token, idx in vocab.items():
            id_to_token[idx] = token
        if any(token is None for token in id_to_token):
            raise ValueError("Tokenizer vocab id tidak kontigu.")
        return id_to_token
    except Exception as error:
        raise RuntimeError(
            f"Gagal memuat tokenizer {tokenizer_name}. Pastikan transformers/cache tokenizer tersedia. Error: {error}"
        ) from error


def build_bundle(source_model: Path, dataset_csv: Path, output: Path, tokenizer_name: str, clip_model_name: str):
    print(f"Loading source checkpoint: {source_model}")
    checkpoint = torch.load(source_model, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Format checkpoint tidak dikenali: {type(checkpoint)}")

    print("Extracting deployment weights")
    image_encoder_fc = extract_image_fc(checkpoint["image_encoder"])
    caption_decoder = clean_state_dict(checkpoint["caption_decoder"])

    print(f"Reading dataset records: {dataset_csv}")
    records = read_dataset_records(dataset_csv)
    basename_counts = Counter(row["basename"] for row in records)
    duplicate_basenames = sorted([name for name, count in basename_counts.items() if count > 1])

    print(f"Loading tokenizer vocab: {tokenizer_name}")
    vocab_tokens = load_bert_vocab(tokenizer_name)

    bundle = {
        "format": "pakaian_adat_captioning_bundle.v1",
        "model_config": {
            "clip_model_name": clip_model_name,
            "tokenizer_name": tokenizer_name,
            "uses_pretrained_clip_visual": True,
            "saved_parts": ["image_encoder.fc", "caption_decoder", "dataset_records", "bert_vocab"],
            "omitted_parts": ["optimizer", "text_encoder", "clip_visual_pretrained_weights"],
        },
        "model_state": {
            "image_encoder_fc": image_encoder_fc,
            "caption_decoder": caption_decoder,
        },
        "dataset": {
            "cond_text": "Image of a",
            "records": records,
            "record_count": len(records),
            "duplicate_basenames": duplicate_basenames,
        },
        "tokenizer": {
            "type": "bert_wordpiece_vocab",
            "vocab_tokens": vocab_tokens,
        },
        "metadata": {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_model": str(source_model),
            "dataset_csv": str(dataset_csv),
            "note": "Deployment bundle excludes optimizer and pretrained CLIP visual weights to reduce size.",
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving bundle: {output}")
    torch.save(bundle, output)
    print(f"Done. Bundle size: {output.stat().st_size / (1024 * 1024):.2f} MB")


def main():
    parser = argparse.ArgumentParser(description="Build portable Streamlit model bundle.")
    parser.add_argument("--source-model", default="../model_captioning_fast.pth")
    parser.add_argument("--dataset-csv", default="../dataset/data_testing_pakaian_adat_2_baru.csv")
    parser.add_argument("--output", default="models/model_pakaian_adat_bundle.pt")
    parser.add_argument("--tokenizer-name", default="bert-base-uncased")
    parser.add_argument("--clip-model-name", default="ViT-L/14@336px")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    source_model = (project_dir / args.source_model).resolve()
    dataset_csv = (project_dir / args.dataset_csv).resolve()
    output = (project_dir / args.output).resolve()

    build_bundle(source_model, dataset_csv, output, args.tokenizer_name, args.clip_model_name)


if __name__ == "__main__":
    main()
