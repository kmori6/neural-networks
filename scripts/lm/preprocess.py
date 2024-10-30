import argparse
import json
import os
from pathlib import Path

from sentencepiece import SentencePieceTrainer
from tqdm import tqdm
from whisper.normalizers import EnglishTextNormalizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, default=1024)
    parser.add_argument("--model_type", type=str, default="unigram", choices=["unigram", "bpe"])
    parser.add_argument("--valid_rate", type=float, default=0.1)
    parser.add_argument("--test_rate", type=float, default=0.1)
    args = parser.parse_args()
    normalizer = EnglishTextNormalizer()
    with open(f"{args.data_dir}/librispeech-lm-norm.txt", "r") as f:
        lines = f.readlines()
    num_samples = len(lines)
    num_train_samples = int(num_samples * (1 - args.valid_rate - args.test_rate))
    subsets = {
        "train": lines[:num_train_samples],
        "valid": lines[num_train_samples : int(num_samples * (1 - args.test_rate))],
        "test": lines[int(num_samples * (1 - args.test_rate)) :],
    }
    text_list = []
    for key in subsets.keys():
        dic = {}
        subset = subsets[key]
        for i, line in tqdm(enumerate(subset), total=len(subset), desc=key):
            sample_id = str(i).zfill(8)
            text = normalizer(line.replace("\n", ""))
            dic[sample_id] = {"text": text}
            if key == "train":
                text_list.append(text + "\n")
        os.makedirs(Path(args.data_dir) / "lm", exist_ok=True)
        with open(Path(args.data_dir) / "lm" / f"{key}.json", "w", encoding="utf-8") as f:
            json.dump(dic, f, ensure_ascii=False, indent=4)

    # train text file for tokenizer
    text_file_path = Path(args.data_dir) / "lm" / "train.txt"
    with open(text_file_path, "w", encoding="utf-8") as f:
        f.writelines(text_list)

    # train tokenizer
    os.makedirs(Path(args.out_dir), exist_ok=True)
    SentencePieceTrainer.train(
        input=text_file_path,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        model_prefix=f"{args.out_dir}/tokenizer",
        character_coverage=1.0,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
    )


if __name__ == "__main__":
    main()
