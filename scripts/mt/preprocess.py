import argparse
import json
import os
from pathlib import Path

from sentencepiece import SentencePieceTrainer
from tqdm import tqdm

DATA_DICT = {
    "training": {"key": "train", "files": ["europarl-v7.de-en", "commoncrawl.de-en", "news-commentary-v9.de-en"]},
    "dev": {"key": "valid", "files": ["newstest2013"]},
    "test-full": {"key": "test", "files": ["newstest2014-deen-src"]},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, default=8000)
    parser.add_argument("--model_type", type=str, default="bpe")
    args = parser.parse_args()
    text_list = []
    for key in DATA_DICT.keys():
        for file_name in DATA_DICT[key]["files"]:
            line_dict = {}
            for lang in ["en", "de"]:
                with open(f"{args.data_dir}/{key}/{file_name}.{lang}", "r", encoding="utf-8", newline="\n") as f:
                    line_dict[lang] = f.readlines()
            assert len(line_dict["en"]) == len(line_dict["de"]), (len(line_dict["en"]), len(line_dict["de"]))
            dic = {}
            for i, (src_line, tgt_line) in tqdm(
                enumerate(zip(line_dict["en"], line_dict["de"])), total=len(line_dict["en"]), desc=file_name
            ):
                sample_id = f"{file_name}_{str(i).zfill(8)}"
                src_text = src_line.strip()
                tgt_text = tgt_line.strip()
                if len(src_text) == 0 or len(tgt_text) == 0:
                    continue
                dic[sample_id] = {"src_text": src_line, "tgt_text": tgt_line}
                if key == "training":
                    text_list.append(dic[sample_id]["src_text"] + "\n" + dic[sample_id]["tgt_text"] + "\n")
        os.makedirs(Path(args.data_dir) / "mt", exist_ok=True)
        with open(Path(args.data_dir) / "mt" / f"{DATA_DICT[key]['key']}.json", "w", encoding="utf-8") as f:
            json.dump(dic, f, ensure_ascii=False, indent=4)

    # train text file for tokenizer
    text_file_path = Path(args.data_dir) / "mt" / "train.txt"
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
