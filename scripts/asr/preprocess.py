import argparse
import json
import os
from pathlib import Path

import torchaudio
from sentencepiece import SentencePieceTrainer
from whisper.normalizers import EnglishTextNormalizer

SUBSETS = {
    "train": ["train-clean-100", "train-clean-360", "train-other-500"],
    "valid": ["dev-clean", "dev-other"],
    "dev-clean": ["dev-clean"],
    "dev-other": ["dev-other"],
    "test-clean": ["test-clean"],
    "test-other": ["test-other"],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, default=1024)
    parser.add_argument("--model_type", type=str, default="unigram", choices=["unigram", "bpe"])
    parser.add_argument("--min_duration", type=float, default=1.0)
    parser.add_argument("--max_duration", type=float, default=20.0)
    args = parser.parse_args()
    root_dir = Path(args.data_dir) / "LibriSpeech"
    normalizer = EnglishTextNormalizer()
    text_list = []
    for key in SUBSETS.keys():
        dic = {}
        subsets = SUBSETS[key]
        for subset in subsets:
            reader_dir = root_dir / subset
            readers = os.listdir(reader_dir)
            for reader in readers:
                chapters = os.listdir(reader_dir / reader)
                for chapter in chapters:
                    utt_dir = reader_dir / reader / chapter
                    with open(f"{str(utt_dir)}/{reader}-{chapter}.trans.txt", "r") as f:
                        lines = f.readlines()
                    for line in lines:
                        utt_id, text = line.split(" ", maxsplit=1)
                        audio_path = f"{str(utt_dir)}/{utt_id}.flac"
                        audio_info = torchaudio.info(audio_path)
                        assert audio_info.num_channels == 1
                        assert audio_info.sample_rate == 16000
                        audio_length = audio_info.num_frames
                        if key in ["train", "valid"] and (
                            audio_length < args.min_duration * 16000 or args.max_duration * 16000 < audio_length
                        ):
                            continue
                        text = normalizer(text.replace("\n", ""))
                        sample = {
                            "audio_path": audio_path,
                            "audio_length": audio_length,
                            "text": text,
                        }
                        dic[utt_id] = sample
                        if key == "train":
                            text_list.append(text + "\n")
        os.makedirs(Path(args.data_dir) / "processed", exist_ok=True)
        with open(Path(args.data_dir) / "processed" / f"{key}.json", "w", encoding="utf-8") as f:
            json.dump(dic, f, ensure_ascii=False, indent=4)

    # train text file for tokenizer
    text_file_path = Path(args.data_dir) / "processed" / "train.txt"
    with open(text_file_path, "w", encoding="utf-8") as f:
        f.writelines(text_list)

    # train tokenizer
    os.makedirs(Path(args.out_dir), exist_ok=True)
    SentencePieceTrainer.train(
        input=text_file_path,
        vocab_size=args.vocab_size - 1,  # preserve the last id for <blank> token
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
