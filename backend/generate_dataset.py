"""Generate a reproducible synthetic dataset for phase4-features-v1."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 42
GROUP_COUNT = 24
SAMPLES_PER_GROUP = 5

FEATURE_COLUMNS = [
    "semantic_score",
    "keyword_score",
    "filler_rate_percent",
    "repetition_rate_percent",
    "average_sentence_words",
    "speaking_rate_wpm",
    "pauses_per_minute",
    "speech_ratio_percent",
    "rms_variation_percent",
    "has_content_features",
    "has_voice_features",
]

OUTPUT_PATH = Path(__file__).resolve().with_name("labeled_scores.csv")


def generate_dataset() -> pd.DataFrame:
    """Create 120 synthetic samples across 24 candidate groups."""
    rng = np.random.default_rng(RANDOM_SEED)
    group_abilities = rng.normal(0.0, 1.0, GROUP_COUNT)
    records: list[dict[str, str | float]] = []

    for group_index, group_ability in enumerate(group_abilities, start=1):
        for _ in range(SAMPLES_PER_GROUP):
            quality = float(np.clip(
                0.5 + group_ability * 0.12 + rng.normal(0.0, 0.09),
                0.0,
                1.0,
            ))
            semantic_score = float(np.clip(
                38.0 + quality * 50.0 + rng.normal(0.0, 8.0), 0.0, 100.0
            ))
            keyword_score = float(np.clip(
                32.0 + quality * 55.0 + rng.normal(0.0, 9.0), 0.0, 100.0
            ))
            filler_rate = float(np.clip(
                22.0 - quality * 16.0 + rng.normal(0.0, 4.0), 0.0, 100.0
            ))
            repetition_rate = float(np.clip(
                18.0 - quality * 12.0 + rng.normal(0.0, 4.0), 0.0, 100.0
            ))
            average_sentence_words = float(np.clip(
                7.0 + quality * 8.0 + rng.normal(0.0, 2.0), 0.0, 500.0
            ))
            speaking_rate = float(np.clip(
                105.0 + quality * 65.0 + rng.normal(0.0, 20.0), 0.0, 600.0
            ))
            pauses_per_minute = float(np.clip(
                12.0 - quality * 7.0 + rng.normal(0.0, 2.0), 0.0, 300.0
            ))
            speech_ratio = float(np.clip(
                62.0 + quality * 25.0 + rng.normal(0.0, 7.0), 0.0, 100.0
            ))
            rms_variation = float(np.clip(
                32.0 + rng.normal(0.0, 12.0), 0.0, 1000.0
            ))

            target_score = float(np.clip(
                0.38 * semantic_score
                + 0.28 * keyword_score
                + 0.10 * (100.0 - filler_rate)
                + 0.08 * (100.0 - repetition_rate)
                + 0.06 * min(average_sentence_words * 6.0, 100.0)
                + 0.06 * (speaking_rate / 6.0)
                + 0.04 * speech_ratio
                + rng.normal(0.0, 4.0),
                0.0,
                100.0,
            ))

            records.append({
                "sample_id": f"sample-{len(records) + 1:03d}",
                "group_id": f"candidate-{group_index:03d}",
                "semantic_score": round(semantic_score, 2),
                "keyword_score": round(keyword_score, 2),
                "filler_rate_percent": round(filler_rate, 2),
                "repetition_rate_percent": round(repetition_rate, 2),
                "average_sentence_words": round(average_sentence_words, 2),
                "speaking_rate_wpm": round(speaking_rate, 2),
                "pauses_per_minute": round(pauses_per_minute, 2),
                "speech_ratio_percent": round(speech_ratio, 2),
                "rms_variation_percent": round(rms_variation, 2),
                "has_content_features": 1.0,
                "has_voice_features": 1.0,
                "target_score": round(target_score, 2),
            })

    columns = [
        "sample_id",
        "group_id",
        *FEATURE_COLUMNS,
        "target_score",
    ]
    return pd.DataFrame.from_records(records, columns=columns)


def main() -> None:
    dataset = generate_dataset()
    dataset.to_csv(OUTPUT_PATH, index=False)
    print(
        f"Generated {len(dataset)} samples across "
        f"{dataset['group_id'].nunique()} candidate groups: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
