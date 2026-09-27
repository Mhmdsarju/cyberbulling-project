import os

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = (
    "data/processed/"
    "clean_cyberbullying_robust_multilingual_final.csv"
)

REPORT_DIR = "reports"
CHART_DIR = "reports/eda_charts"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("STEP 3 — MULTILINGUAL CYBERBULLYING EDA")
    print("=" * 75)

    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(CHART_DIR, exist_ok=True)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    if not os.path.exists(INPUT_PATH):

        raise FileNotFoundError(
            f"Dataset not found: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    print("\nDataset loaded successfully.")
    print(f"Rows    : {len(df)}")
    print(f"Columns : {len(df.columns)}")

    # ========================================================
    # 1. CYBERBULLYING DISTRIBUTION
    # ========================================================

    print("\n" + "-" * 75)
    print("1. CYBERBULLYING DISTRIBUTION")
    print("-" * 75)

    cyber_counts = (
        df["cyberbullying"]
        .value_counts()
        .sort_index()
    )

    cyber_distribution = pd.DataFrame({
        "cyberbullying": cyber_counts.index,
        "count": cyber_counts.values,
        "percentage": (
            cyber_counts.values / len(df) * 100
        ).round(2)
    })

    print(
        cyber_distribution.to_string(
            index=False
        )
    )

    cyber_distribution.to_csv(
        f"{REPORT_DIR}/cyberbullying_distribution.csv",
        index=False
    )

    plt.figure(figsize=(7, 5))

    labels = [
        "Non-Cyberbullying",
        "Cyberbullying"
    ]

    plt.bar(
        labels,
        cyber_counts.values
    )

    plt.title("Cyberbullying Distribution")
    plt.xlabel("Class")
    plt.ylabel("Number of Samples")
    plt.tight_layout()

    plt.savefig(
        f"{CHART_DIR}/01_cyberbullying_distribution.png",
        dpi=150
    )

    plt.close()

    # ========================================================
    # 2. LANGUAGE DISTRIBUTION
    # ========================================================

    print("\n" + "-" * 75)
    print("2. LANGUAGE DISTRIBUTION")
    print("-" * 75)

    language_counts = (
        df["language"]
        .value_counts()
    )

    language_distribution = pd.DataFrame({
        "language": language_counts.index,
        "count": language_counts.values,
        "percentage": (
            language_counts.values / len(df) * 100
        ).round(2)
    })

    print(
        language_distribution.to_string(
            index=False
        )
    )

    language_distribution.to_csv(
        f"{REPORT_DIR}/language_distribution_eda.csv",
        index=False
    )

    plt.figure(figsize=(7, 5))

    plt.bar(
        language_distribution["language"],
        language_distribution["count"]
    )

    plt.title("Language Distribution")
    plt.xlabel("Language")
    plt.ylabel("Number of Samples")
    plt.tight_layout()

    plt.savefig(
        f"{CHART_DIR}/02_language_distribution.png",
        dpi=150
    )

    plt.close()

    # ========================================================
    # 3. DATA SOURCE DISTRIBUTION
    # ========================================================

    print("\n" + "-" * 75)
    print("3. DATA SOURCE DISTRIBUTION")
    print("-" * 75)

    source_counts = (
        df["data_source"]
        .value_counts()
    )

    source_distribution = pd.DataFrame({
        "data_source": source_counts.index,
        "count": source_counts.values,
        "percentage": (
            source_counts.values / len(df) * 100
        ).round(2)
    })

    print(
        source_distribution.to_string(
            index=False
        )
    )

    source_distribution.to_csv(
        f"{REPORT_DIR}/data_source_distribution_eda.csv",
        index=False
    )

    # ========================================================
    # 4. LANGUAGE vs CYBERBULLYING
    # ========================================================

    print("\n" + "-" * 75)
    print("4. LANGUAGE vs CYBERBULLYING")
    print("-" * 75)

    language_cyber = pd.crosstab(
        df["language"],
        df["cyberbullying"]
    )

    print(
        language_cyber.to_string()
    )

    language_cyber.to_csv(
        f"{REPORT_DIR}/language_vs_cyberbullying.csv"
    )

    # ========================================================
    # 5. CATEGORY DISTRIBUTION
    # ========================================================

    print("\n" + "-" * 75)
    print("5. CATEGORY DISTRIBUTION")
    print("-" * 75)

    category_columns = [
        "insult",
        "threat",
        "violence",
        "hate",
        "sexual_harassment",
        "sexual_abuse",
        "profanity",
    ]

    category_data = []

    for category in category_columns:

        count = int(
            df[category].sum()
        )

        percentage = round(
            count / len(df) * 100,
            2
        )

        category_data.append({
            "category": category,
            "positive_count": count,
            "percentage": percentage
        })

        print(
            f"{category:20s}: "
            f"{count:4d} "
            f"({percentage:.2f}%)"
        )

    category_distribution = pd.DataFrame(
        category_data
    )

    category_distribution.to_csv(
        f"{REPORT_DIR}/category_distribution_eda.csv",
        index=False
    )

    plt.figure(figsize=(10, 6))

    plt.bar(
        category_distribution["category"],
        category_distribution["positive_count"]
    )

    plt.title(
        "Cyberbullying Category Distribution"
    )

    plt.xlabel("Category")
    plt.ylabel("Number of Positive Samples")

    plt.xticks(
        rotation=35,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        f"{CHART_DIR}/03_category_distribution.png",
        dpi=150
    )

    plt.close()

    # ========================================================
    # 6. CATEGORY DISTRIBUTION BY LANGUAGE
    # ========================================================

    print("\n" + "-" * 75)
    print("6. CATEGORY DISTRIBUTION BY LANGUAGE")
    print("-" * 75)

    language_category_rows = []

    for language in sorted(
        df["language"].unique()
    ):

        language_df = df[
            df["language"] == language
        ]

        for category in category_columns:

            count = int(
                language_df[category].sum()
            )

            language_category_rows.append({
                "language": language,
                "category": category,
                "positive_count": count
            })

            print(
                f"{language:10s} | "
                f"{category:20s} | "
                f"{count}"
            )

    language_category_distribution = pd.DataFrame(
        language_category_rows
    )

    language_category_distribution.to_csv(
        f"{REPORT_DIR}/category_distribution_by_language.csv",
        index=False
    )

    # ========================================================
    # 7. SEVERITY DISTRIBUTION
    # ========================================================

    print("\n" + "-" * 75)
    print("7. SEVERITY DISTRIBUTION")
    print("-" * 75)

    severity_counts = (
        df["severity"]
        .value_counts()
        .reindex(
            [
                "none",
                "low",
                "medium",
                "high"
            ],
            fill_value=0
        )
    )

    severity_distribution = pd.DataFrame({
        "severity": severity_counts.index,
        "count": severity_counts.values,
        "percentage": (
            severity_counts.values / len(df) * 100
        ).round(2)
    })

    print(
        severity_distribution.to_string(
            index=False
        )
    )

    severity_distribution.to_csv(
        f"{REPORT_DIR}/severity_distribution_eda.csv",
        index=False
    )

    plt.figure(figsize=(7, 5))

    plt.bar(
        severity_distribution["severity"],
        severity_distribution["count"]
    )

    plt.title("Severity Distribution")
    plt.xlabel("Severity")
    plt.ylabel("Number of Samples")

    plt.tight_layout()

    plt.savefig(
        f"{CHART_DIR}/04_severity_distribution.png",
        dpi=150
    )

    plt.close()

    # ========================================================
    # 8. TARGET TYPE DISTRIBUTION
    # ========================================================

    print("\n" + "-" * 75)
    print("8. TARGET TYPE DISTRIBUTION")
    print("-" * 75)

    target_counts = (
        df["target_type"]
        .value_counts()
    )

    target_distribution = pd.DataFrame({
        "target_type": target_counts.index,
        "count": target_counts.values,
        "percentage": (
            target_counts.values / len(df) * 100
        ).round(2)
    })

    print(
        target_distribution.to_string(
            index=False
        )
    )

    target_distribution.to_csv(
        f"{REPORT_DIR}/target_type_distribution_eda.csv",
        index=False
    )

    plt.figure(figsize=(8, 5))

    plt.bar(
        target_distribution["target_type"],
        target_distribution["count"]
    )

    plt.title("Target Type Distribution")
    plt.xlabel("Target Type")
    plt.ylabel("Number of Samples")

    plt.tight_layout()

    plt.savefig(
        f"{CHART_DIR}/05_target_type_distribution.png",
        dpi=150
    )

    plt.close()

    # ========================================================
    # 9. TEXT LENGTH ANALYSIS
    # ========================================================

    print("\n" + "-" * 75)
    print("9. TEXT LENGTH ANALYSIS")
    print("-" * 75)

    df["text_length_chars"] = (
        df["text"]
        .astype(str)
        .str.len()
    )

    df["text_length_words"] = (
        df["text"]
        .astype(str)
        .str.split()
        .str.len()
    )

    text_stats = pd.DataFrame({
        "metric": [
            "minimum_characters",
            "maximum_characters",
            "mean_characters",
            "median_characters",
            "minimum_words",
            "maximum_words",
            "mean_words",
            "median_words",
        ],

        "value": [

            df["text_length_chars"].min(),

            df["text_length_chars"].max(),

            round(
                df["text_length_chars"].mean(),
                2
            ),

            df["text_length_chars"].median(),

            df["text_length_words"].min(),

            df["text_length_words"].max(),

            round(
                df["text_length_words"].mean(),
                2
            ),

            df["text_length_words"].median(),
        ]
    })

    print(
        text_stats.to_string(
            index=False
        )
    )

    text_stats.to_csv(
        f"{REPORT_DIR}/text_length_statistics.csv",
        index=False
    )

    plt.figure(figsize=(9, 5))

    plt.hist(
        df["text_length_chars"],
        bins=30
    )

    plt.title("Text Length Distribution")
    plt.xlabel("Characters")
    plt.ylabel("Number of Samples")

    plt.tight_layout()

    plt.savefig(
        f"{CHART_DIR}/06_text_length_distribution.png",
        dpi=150
    )

    plt.close()

    # ========================================================
    # 10. TEXT LENGTH BY LANGUAGE
    # ========================================================

    print("\n" + "-" * 75)
    print("10. TEXT LENGTH BY LANGUAGE")
    print("-" * 75)

    language_text_stats = (
        df.groupby("language")
        .agg(
            mean_characters=(
                "text_length_chars",
                "mean"
            ),
            median_characters=(
                "text_length_chars",
                "median"
            ),
            mean_words=(
                "text_length_words",
                "mean"
            ),
            median_words=(
                "text_length_words",
                "median"
            )
        )
        .round(2)
        .reset_index()
    )

    print(
        language_text_stats.to_string(
            index=False
        )
    )

    language_text_stats.to_csv(
        f"{REPORT_DIR}/text_length_by_language.csv",
        index=False
    )

    # ========================================================
    # 11. CYBERBULLYING vs SEVERITY
    # ========================================================

    print("\n" + "-" * 75)
    print("11. CYBERBULLYING vs SEVERITY")
    print("-" * 75)

    cyber_severity = pd.crosstab(
        df["cyberbullying"],
        df["severity"]
    )

    print(
        cyber_severity.to_string()
    )

    cyber_severity.to_csv(
        f"{REPORT_DIR}/cyberbullying_vs_severity.csv"
    )

    # ========================================================
    # 12. CATEGORY CO-OCCURRENCE
    # ========================================================

    print("\n" + "-" * 75)
    print("12. CATEGORY CO-OCCURRENCE")
    print("-" * 75)

    cooccurrence = pd.DataFrame(
        0,
        index=category_columns,
        columns=category_columns
    )

    for category_a in category_columns:

        for category_b in category_columns:

            cooccurrence.loc[
                category_a,
                category_b
            ] = int(
                (
                    (df[category_a] == 1)
                    &
                    (df[category_b] == 1)
                ).sum()
            )

    print(
        cooccurrence.to_string()
    )

    cooccurrence.to_csv(
        f"{REPORT_DIR}/category_cooccurrence.csv"
    )

    # ========================================================
    # 13. DATASET SUMMARY
    # ========================================================

    print("\n" + "=" * 75)
    print("EDA SUMMARY")
    print("=" * 75)

    print(
        f"Total samples          : "
        f"{len(df)}"
    )

    print(
        f"Tanglish samples       : "
        f"{int((df['language'] == 'tanglish').sum())}"
    )

    print(
        f"English samples        : "
        f"{int((df['language'] == 'english').sum())}"
    )

    print(
        f"Cyberbullying samples  : "
        f"{int(df['cyberbullying'].sum())}"
    )

    print(
        f"Non-bullying samples   : "
        f"{int((df['cyberbullying'] == 0).sum())}"
    )

    print(
        f"Average text length    : "
        f"{df['text_length_chars'].mean():.2f} characters"
    )

    print("\nReports created in:")
    print(f"  {REPORT_DIR}/")

    print("\nCharts created in:")
    print(f"  {CHART_DIR}/")

    print("\nStep 3 completed successfully.")


if __name__ == "__main__":
    main()