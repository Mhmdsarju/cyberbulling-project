import os

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIG
# ============================================================

INPUT_PATH = (
    "data/processed/"
    "clean_cyberbullying_25000_realworld_multilingual_v3.csv"
)

REPORT_DIR = "reports"
CHART_DIR = "reports/eda_charts"


# ============================================================
# CATEGORY COLUMNS
# ============================================================

CATEGORY_COLUMNS = [
    "insult",
    "threat",
    "violence",
    "hate",
    "sexual_harassment",
    "sexual_abuse",
    "profanity",
]


# ============================================================
# HELPER — SAVE BAR CHART
# ============================================================

def save_bar_chart(
    labels,
    values,
    title,
    xlabel,
    ylabel,
    output_path,
    figsize=(8, 5),
    rotation=0
):

    plt.figure(figsize=figsize)

    plt.bar(
        labels,
        values
    )

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if rotation != 0:

        plt.xticks(
            rotation=rotation,
            ha="right"
        )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("STEP 3 — REAL-WORLD MULTILINGUAL CYBERBULLYING EDA")
    print("=" * 75)

    os.makedirs(
        REPORT_DIR,
        exist_ok=True
    )

    os.makedirs(
        CHART_DIR,
        exist_ok=True
    )

    # ========================================================
    # LOAD DATASET
    # ========================================================

    if not os.path.exists(INPUT_PATH):

        raise FileNotFoundError(
            f"Dataset not found: {INPUT_PATH}"
        )

    df = pd.read_csv(
        INPUT_PATH,
        encoding="utf-8-sig"
    )

    print("\nDataset loaded successfully.")

    print(
        f"Rows    : {len(df)}"
    )

    print(
        f"Columns : {len(df.columns)}"
    )

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
        "label": [
            "Non-Cyberbullying"
            if value == 0
            else "Cyberbullying"
            for value in cyber_counts.index
        ],
        "count": cyber_counts.values,
        "percentage": (
            cyber_counts.values
            / len(df)
            * 100
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

    save_bar_chart(
        labels=cyber_distribution["label"],
        values=cyber_distribution["count"],
        title="Cyberbullying Distribution",
        xlabel="Class",
        ylabel="Number of Samples",
        output_path=(
            f"{CHART_DIR}/01_cyberbullying_distribution.png"
        )
    )

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
            language_counts.values
            / len(df)
            * 100
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

    save_bar_chart(
        labels=language_distribution["language"],
        values=language_distribution["count"],
        title="Language Distribution",
        xlabel="Language",
        ylabel="Number of Samples",
        output_path=(
            f"{CHART_DIR}/02_language_distribution.png"
        )
    )

    # ========================================================
    # 3. CONTEXT TYPE DISTRIBUTION
    # ========================================================

    print("\n" + "-" * 75)
    print("3. CONTEXT TYPE DISTRIBUTION")
    print("-" * 75)

    context_counts = (
        df["context_type"]
        .value_counts()
    )

    context_distribution = pd.DataFrame({
        "context_type": context_counts.index,
        "count": context_counts.values,
        "percentage": (
            context_counts.values
            / len(df)
            * 100
        ).round(2)
    })

    print(
        context_distribution.to_string(
            index=False
        )
    )

    context_distribution.to_csv(
        f"{REPORT_DIR}/context_type_distribution_eda.csv",
        index=False
    )

    save_bar_chart(
        labels=context_distribution["context_type"],
        values=context_distribution["count"],
        title="Social vs Non-Social Distribution",
        xlabel="Context Type",
        ylabel="Number of Samples",
        output_path=(
            f"{CHART_DIR}/03_context_type_distribution.png"
        )
    )

    # ========================================================
    # 4. ENVIRONMENT DISTRIBUTION
    # ========================================================

    print("\n" + "-" * 75)
    print("4. ENVIRONMENT DISTRIBUTION")
    print("-" * 75)

    environment_counts = (
        df["environment"]
        .value_counts()
    )

    environment_distribution = pd.DataFrame({
        "environment": environment_counts.index,
        "count": environment_counts.values,
        "percentage": (
            environment_counts.values
            / len(df)
            * 100
        ).round(2)
    })

    print(
        environment_distribution.to_string(
            index=False
        )
    )

    environment_distribution.to_csv(
        f"{REPORT_DIR}/environment_distribution_eda.csv",
        index=False
    )

    save_bar_chart(
        labels=environment_distribution["environment"],
        values=environment_distribution["count"],
        title="Environment Distribution",
        xlabel="Environment",
        ylabel="Number of Samples",
        output_path=(
            f"{CHART_DIR}/04_environment_distribution.png"
        ),
        figsize=(11, 6),
        rotation=35
    )

    # ========================================================
    # 5. PLATFORM DISTRIBUTION
    # ========================================================

    print("\n" + "-" * 75)
    print("5. PLATFORM DISTRIBUTION")
    print("-" * 75)

    platform_counts = (
        df["platform"]
        .value_counts()
    )

    platform_distribution = pd.DataFrame({
        "platform": platform_counts.index,
        "count": platform_counts.values,
        "percentage": (
            platform_counts.values
            / len(df)
            * 100
        ).round(2)
    })

    print(
        platform_distribution.to_string(
            index=False
        )
    )

    platform_distribution.to_csv(
        f"{REPORT_DIR}/platform_distribution_eda.csv",
        index=False
    )

    save_bar_chart(
        labels=platform_distribution["platform"],
        values=platform_distribution["count"],
        title="Platform Distribution",
        xlabel="Platform",
        ylabel="Number of Samples",
        output_path=(
            f"{CHART_DIR}/05_platform_distribution.png"
        )
    )

    # ========================================================
    # 6. MEDIA TYPE DISTRIBUTION
    # ========================================================

    print("\n" + "-" * 75)
    print("6. MEDIA TYPE DISTRIBUTION")
    print("-" * 75)

    media_counts = (
        df["media_type"]
        .value_counts()
    )

    media_distribution = pd.DataFrame({
        "media_type": media_counts.index,
        "count": media_counts.values,
        "percentage": (
            media_counts.values
            / len(df)
            * 100
        ).round(2)
    })

    print(
        media_distribution.to_string(
            index=False
        )
    )

    media_distribution.to_csv(
        f"{REPORT_DIR}/media_type_distribution_eda.csv",
        index=False
    )

    save_bar_chart(
        labels=media_distribution["media_type"],
        values=media_distribution["count"],
        title="Media Type Distribution",
        xlabel="Media Type",
        ylabel="Number of Samples",
        output_path=(
            f"{CHART_DIR}/06_media_type_distribution.png"
        ),
        rotation=30
    )

    # ========================================================
    # 7. INTENT DISTRIBUTION
    # ========================================================

    print("\n" + "-" * 75)
    print("7. INTENT DISTRIBUTION")
    print("-" * 75)

    intent_counts = (
        df["intent"]
        .value_counts()
    )

    intent_distribution = pd.DataFrame({
        "intent": intent_counts.index,
        "count": intent_counts.values,
        "percentage": (
            intent_counts.values
            / len(df)
            * 100
        ).round(2)
    })

    print(
        intent_distribution.to_string(
            index=False
        )
    )

    intent_distribution.to_csv(
        f"{REPORT_DIR}/intent_distribution_eda.csv",
        index=False
    )

    save_bar_chart(
        labels=intent_distribution["intent"],
        values=intent_distribution["count"],
        title="Intent Distribution",
        xlabel="Intent",
        ylabel="Number of Samples",
        output_path=(
            f"{CHART_DIR}/07_intent_distribution.png"
        ),
        figsize=(11, 6),
        rotation=35
    )

    # ========================================================
    # 8. CONTENT CATEGORY DISTRIBUTION
    # ========================================================

    print("\n" + "-" * 75)
    print("8. CONTENT CATEGORY DISTRIBUTION")
    print("-" * 75)

    content_counts = (
        df["content_category"]
        .value_counts()
    )

    content_distribution = pd.DataFrame({
        "content_category": content_counts.index,
        "count": content_counts.values,
        "percentage": (
            content_counts.values
            / len(df)
            * 100
        ).round(2)
    })

    print(
        content_distribution.to_string(
            index=False
        )
    )

    content_distribution.to_csv(
        f"{REPORT_DIR}/content_category_distribution_eda.csv",
        index=False
    )

    save_bar_chart(
        labels=content_distribution["content_category"],
        values=content_distribution["count"],
        title="Content Category Distribution",
        xlabel="Content Category",
        ylabel="Number of Samples",
        output_path=(
            f"{CHART_DIR}/08_content_category_distribution.png"
        ),
        figsize=(12, 6),
        rotation=40
    )

    # ========================================================
    # 9. OFFENSE CATEGORY DISTRIBUTION
    # ========================================================

    print("\n" + "-" * 75)
    print("9. OFFENSE CATEGORY DISTRIBUTION")
    print("-" * 75)

    category_data = []

    for category in CATEGORY_COLUMNS:

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
            f"{count:5d} "
            f"({percentage:.2f}%)"
        )

    category_distribution = pd.DataFrame(
        category_data
    )

    category_distribution.to_csv(
        f"{REPORT_DIR}/category_distribution_eda.csv",
        index=False
    )

    save_bar_chart(
        labels=category_distribution["category"],
        values=category_distribution["positive_count"],
        title="Offense Category Distribution",
        xlabel="Category",
        ylabel="Number of Positive Samples",
        output_path=(
            f"{CHART_DIR}/09_category_distribution.png"
        ),
        figsize=(11, 6),
        rotation=35
    )

    # ========================================================
    # 10. CATEGORY DISTRIBUTION BY LANGUAGE
    # ========================================================

    print("\n" + "-" * 75)
    print("10. CATEGORY DISTRIBUTION BY LANGUAGE")
    print("-" * 75)

    language_category_rows = []

    for language in sorted(
        df["language"].unique()
    ):

        language_df = df[
            df["language"] == language
        ]

        for category in CATEGORY_COLUMNS:

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
    # 11. SEVERITY DISTRIBUTION
    # ========================================================

    print("\n" + "-" * 75)
    print("11. SEVERITY DISTRIBUTION")
    print("-" * 75)

    severity_order = [
        "none",
        "low",
        "medium",
        "high"
    ]

    severity_counts = (
        df["severity"]
        .value_counts()
        .reindex(
            severity_order,
            fill_value=0
        )
    )

    severity_distribution = pd.DataFrame({
        "severity": severity_counts.index,
        "count": severity_counts.values,
        "percentage": (
            severity_counts.values
            / len(df)
            * 100
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

    save_bar_chart(
        labels=severity_distribution["severity"],
        values=severity_distribution["count"],
        title="Severity Distribution",
        xlabel="Severity",
        ylabel="Number of Samples",
        output_path=(
            f"{CHART_DIR}/10_severity_distribution.png"
        )
    )

    # ========================================================
    # 12. TARGET TYPE DISTRIBUTION
    # ========================================================

    print("\n" + "-" * 75)
    print("12. TARGET TYPE DISTRIBUTION")
    print("-" * 75)

    target_counts = (
        df["target_type"]
        .value_counts()
    )

    target_distribution = pd.DataFrame({
        "target_type": target_counts.index,
        "count": target_counts.values,
        "percentage": (
            target_counts.values
            / len(df)
            * 100
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

    save_bar_chart(
        labels=target_distribution["target_type"],
        values=target_distribution["count"],
        title="Target Type Distribution",
        xlabel="Target Type",
        ylabel="Number of Samples",
        output_path=(
            f"{CHART_DIR}/11_target_type_distribution.png"
        )
    )

    # ========================================================
    # 13. TEXT LENGTH ANALYSIS
    # ========================================================

    print("\n" + "-" * 75)
    print("13. TEXT LENGTH ANALYSIS")
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

    plt.title(
        "Text Length Distribution"
    )

    plt.xlabel(
        "Characters"
    )

    plt.ylabel(
        "Number of Samples"
    )

    plt.tight_layout()

    plt.savefig(
        f"{CHART_DIR}/12_text_length_distribution.png",
        dpi=150
    )

    plt.close()

    # ========================================================
    # 14. TEXT LENGTH BY LANGUAGE
    # ========================================================

    print("\n" + "-" * 75)
    print("14. TEXT LENGTH BY LANGUAGE")
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
    # 15. CYBERBULLYING vs LANGUAGE
    # ========================================================

    print("\n" + "-" * 75)
    print("15. CYBERBULLYING vs LANGUAGE")
    print("-" * 75)

    language_cyber = pd.crosstab(
        df["language"],
        df["cyberbullying"]
    )

    language_cyber.columns = [
        "non_cyberbullying",
        "cyberbullying"
    ]

    print(
        language_cyber.to_string()
    )

    language_cyber.to_csv(
        f"{REPORT_DIR}/language_vs_cyberbullying.csv"
    )

    # ========================================================
    # 16. CYBERBULLYING vs CONTEXT
    # ========================================================

    print("\n" + "-" * 75)
    print("16. CYBERBULLYING vs CONTEXT")
    print("-" * 75)

    context_cyber = pd.crosstab(
        df["context_type"],
        df["cyberbullying"]
    )

    context_cyber.columns = [
        "non_cyberbullying",
        "cyberbullying"
    ]

    print(
        context_cyber.to_string()
    )

    context_cyber.to_csv(
        f"{REPORT_DIR}/context_vs_cyberbullying.csv"
    )

    # ========================================================
    # 17. CYBERBULLYING vs ENVIRONMENT
    # ========================================================

    print("\n" + "-" * 75)
    print("17. CYBERBULLYING vs ENVIRONMENT")
    print("-" * 75)

    environment_cyber = pd.crosstab(
        df["environment"],
        df["cyberbullying"]
    )

    environment_cyber.columns = [
        "non_cyberbullying",
        "cyberbullying"
    ]

    print(
        environment_cyber.to_string()
    )

    environment_cyber.to_csv(
        f"{REPORT_DIR}/environment_vs_cyberbullying.csv"
    )

    # ========================================================
    # 18. CYBERBULLYING vs PLATFORM
    # ========================================================

    print("\n" + "-" * 75)
    print("18. CYBERBULLYING vs PLATFORM")
    print("-" * 75)

    platform_cyber = pd.crosstab(
        df["platform"],
        df["cyberbullying"]
    )

    platform_cyber.columns = [
        "non_cyberbullying",
        "cyberbullying"
    ]

    print(
        platform_cyber.to_string()
    )

    platform_cyber.to_csv(
        f"{REPORT_DIR}/platform_vs_cyberbullying.csv"
    )

    # ========================================================
    # 19. CYBERBULLYING vs INTENT
    # ========================================================

    print("\n" + "-" * 75)
    print("19. CYBERBULLYING vs INTENT")
    print("-" * 75)

    intent_cyber = pd.crosstab(
        df["intent"],
        df["cyberbullying"]
    )

    intent_cyber.columns = [
        "non_cyberbullying",
        "cyberbullying"
    ]

    print(
        intent_cyber.to_string()
    )

    intent_cyber.to_csv(
        f"{REPORT_DIR}/intent_vs_cyberbullying.csv"
    )

    # ========================================================
    # 20. CYBERBULLYING vs CONTENT CATEGORY
    # ========================================================

    print("\n" + "-" * 75)
    print("20. CYBERBULLYING vs CONTENT CATEGORY")
    print("-" * 75)

    content_cyber = pd.crosstab(
        df["content_category"],
        df["cyberbullying"]
    )

    content_cyber.columns = [
        "non_cyberbullying",
        "cyberbullying"
    ]

    print(
        content_cyber.to_string()
    )

    content_cyber.to_csv(
        f"{REPORT_DIR}/content_category_vs_cyberbullying.csv"
    )

    # ========================================================
    # 21. CYBERBULLYING vs SEVERITY
    # ========================================================

    print("\n" + "-" * 75)
    print("21. CYBERBULLYING vs SEVERITY")
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
    # 22. CYBERBULLYING vs TARGET TYPE
    # ========================================================

    print("\n" + "-" * 75)
    print("22. CYBERBULLYING vs TARGET TYPE")
    print("-" * 75)

    cyber_target = pd.crosstab(
        df["target_type"],
        df["cyberbullying"]
    )

    cyber_target.columns = [
        "non_cyberbullying",
        "cyberbullying"
    ]

    print(
        cyber_target.to_string()
    )

    cyber_target.to_csv(
        f"{REPORT_DIR}/target_type_vs_cyberbullying.csv"
    )

    # ========================================================
    # 23. CATEGORY CO-OCCURRENCE
    # ========================================================

    print("\n" + "-" * 75)
    print("23. CATEGORY CO-OCCURRENCE")
    print("-" * 75)

    cooccurrence = pd.DataFrame(
        0,
        index=CATEGORY_COLUMNS,
        columns=CATEGORY_COLUMNS
    )

    for category_a in CATEGORY_COLUMNS:

        for category_b in CATEGORY_COLUMNS:

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
    # 24. CATEGORY POSITIVE COUNT BY CONTEXT
    # ========================================================

    print("\n" + "-" * 75)
    print("24. CATEGORY DISTRIBUTION BY CONTEXT")
    print("-" * 75)

    context_category_rows = []

    for context in sorted(
        df["context_type"].unique()
    ):

        context_df = df[
            df["context_type"] == context
        ]

        for category in CATEGORY_COLUMNS:

            count = int(
                context_df[category].sum()
            )

            context_category_rows.append({
                "context_type": context,
                "category": category,
                "positive_count": count
            })

            print(
                f"{context:22s} | "
                f"{category:20s} | "
                f"{count}"
            )

    context_category_distribution = pd.DataFrame(
        context_category_rows
    )

    context_category_distribution.to_csv(
        f"{REPORT_DIR}/category_distribution_by_context.csv",
        index=False
    )

    # ========================================================
    # 25. DATASET SUMMARY
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
        f"Social media samples   : "
        f"{int((df['context_type'] == 'social_media').sum())}"
    )

    print(
        f"Non-social samples     : "
        f"{int((df['context_type'] == 'non_social_realworld').sum())}"
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

    print(
        f"Median text length     : "
        f"{df['text_length_chars'].median():.2f} characters"
    )

    print(
        f"Average word count     : "
        f"{df['text_length_words'].mean():.2f}"
    )

    print("\nReports created in:")
    print(
        f"  {REPORT_DIR}/"
    )

    print("\nCharts created in:")
    print(
        f"  {CHART_DIR}/"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()