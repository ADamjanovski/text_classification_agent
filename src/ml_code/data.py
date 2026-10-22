# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import matplotlib.pyplot as plt

CLASS_MAPPING = {1 : "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}

def load_dataset(file_path : str):
    # Load the latest version
    df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "amananandrai/ag-news-classification-dataset",
    file_path,
    # Provide any additional arguments like 
    # sql_query or pandas_kwargs. See the 
    # documenation for more information:
    # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
    )
    
    return df

def preprocess_ag_dataset(df: pd.DataFrame):
    cmap = {c.lower().strip(): c for c in df.columns}
    title_col = cmap.get("title")
    desc_col = cmap.get("description")
    label_col = cmap.get("class index")

    text = (
        df[title_col].astype("string").str.strip()
        + "\n"
        + df[desc_col].astype("string").str.strip()
    )

    slim = pd.DataFrame({"text": text, "Class Index": df[label_col]})[
        ["text", "Class Index"]
    ]

    slim = slim.rename(columns={"Class Index": "class"})    
    return slim

def audit_dataset(df : pd.DataFrame):
    label_dist = df["class"].value_counts(dropna=False)
    print("Label distribution:")
    print(label_dist)

    label_dist.plot(kind="bar", figsize=(8, 4), title="Label Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


    # 2) Text length distribution
    # Length in number of characters
    df["text_length_chars"] = df["text"].astype(str).str.len()

    print("\nText length summary:")
    print(df["text_length_chars"].describe())

    plt.figure(figsize=(8, 4))
    df["text_length_chars"].hist(bins=30)
    plt.title("Text Length Distribution (Characters)")
    plt.xlabel("Text Length")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    # Length in words
    df["text_length_words"] = df["text"].astype(str).str.split().str.len()

    print("\nWord length summary:")
    print(df["text_length_words"].describe())

    plt.figure(figsize=(8, 4))
    df["text_length_words"].hist(bins=30)
    plt.title("Text Length Distribution (Words)")
    plt.xlabel("Word Count")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


    # 3) Three example rows per label
    examples = (
        df.groupby("class", group_keys=False)
        .head(3)[["class", "text"]]
    )

    print("\nThree example rows per label:")
    with pd.option_context("display.max_colwidth", None, "display.width", 0):
        print(examples)
