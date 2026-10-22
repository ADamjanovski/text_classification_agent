import pandas as pd


def compare_model_predictions(
    X,
    y_true,
    preds_a,
    preds_b,
    label_names,
    model_a_name="TF-IDF",
    model_b_name="Transformer",
):
    df = pd.DataFrame(
        {
            "text": X.values if hasattr(X, "values") else X,
            "true_label": y_true.values if hasattr(y_true, "values") else y_true,
            "pred_a": preds_a,
            "pred_b": preds_b,
        }
    )

    df["true_label_name"] = df["true_label"].map(label_names)
    df["pred_a_name"] = df["pred_a"].map(label_names)
    df["pred_b_name"] = df["pred_b"].map(label_names)

    df["pred_a_correct"] = df["pred_a"] == df["true_label"]
    df["pred_b_correct"] = df["pred_b"] == df["true_label"]

    disagreements = df[df["pred_a"] != df["pred_b"]].copy()

    print(f"Total compared examples: {len(df)}")
    print(
        f"Disagreements between {model_a_name} and {model_b_name}: "
        f"{len(disagreements)}"
    )

    useful = disagreements[
        disagreements["pred_a_correct"] != disagreements["pred_b_correct"]
    ].head(5)

    if useful.empty:
        print("No cases found where one model is right and the other is wrong.")
        return 

    for idx, (_, row) in enumerate(useful.iterrows(), start=1):
        text = str(row["text"]).replace("\n", " ").strip()


        print(f"\n{'=' * 80}")
        print(f"Example {idx}")
        print(f"{'-' * 80}")
        print(f"True label       : {row['true_label_name']}")
        print(f"{model_a_name:<16}: {row['pred_a_name']} "
              f"(correct={row['pred_a_correct']})")
        print(f"{model_b_name:<16}: {row['pred_b_name']} "
              f"(correct={row['pred_b_correct']})")
        print(f"Text             : {text}")
    return useful


def get_low_confidence_examples(
    X,
    y_true,
    preds,
    confidences,
    label_names,
    n=5,
):
    df = pd.DataFrame(
        {
            "text": X.values if hasattr(X, "values") else X,
            "true_label": y_true.values if hasattr(y_true, "values") else y_true,
            "pred": preds,
            "confidence": confidences,
        }
    )

    df["true_label_name"] = df["true_label"].map(label_names)
    df["pred_label_name"] = df["pred"].map(label_names)
    df["correct"] = df["pred"] == df["true_label"]

    result = df.sort_values("confidence", ascending=True).head(n)

    for idx, (_, row) in enumerate(result.iterrows(), start=1):
        print(f"\n{'=' * 80}")
        print(f"Example {idx}")
        print(f"{'-' * 80}")
        print(f"True label : {row['true_label_name']}")
        print(f"Prediction : {row['pred_label_name']}")
        print(f"Confidence : {row['confidence']:.4f}")
        print(f"Correct    : {row['correct']}")
        print(f"Text       : {row['text']}")

    return result

def get_both_wrong_examples(
    X,
    y_true,
    preds_a,
    preds_b,
    label_names,
    model_a_name="TF-IDF",
    model_b_name="Transformer",
    n=5,
):
    df = pd.DataFrame(
        {
            "text": X.values if hasattr(X, "values") else X,
            "true_label": y_true.values if hasattr(y_true, "values") else y_true,
            "pred_a": preds_a,
            "pred_b": preds_b,
        }
    )

    df["true_label_name"] = df["true_label"].map(label_names)
    df["pred_a_name"] = df["pred_a"].map(label_names)
    df["pred_b_name"] = df["pred_b"].map(label_names)

    df["pred_a_correct"] = df["pred_a"] == df["true_label"]
    df["pred_b_correct"] = df["pred_b"] == df["true_label"]

    result = df[
        (~df["pred_a_correct"]) & (~df["pred_b_correct"])
    ].head(n)

    if result.empty:
            print("No examples found where both models are wrong.")
    else:
        for idx, (_, row) in enumerate(result.iterrows(), start=1):
            print(f"\n{'=' * 80}")
            print(f"Example {idx}")
            print(f"{'-' * 80}")
            print(f"True label        : {row['true_label_name']}")
            print(f"{model_a_name:<18}: {row['pred_a_name']}")
            print(f"{model_b_name:<18}: {row['pred_b_name']}")
            print(f"Text              : {row['text']}")

    return result