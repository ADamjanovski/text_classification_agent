import json
import os
from typing import Any
from urllib.parse import urlencode
from urllib import error, request

import pandas as pd
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
CLASS_MAPPING = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}


def api_get(path: str) -> Any:
    url = f"{BACKEND_URL}{path}"
    req = request.Request(url=url, method="GET")
    try:
        with request.urlopen(req, timeout=20) as response:
            body = response.read().decode("utf-8")
            return json.loads(body)
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {exc.code} from {path}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Cannot reach backend at {BACKEND_URL}") from exc


def api_post(path: str, payload: dict[str, Any]) -> Any:
    url = f"{BACKEND_URL}{path}"
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url=url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=30) as response:
            body = response.read().decode("utf-8")
            return json.loads(body)
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {exc.code} from {path}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Cannot reach backend at {BACKEND_URL}") from exc


@st.cache_data(ttl=30)
def fetch_models() -> list[str]:
    models = api_get("/models")
    return [model["name"] for model in models]


def class_label(class_id: int) -> str:
    return CLASS_MAPPING.get(class_id, "Unknown")


def build_history_query(
    model_name: str,
    needs_review_only: bool,
    confirmed_filter: str,
    limit: int,
) -> str:
    params: dict[str, Any] = {
        "limit": limit,
        "offset": 0,
        "needs_review": needs_review_only,
    }

    if model_name != "All":
        params["by_model"] = model_name

    if confirmed_filter == "Confirmed":
        params["confirmed_only"] = True
    else:
        params["confirmed_only"] = False

    return "/predictions?" + urlencode(params)


def style_history_rows(row: pd.Series) -> list[str]:
    is_needs_review = row["Review Status"] == "Needs Review"
    is_confirmed = row["Confirmed Class"] != "-"
    if is_needs_review and not is_confirmed:
        return ["background-color: #fff3bf"] * len(row)
    return [""] * len(row)


def render_prediction_card(pred: dict[str, Any]) -> None:
    model_name = pred.get("model_name", "unknown")
    predicted_class = pred.get("predicted_class")
    confidence = pred.get("confidence", 0.0)

    st.markdown(f"### {model_name}")
    st.write(f"Prediction: **{class_label(predicted_class)}**")
    st.write(f"Confidence: **{confidence * 100:.2f}%**")

    if pred.get("needs_review"):
        note = pred.get("note") or "Prediction needs manual review."
        st.warning(note)


def render_predict_page() -> None:
    st.title("Predict")

    text = st.text_area("Text", placeholder="Paste or write news text here...", height=180)
    mode = st.radio("Prediction mode", options=["Single model", "Predict all"], horizontal=True)

    selected_model = None
    if mode == "Single model":
        try:
            models = fetch_models()
        except RuntimeError as exc:
            st.error(str(exc))
            models = []
        selected_model = st.selectbox("Model", options=models, disabled=not models)

    if st.button("Predict", type="primary", use_container_width=True):
        if not text.strip():
            st.error("Please enter text first.")
            return

        try:
            if mode == "Single model":
                response = api_post("/predict", {"text": text, "model_name": selected_model})
                st.session_state["latest_predictions"] = [response]
                st.session_state["latest_disagreement"] = False
            else:
                response = api_post("/predict_all", {"text": text})
                st.session_state["latest_predictions"] = response["predictions"]
                st.session_state["latest_disagreement"] = response["disagreement"]
        except RuntimeError as exc:
            st.error(str(exc))
            return

    predictions: list[dict[str, Any]] = st.session_state.get("latest_predictions", [])
    if not predictions:
        return

    st.divider()
    st.subheader("Prediction result")

    if len(predictions) == 1:
        render_prediction_card(predictions[0])
    else:
        cols = st.columns(len(predictions))
        for idx, pred in enumerate(predictions):
            with cols[idx]:
                render_prediction_card(pred)

    agreed_prediction = predictions[0]["predicted_class"] if len(predictions) == 1 else None
    if len(predictions) > 1:
        predicted_classes = {pred["predicted_class"] for pred in predictions}
        if len(predicted_classes) == 1:
            agreed_prediction = next(iter(predicted_classes))

    st.divider()
    st.subheader("Confirm prediction")

    if len(predictions) > 1 and st.session_state.get("latest_disagreement"):
        st.error("Models disagree. Please choose the confirmed class manually.")
    elif len(predictions) > 1:
        st.warning("Models agree. Default confirmed class is preselected.")

    options = [1, 2, 3, 4]
    default_index = options.index(agreed_prediction) if agreed_prediction in options else None
    confirmed_class = st.selectbox(
        "Confirmed class",
        options=options,
        index=default_index,
        placeholder="Select class",
        format_func=class_label,
    )
    notes = st.text_input("Notes (optional)")

    if st.button("Confirm", use_container_width=True):
        if confirmed_class is None:
            st.error("Please select a confirmed class.")
            return

        confirm_errors = []
        for pred in predictions:
            prediction_id = pred["prediction_id"]
            try:
                api_post(
                    f"/predictions/{prediction_id}/confirm",
                    {"confirmed_class": confirmed_class, "notes": notes or None},
                )
            except RuntimeError as exc:
                confirm_errors.append(str(exc))

        if confirm_errors:
            st.error("Some confirmations failed:")
            for msg in confirm_errors:
                st.write(f"- {msg}")
        else:
            st.success("Prediction(s) confirmed successfully.")


def render_prediction_history_page() -> None:
    st.title("Prediction History")

    try:
        models = fetch_models()
    except RuntimeError as exc:
        st.error(str(exc))
        return

    with st.container(border=True):
        st.subheader("Filters")
        c1, c2, c3, c4 = st.columns(4)
        model_filter = c1.selectbox("Model", options=["All", *models])
        needs_review_filter = c2.checkbox("Needs review only", value=False)
        confirmed_filter = c3.selectbox(
            "Confirmation",
            options=["All", "Confirmed", "Not Confirmed"],
        )
        limit = c4.selectbox("Rows", options=[20, 50, 100], index=1)

    query_path = build_history_query(
        model_name=model_filter,
        needs_review_only=needs_review_filter,
        confirmed_filter=confirmed_filter,
        limit=limit,
    )

    try:
        response = api_get(query_path)
    except RuntimeError as exc:
        st.error(str(exc))
        return

    records: list[dict[str, Any]] = response.get("results", [])
    if confirmed_filter == "Not Confirmed":
        records = [record for record in records if not record.get("confirmed", False)]

    st.caption(f"Showing {len(records)} predictions")
    if not records:
        st.info("No predictions found for selected filters.")
        return

    table_rows = []
    for record in records:
        predicted_class = int(record.get("predicted_class", 0))
        confirmed_class = record.get("confirmed_class")
        table_rows.append(
            {
                "Prediction ID": record.get("prediction_id", ""),
                "Model": record.get("model_name", ""),
                "Prediction": class_label(predicted_class),
                "Confidence": f"{float(record.get('confidence', 0.0)) * 100:.2f}%",
                "Review Status": "Needs Review" if bool(record.get("needs_review", False)) else "OK",
                "Confirmed Class": class_label(int(confirmed_class)) if confirmed_class else "-",
            }
        )

    table_df = pd.DataFrame(table_rows)
    styled_df = table_df.style.apply(style_history_rows, axis=1)

    event = st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
    )

    selected_rows = event.selection.get("rows", []) if event else []
    if not selected_rows:
        st.info("Click a row in the table to see detailed info.")
        return

    selected_idx = selected_rows[0]
    selected_record = records[selected_idx]

    st.divider()
    st.subheader("Prediction details")

    group_id = selected_record.get("prediction_group_id")
    if group_id:
        group_records = [r for r in records if r.get("prediction_group_id") == group_id]
    else:
        group_records = [selected_record]

    if len(group_records) > 1:
        cols = st.columns(len(group_records))
        for idx, rec in enumerate(group_records):
            with cols[idx]:
                st.markdown(f"### {rec.get('model_name', 'unknown')}")
                st.write(f"Predicted: **{class_label(int(rec.get('predicted_class', 0)))}**")
                st.write(f"Confidence: **{float(rec.get('confidence', 0.0)) * 100:.2f}%**")
                if rec.get("needs_review") and not rec.get("confirmed"):
                    st.warning(rec.get("note") or "Needs review.")
    else:
        rec = group_records[0]
        st.write(f"Model: **{rec.get('model_name', 'unknown')}**")
        st.write(f"Predicted class: **{class_label(int(rec.get('predicted_class', 0)))}**")
        st.write(f"Confidence: **{float(rec.get('confidence', 0.0)) * 100:.2f}%**")
        if rec.get("needs_review") and not rec.get("confirmed"):
            st.warning(rec.get("note") or "This prediction needs manual review.")

    rec = group_records[0]
    if rec.get("confirmed"):
        st.success(f"Confirmed: {class_label(int(rec.get('confirmed_class', 0)))}")
        confirmation_note = rec.get("notes")
        if confirmation_note:
            st.info(f"Note: {confirmation_note}")
    st.divider()
    input_text = group_records[0].get("text", "")
    st.markdown("#### Input text")
    st.markdown(
        f"<div style='background:#ffffff;border:1px solid #d8dee4;border-radius:8px;padding:12px;color:#111827;line-height:1.6;white-space:pre-wrap'>{input_text}</div>",
        unsafe_allow_html=True,
    )

    all_confirmed = all(rec.get("confirmed") for rec in group_records)
    if not all_confirmed:
        st.divider()
        st.subheader("Confirm prediction")

    if all_confirmed:
        return

    options = [1, 2, 3, 4]
    agreed_class = (
        next(iter({rec["predicted_class"] for rec in group_records}))
        if len({rec["predicted_class"] for rec in group_records}) == 1
        else None
    )
    default_index_hist = options.index(agreed_class) if agreed_class in options else 0
    confirm_class = st.selectbox(
        "Confirmed class",
        options=options,
        index=default_index_hist,
        format_func=class_label,
        key=f"history_confirm_class_{selected_record.get('prediction_id', '')}",
    )
    confirm_notes = st.text_input(
        "Notes (optional)",
        key=f"history_confirm_notes_{selected_record.get('prediction_id', '')}",
    )

    if st.button("Confirm", use_container_width=True):
        errors = []
        for rec in group_records:
            try:
                api_post(
                    f"/predictions/{rec.get('prediction_id', '')}/confirm",
                    {"confirmed_class": confirm_class, "notes": confirm_notes or None},
                )
            except RuntimeError as exc:
                errors.append(f"{rec.get('model_name', '?')}: {exc}")
        if errors:
            for msg in errors:
                st.error(msg)
        else:
            st.success("All predictions confirmed successfully.")
            st.rerun()


def render_model_info_page() -> None:
    st.title("Model Info")

    try:
        models_info = api_get("/models_info")
    except RuntimeError as exc:
        st.error(str(exc))
        return

    if not models_info:
        st.info("No loaded models found.")
        return

    st.caption("Loaded model metrics from backend registry")
    cols = st.columns(len(models_info))

    for idx, model_info in enumerate(models_info):
        with cols[idx]:
            name = model_info.get("name", "unknown")
            metrics = model_info.get("metrics", {})

            st.subheader(name)
            if not metrics:
                st.warning("No metrics available for this model.")
                continue

            if "accuracy" in metrics:
                st.metric("Accuracy", f"{float(metrics['accuracy']) * 100:.2f}%")
            if "macro_f1" in metrics:
                st.metric("Macro F1", f"{float(metrics['macro_f1']) * 100:.2f}%")
            if "fit_time_sec" in metrics:
                total_seconds = int(round(float(metrics["fit_time_sec"])))
                hours, rem = divmod(total_seconds, 3600)
                minutes, seconds = divmod(rem, 60)

                if hours > 0:
                    fit_time_text = f"{hours}h {minutes}m {seconds}s"
                elif minutes > 0:
                    fit_time_text = f"{minutes}m {seconds}s"
                else:
                    fit_time_text = f"{seconds}s"

                st.metric("Fit Time", fit_time_text)

            with st.expander("Raw metrics", expanded=False):
                st.json(metrics)


def main() -> None:
    st.set_page_config(page_title="News Classification", layout="wide")

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            background-color: #0d2b4e;
        }
        [data-testid="stSidebar"] .st-emotion-cache-1gv3huu {
            color: #ffffff;
        }
        [data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
            color: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    page = st.sidebar.radio(
        "Navigation",
        options=["Predict", "Prediction History", "Model Info"],
    )

    if page == "Predict":
        render_predict_page()
    elif page == "Prediction History":
        render_prediction_history_page()
    else:
        render_model_info_page()


if __name__ == "__main__":
    main()
