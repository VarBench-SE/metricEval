from datasets import Dataset, load_dataset
import pandas as pd
import streamlit as st
from PIL import Image
import io
import random

# Load datasets
@st.cache_data
def load_data():
    raw_dataset: Dataset = load_dataset("CharlyR/varbench-metric-evaluation", "raw", split="train").to_pandas()
    try:
        treated_dataset = load_dataset("CharlyR/varbench-metric-evaluation", "treated", split="train").to_pandas()
    except:
        treated_df = raw_dataset.iloc[:0]  # Keeps the structure but removes rows
        treated_dataset = Dataset.from_pandas(treated_df)

    # Remove treated entries from raw dataset
    untreated = raw_dataset[~raw_dataset["id"].isin(treated_dataset["id"])]
    return untreated, treated_dataset

untreated_data, treated_data = load_data()

if untreated_data.empty:
    st.write("No more entries to review!")
    st.stop()

# Select a random entry
if "selected_entry" not in st.session_state:
    st.session_state.selected_entry = untreated_data.sample(n=1).iloc[0]

entry = st.session_state.selected_entry

# Display the instruction
st.title("Code Fix Review")
st.subheader("Instruction:")
st.write(entry["instruction"])

# Display images side by side
st.subheader("Comparison")
col1, col2 = st.columns(2)

if entry["image_input"]:
    with col1:
        image_input = Image.open(io.BytesIO(entry["image_input"]["bytes"]))
        st.image(image_input, caption="Input Image", use_column_width=True)
else:
    with col1:
        st.write("No input image available.")

if entry["images_result"]:
    with col2:
        image_result = Image.open(io.BytesIO(entry["images_result"]["bytes"]))
        st.image(image_result, caption="Result Image", use_column_width=True)
else:
    with col2:
        st.write("No result image available.")

# Rating scale
st.subheader("How well was the instruction applied?")
score = st.slider("Rate from 1 (not applied) to 10 (perfectly applied)", 1, 10, 5)

# Submit response
if st.button("Submit Review"):
    new_entry = entry.copy()
    new_entry["score"] = score
    treated_data = pd.concat([treated_data, pd.DataFrame([new_entry])], ignore_index=True)
    treated_data.to_csv("treated_data.csv", index=False)
    st.session_state.pop("selected_entry")  # Reset selection for a new entry
    st.success("Review submitted! Refresh for a new entry.")

# Button to update remote dataset
if st.button("Update Remote Dataset"):
    treated_dataset = Dataset.from_pandas(treated_data)
    treated_dataset.push_to_hub("CharlyR/varbench-metric-evaluation", config_name="treated")
    st.success("Remote dataset updated!")
