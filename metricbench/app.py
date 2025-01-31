from datasets import Dataset, load_dataset, load_dataset_builder
import pandas as pd
import streamlit as st
import pandas as pd
from PIL import Image
import io

# dataset_split_name ="simpleLLM_benchmark_llama3.18binstant_t_0.7"
# dataset_split_name ="simpleLLM_benchmark_llama3.370bversatile_t_0.7"
# dataset_split_name ="simpleLLM_benchmark_deepseekr17b_t_0.7"
dataset_split_name = "simpleLLM_benchmark_mixtral8x7b32768_t_0.7"


# Load dataset (replace with actual dataset loading mechanism)
@st.cache_data
def load_data():
    # Placeholder: Load dataset from a source

    dataset: Dataset = load_dataset(
        "CharlyR/varbench-metric-evaluation", dataset_split, split="tikz"
    )

    return dataset.to_pandas()


data = load_data()

# Select an entry to display (assuming first row for simplicity)
entry = data.iloc[0]

# Display the instruction
st.title("Code Fix Review")
st.subheader("Instruction:")
st.write(entry["instruction"])

# Display the input image
st.subheader("Input Image:")
if entry["image_input"]:
    image_input = Image.open(io.BytesIO(entry["image_input"]))
    st.image(image_input, caption="Image Input", use_column_width=True)
else:
    st.write("No input image available.")

# Display the result image
st.subheader("Result Image:")
if entry["images_result"]:
    image_result = Image.open(io.BytesIO(entry["images_result"]))
    st.image(image_result, caption="Image Result", use_column_width=True)
else:
    st.write("No result image available.")

# Buttons for user response
col1, col2 = st.columns(2)
with col1:
    if st.button("Applied"):
        st.write("You marked this as applied.")

with col2:
    if st.button("Not Applied"):
        st.write("You marked this as not applied.")
