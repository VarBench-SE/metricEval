import PIL.Image
from datasets import Dataset, load_dataset, concatenate_datasets, Value, Image
from huggingface_hub import login
import pandas as pd
import streamlit as st
import PIL
import io
import random
import hashlib


# st.markdown("# :red[MAINTENANCE]")
# st.stop()
if "token" not in st.session_state:
    st.session_state.token = ""
token = st.text_input("HuggingFace Token")


if not token:
    st.warning("Please enter your HuggingFace Token to continue.")
    st.stop()  # Stops execution until token is provided

st.session_state.token = token  # Store token
try:
    login(token=token)
except:
    st.warning("invalid token")
    st.stop()


# Load datasets
@st.cache_data
def load_data() -> Dataset:
    raw_dataset: Dataset = load_dataset(
        "CharlyR/varbench-metric-evaluation", "raw", split="train"
    )
    return raw_dataset


raw_dataset: Dataset = load_data()
raw_dataframe: pd.DataFrame = raw_dataset.to_pandas()

# Store treated entries locally
if "treated_entries" not in st.session_state:
    st.session_state.treated_entries = []

# Select a random entry
if "selected_entry" not in st.session_state:
    st.session_state.selected_entry = raw_dataframe.sample(n=1).iloc[0].to_dict()

entry = st.session_state.selected_entry

# Display the instruction
st.title("Code Edit Review")
st.subheader("Instruction:")
st.write(entry["instruction"])

# Display images side by side
st.subheader("Comparison")
col1, col2, col3 = st.columns(3)

if entry["image_input"]:
    with col1:
        image_input = PIL.Image.open(io.BytesIO(entry["image_input"]["bytes"]))
        st.image(image_input, caption="Input Image", use_container_width=True)
else:
    with col1:
        st.write("No input image available.")

if entry["images_result"]:
    with col2:
        image_result = PIL.Image.open(io.BytesIO(entry["images_result"]["bytes"]))
        st.image(image_result, caption="LLM-Generated Image", use_container_width=True)
else:
    with col2:
        st.write("No result image available.")

if entry["image_solution"]:
    with col3:
        image_result = PIL.Image.open(io.BytesIO(entry["image_solution"]["bytes"]))
        st.image(
            image_result, caption='Reference "wanted" Image', use_container_width=True
        )
else:
    with col3:
        st.write("No result image available.")


st.subheader("How well was the instruction applied?")

# Create the slider
score = st.slider(
    "Rate from 1 (not applied) to 5 (perfectly applied)",
    1,
    3,
    5,
)

# Add custom labels above the slider
st.markdown(
    """
    <style>
        .slider-container {
            display: flex;
            justify-content: space-between;
            width: 100%;
            padding: 0 10px;
        }
        .slider-container span {
            font-size: 14px;
            font-weight: bold;
            color: #4e4e4e;
        }
    </style>
    <div class="slider-container">
        <span>Not at all</span>
        <span>Partially</span>
        <span>Moderately</span>
        <span>Well</span>
        <span>Perfectly</span>
    </div>
""",
    unsafe_allow_html=True,
)


def get_dataset_treated() -> pd.DataFrame:
    try:
        return load_dataset(
            "CharlyR/varbench-metric-evaluation", "treated_new", split="train"
        )
    except:
        treated_df = raw_dataframe.iloc[:0]  # Keeps the structure but removes rows
        treated_ds_initial: Dataset = (
            Dataset.from_pandas(treated_df, features=raw_dataset.features)
            .add_column("human_score", [], feature=Value("int64"))
            .add_column("reviewer_id", [], feature=Value("string"))
        )
        return treated_ds_initial


def update_remote():
    remote_treated_dataset = get_dataset_treated()
    local_df = pd.DataFrame(st.session_state.treated_entries)
    new_ds = Dataset.from_pandas(local_df, features=remote_treated_dataset.features)
    local_dataset: Dataset = (
        new_ds.cast_column("image_solution", Image(decode=True))
        .cast_column("images_result", Image(decode=True))
        .cast_column("image_input", Image(decode=True))
    )
    if len(remote_treated_dataset) > 0:
        topush_dataset = concatenate_datasets([remote_treated_dataset, local_dataset])
    else:
        topush_dataset = local_dataset
    topush_dataset.push_to_hub(
        "CharlyR/varbench-metric-evaluation", config_name="treated_new"
    )
    st.session_state.treated_entries = []  # Clear after pushing
    st.success("All local reviews have been pushed to the remote repository!")
    st.rerun()


# Display the count of reviews not pushed
st.subheader("Pending Reviews to Push: ")
st.write(len(st.session_state.treated_entries))


combined_values = "".join(
    [
        value
        for key, value in st.context.headers.items()
        if "Sec-Websocket" not in key or "cookie" not in key
    ]
)
hashed_combined = hashlib.sha256(combined_values.encode()).hexdigest()

# Submit response locally
if st.button("Submit Review"):
    new_entry = entry.copy()
    new_entry["human_score"] = score
    new_entry["reviewer_id"] = hashed_combined
    st.session_state.treated_entries.append(new_entry)
    st.session_state.pop("selected_entry")  # Reset selection for a new entry
    st.success("Review saved locally! Refresh for a new entry.")
    st.rerun()

# Push all saved reviews to remote
if st.button("Push Reviews to Remote"):
    if st.session_state.treated_entries:
        update_remote()
    else:
        st.warning("No reviews to push!")
