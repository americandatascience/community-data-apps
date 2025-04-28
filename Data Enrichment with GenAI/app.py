import streamlit as st
import pandas as pd
import os
import asyncio
from ai21 import AsyncAI21Client
from ai21.models.chat import ChatMessage
from io import StringIO

st.set_page_config(page_title="AI Data Enrichment Tool", layout="wide")
st.title("🤖 AI Data Enrichment Tool")
st.write("Upload your CSV and enrich it with AI-powered features!")

# --- Sidebar: Model and Batch Size ---
st.sidebar.header("Configuration")
model = st.sidebar.selectbox("Model", ["jamba-mini-1.6-2025-03"], index=0)
batch_size = st.sidebar.slider("Batch Size (for async)", 1, 10, 5)

# --- Prompt Templates ---
prompt_templates = {
    "Sentiment Analysis": "Extract sentiment as Positive/Negative, or as specified.",
    "Summarization": "Summarize the content in one sentence.",
    "Entity Extraction": "Extract all drug names mentioned in the text.",
    "Custom": ""
}

st.write("#### 1. Choose a prompt template or write your own:")
template_choice = st.selectbox("Prompt Template", list(prompt_templates.keys()), index=0, key="template_choice")

# Use session state to manage the prompt text area
if "user_prompt" not in st.session_state:
    st.session_state.user_prompt = prompt_templates[template_choice]

# Update prompt when template changes, but not when user edits
if st.session_state.template_choice != "Custom":
    st.session_state.user_prompt = prompt_templates[st.session_state.template_choice]

user_prompt = st.text_area(
    "Prompt (describe what you want to enrich for each row):",
    value=st.session_state.user_prompt,
    height=80,
    key="user_prompt"
)

# --- File Upload ---
st.write("#### 2. Upload your CSV file:")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded Data:")
    st.dataframe(df.head())

    # --- Column Selection ---
    st.write("#### 3. Select column(s) to enrich:")
    columns = df.columns.tolist()
    selected_cols = st.multiselect("Columns to enrich", columns, default=[columns[0]])
    if not selected_cols:
        st.warning("Please select at least one column to enrich.")

    # --- Output Column Name ---
    st.write("#### 4. Name your output column:")
    default_output_col = f"enriched_{'_'.join(selected_cols)}"
    output_col = st.text_input("Output column name", value=default_output_col)

    # --- Helper: Async Enrichment Function ---
    async def enrich_dataframe(df, prompt, model, batch_size, selected_cols, output_col):
        async_client = AsyncAI21Client()
        n = len(df)
        enriched_col = []
        progress = st.progress(0)
        status = st.empty()
        live_table = st.empty()
        temp_df = df.copy()
        
        async def enrich_row(row, idx):
            # Build the user message from selected columns
            col_text = ', '.join([f"{col}: {row[col]}" for col in selected_cols])
            messages = [
                ChatMessage(content="Don't be verbose, and don't use uncessary words. You're generating raw data, not paragraphs/sentences of text. Anything you output is used directly as data enrichment. Must be extremely consistent.", role="system"),
                ChatMessage(content=prompt, role="system"),
                ChatMessage(content=f"{col_text}", role="user"),
            ]
            try:
                # Stream the response
                full_response = ""
                async for chunk in await async_client.chat.completions.create(
                    messages=messages,
                    model=model,
                    stream=True,
                ):
                    if chunk.choices and chunk.choices[0].delta.content:
                        full_response += chunk.choices[0].delta.content
                return full_response.strip()
            except Exception as e:
                return f"Error: {str(e)}"
        
        # Process in batches
        for i in range(0, n, batch_size):
            batch = df.iloc[i:i+batch_size]
            tasks = [enrich_row(row, idx) for idx, row in batch.iterrows()]
            batch_results = await asyncio.gather(*tasks)
            enriched_col.extend(batch_results)
            temp_df.loc[batch.index, output_col] = batch_results
            progress.progress(min((i+batch_size)/n, 1.0))
            status.info(f"Processed {min(i+batch_size, n)} of {n} rows...")
            live_table.dataframe(temp_df.head(10))
        progress.empty()
        status.success("Enrichment complete!")
        return enriched_col

    if st.button("Run Enrichment with AI"):
        if not user_prompt.strip():
            st.error("Please enter a prompt for enrichment.")
        elif not selected_cols:
            st.error("Please select at least one column to enrich.")
        elif not output_col.strip():
            st.error("Please enter a name for the output column.")
        else:
            st.info("Running enrichment... This may take a moment.")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            print(user_prompt)
            enriched_col = loop.run_until_complete(
                enrich_dataframe(df, user_prompt, model, batch_size, selected_cols, output_col)
            )
            df[output_col] = enriched_col
            st.success("Enrichment complete!")
            st.write("### Preview of Enriched Data:")
            st.dataframe(df.head())
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Enriched CSV", csv, "enriched_data.csv", "text/csv")
else:
    st.info("Please upload a CSV file to get started.")

st.markdown("---")
st.caption("Built with ❤️ by American Data Science | Powered by AI21 Labs")

# Note: The AI21 API key must be set as the environment variable AI21_API_KEY on the server.