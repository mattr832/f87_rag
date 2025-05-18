import os
import gspread
import pandas as pd
from google.oauth2.service_account import Credentials
import streamlit as st
from dotenv import load_dotenv
from datasets import Dataset, Features, Value, Sequence
from langchain.chat_models import ChatOpenAI
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

scope = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# Set your OpenAI API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load from Streamlit secrets to assess Google key
service_account_info = st.secrets["gcp_service_account"]
creds = Credentials.from_service_account_info(service_account_info, scopes=scope)
client = gspread.authorize(creds)

# Open your sheet and read data as df
sheet = client.open("f87_rag_logs").sheet1
data = sheet.get_all_records()
df = pd.DataFrame(data)

# Convert context columns to single column in df
context_list = []

for index in df.iterrows():
    cont = df.iloc[index[0],3:6].to_list()
    context_list.append(cont)

df["contexts"] = context_list

# Define schema for Dataset creation
features = Features({
    "question": Value("string"),
    "answer": Value("string"),
    "contexts": Sequence(Value("string"))
})

# Convert to Dataset with schema
ragas_dataset = Dataset.from_pandas(df[["question", "answer", "contexts"]], features=features)

# Designate LLM to use for evaluation
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Run evaluation using ragas.evaluate()
results = evaluate(
    ragas_dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        # context_precision,
        # context_recall,
    ],
    llm=llm,
)

print(results)



