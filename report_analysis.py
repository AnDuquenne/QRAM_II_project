import pdfplumber
from pydantic import BaseModel

from openai import OpenAI

# Load api key from .env file
from dotenv import load_dotenv
import os


def extract_text_with_pdfplumber(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text


class ReportRequest(BaseModel):
    Stock_analysed: str
    Ticker: str
    Report_date: str
    Company_writing_report: str
    Actual_price: float
    Expected_price: float
    Min_expected_price: float
    Max_expected_price: float
    Forecasting_horizon: int
    Currency: str


def ask_gpt(prompt, ak=None):

    if ak:
        API_KEY = ak
    else:
        load_dotenv()
        API_KEY = os.getenv("OPENAI_API_KEY")

    client = OpenAI(api_key=API_KEY)

    completion = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a large langage model designed to help a finance professional"
                                          " summarize financial reports. Your role is to extract the key information"
                                          " from the report and provide it to the user in a structured way."},
            {"role": "user", "content": "Can you give me: the stock analyzed, the ticker of the stock (it sould be in "
                                        "this list (INTC, AAPL, MSFT, AMZN, WMT, JPM, V, UNH, HD, PG, JNJ, CRM, CVX,"
                                        " KO, MRK, CSCO, MCD, AXP, IBM, GS, CAT, DIS, VZ, AMGN, HON, NKE, BA,"
                                        " SHW, MMM, TRV, NVDA), the report date (format: YYYY-MM-DD), the company"
                                        " writting the report, the expected price of the analyzed stock, the minimal"
                                        " expected price of the analyzed stock (it should be different from the mean"
                                        " expected price), the maximal expected price of the analyzed stock (it should"
                                        " be different from the mean expected price), the  forecasting horizon"
                                        " (in days)  and the currency of the report?"},
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format=ReportRequest,
    )
    return completion.choices[0].message
