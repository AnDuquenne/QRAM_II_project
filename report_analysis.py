import pdfplumber
from pydantic import BaseModel

from openai import OpenAI

# Load api key from .env file
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=API_KEY)

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

def ask_gpt(prompt):
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


class GammaRequest(BaseModel):
    Analyse: str
    Contextual_factors: str
    Suggested_gamma_very_risk_averse: float
    Suggested_gamma_risk_averse: float
    Suggested_gamma_not_risk_averse: float
    Suggested_value_of_gamma: float
    Practical_implication_of_gamma: str

def ask_gpt_gamma(prompt):
    completion = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a large langage model used to help a finance professional."
                                          " Your goal is to help the user compute the value of the risk aversion for"
                                          " a portfolio optimization. You should provide detailed analysis using the"
                                          " personnal information provided by the user"},
            {"role": "user", "content": "Can you give me: the analysis, the contextual factors, the suggested range of"
                                        " gamma, the suggested value of gamma and the practical implication of gamma?"},
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format=GammaRequest,
    )
    return completion.choices[0].message

# Use the extracted PDF text
# response = ask_gpt(pdf_text)
# print(response)

# response = ask_gpt_gamma("Considering that I am a 25 years old student, I woud like to invest 10,000$."
#                          " I have no house, and consider myself as not too risk averse."
#                          " What value of gamma would you give me?")
# print(response)
#
# # print a parsed response as json
# print(response.json())