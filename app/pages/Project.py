import streamlit as st
st.title('Project Page')

tab_architecture, tab_data, tab_constraints, tab_views, tab_maths = st.tabs(["Architecture", "Data", "Constraints", "Views", "Mathematical fundations"])

with tab_architecture:

    st.image("app/io/QRAM.png")

    with st.expander("Data & Situation"):
        st.markdown('''
        This project consists in creating a daily optimized portfolio.\n
        We are working on:
        - Historical data: daily return for the 2023 year
        - Financial data: Stocks composing the Dow jones, bonds, and the principal cryptocurrencies.
        
        Therefore, the mean and variance are computed over this period.
        
        Be careful that inputs will have to be consistent with these hypothesis.''')


    with st.expander("Constraints"):
        st.markdown('''
        This framework allows the user to chose from a set of constraints:
        - The portfolio is fully invested
        - No short-selling is authorized
        - A credit-rating constraint''')

    with st.expander("Views"):
        st.markdown('''
        Views can be defined in three ways:
        - A user input (high priority)
        - Financial report analysis (medium priority)
        - AI return prediction (low priority)
        ''')

with tab_data:
    st.markdown('''
    This application allow the user to chose any combination using the following financial products.
    ''')
    with st.expander("Dow jones"):
        st.markdown('''
        The Dow Jones Industrial Average (DJIA), commonly known as the Dow, is a stock market index that tracks 30 prominent U.S. companies across various industries, excluding transportation and utilities.
        - 3M (MMM)
        - Amazon.com (AMZN)
        - American Express (AXP)
        - Amgen (AMGN)
        - Apple (AAPL)
        - Boeing (BA)
        - Caterpillar (CAT)
        - Chevron (CVX)
        - Cisco Systems (CSCO)
        - The Coca-Cola Company (KO)
        - The Goldman Sachs Group (GS)
        - The Home Depot (HD)
        - Honeywell International (HON)
        - Intel corp. (INTC)
        - International Business Machines (IBM)
        - Johnson & Johnson (JNJ)
        - JPMorgan Chase & Co. (JPM)
        - McDonald's (MCD)
        - Merck & Co. (MRK)
        - Microsoft (MSFT)
        - Nike (NKE)
        - Procter & Gamble (PG)
        - Salesforce (CRM)
        - Sherwin-Williams (SHW)
        - The Travelers Companies (TRV)
        - UnitedHealth Group (UNH)
        - Verizon Communications (VZ)
        - Visa (V)
        - Walmart (WMT)
        - The Walt Disney Company (DIS)
        ''')

    with st.expander("Bonds"):
        st.markdown('''
        - **^TNX**: represents the CBOE 10-Year Treasury Note Yield Index, which reflects the yield on U.S. 10-year Treasury notes. This index is widely used as a benchmark for long-term interest rates and plays a crucial role in financial markets, influencing various economic factors such as mortgage rates and bond pricing.
        - **^FVX**: represents the CBOE 5-Year Treasury Note Yield Index, which reflects the yield on U.S. 5-year Treasury notes. This index serves as a key indicator of medium-term interest rates, influencing various financial instruments and economic conditions.
        - **^IRX**: represents the CBOE 13-Week Treasury Bill Yield Index, which reflects the yield on U.S. 13-week Treasury bills. This index serves as a key indicator of short-term interest rates and is closely monitored by investors and economists to gauge economic conditions and monetary policy expectations.
        ''')

    with st.expander("Crypto currencies"):
        st.markdown('''
        - BTC-USD: Bitcoin 
        - ETH-USD: Ethereum
        - BNB-USD: Binance coin
        - SOL-USD: Solana
        - ADA-USD: Cardano 
        - XRP-USD: Ripple
        ''')

    with st.expander("Credit ratings"):
        st.markdown('''As of end of year 2023, these were the credit ratings given by S&P:''')
        st.markdown('''
        |Company Name|S&P Rating|
        |--------------------------------------|----------|
        |3M Company|AA-|
        |American Express|BBB+|
        |Amgen|A-|
        |Apple|AA+|
        |Boeing|BBB-|
        |Caterpillar|A|
        |Chevron|AA|
        |Cisco Systems|AA-|
        |Coca-Cola|A+|
        |Dow Inc.|BBB+|
        |Goldman Sachs|BBB+|
        |Home Depot|A|
        |Honeywell International|A|
        |Intel|A+|
        |International Business Machines (IBM)|A-|
        |Johnson & Johnson|AAA|
        |JPMorgan Chase|A-|
        |McDonald's|BBB+|
        |Merck & Co.|A+|
        |Microsoft|AAA|
        |Nike|AA-|
        |Procter & Gamble|AA-|
        |Salesforce|A|
        |Travelers Companies|AA|
        |UnitedHealth Group|A+|
        |Verizon Communications|BBB+|
        |Visa|AA-|
        |Walgreens Boots Alliance|BBB|
        |Walmart|AA|
        |Walt Disney|A-|
        ''')

with tab_constraints:
    st.write("This application offers a set of constraints that the user can tune:")

    with st.expander("Fully invested"):
        st.markdown('''
        Constraining the model to create a portfolio where all the wealth is used. It is mathematically transcribed as:
        ''')
        st.latex(r'''
            \mathbf{1}_n^{\top} x=1
        ''')

    with st.expander("No short-selling"):
        st.markdown('''
        Constraining the model to create a portfolio without short positions. It is mathematically transcribed as:
        ''')
        st.latex(r'''
        x_i \geq 0
        ''')

    with st.expander("Credit rating constraint"):
        st.markdown('''
        This constraint allows the user to add a threshold depending on the credit rating of the companies. This constraint does not involve bonds or crypto currencies.\n
        More specifically, the user can define a subset of the credit rating domain and force the model to allocate a given proportion of the portfolio within this range.\n\n
        The credit ratings are given by Standard & Poor's. Standard & Poor's (S&P) is a leading credit rating agency that evaluates the creditworthiness of entities, including corporations and governments.
        The possible ratings are:
        - **AAA**: Highest quality, extremely strong capacity to meet financial commitments.
        - **AA+, AA, AA-**: High quality, very strong capacity to meet financial commitments.
        - **A+, A, A-**: Upper medium grade, strong capacity to meet financial commitments but somewhat susceptible to economic conditions.
        - **BBB+, BBB, BBB-**: Lower medium grade, adequate capacity to meet financial commitments but more susceptible to economic conditions.
        - **BB+, BB, BB-**: Less vulnerable in the near term but faces major uncertainties or exposure to adverse conditions.
        - **B+, B, B-**: More vulnerable but still has the capacity to meet financial commitments under adverse conditions.
        - **CCC+, CCC, CCC-**: Currently vulnerable and dependent on favorable conditions to meet financial commitments.
        - **CC**: Highly vulnerable and likely to default, or already in default with some prospect of recovery.
        - **C**: Currently highly vulnerable and expected to default with little prospect of recovery.
        - **D**: Defaulted, failed to meet financial obligations.
        ''')
        st.markdown('''
        This can be mathematically transcribed as:
        ''')
        st.latex(r'''
        \mathbf{1}_n^{\top} x_{j \in i} \geq \xi
        ''')
        st.markdown(r'''
        where $j$ are the indexes contained in the range chosen by the user, and $\xi$ is the threshold chosen by the user. 
        ''')

with tab_views:
    st.markdown('''
    Views can be defined in various ways. All views are absolute. Moreover, if two absolute views are defined on a same stock, the one with the highest priority is kept.
    ''')

    with st.expander("User input view (High priority)"):
        st.markdown('''
        The user can define an absolute view for each ticker involved in the portfolio optimization. The user must input a value for mu and omega.
        ''')

    with st.expander("Financial report view (Medium priority)"):
        st.markdown('''
        The user can upload a professional financial analysis to create a view.\n\n
        The model will automatically retrieve the interesting values to create a view from the reports.
        - Actual date
        - Report date
        - Prediction value
        - Prediction value (high)
        - Prediction value (low)
        - Forecasting horizon
        ''')
        st.markdown('''Using this metrics, we construct a view such that:''')
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('''
            - The mean is directly retrieved from the report, as the prediction value.
            - A factor to compute omega is computed using the range [prediction value (high), prediction value (low)]
            ''')
        with col2:
            st.image("app/io/price_range.png")

        st.latex(r'''
        \Omega_\text{scale} = \frac{\text{high}-\text{min}}{\text{mean}}
        ''')

        st.markdown('''
        The mean return is then obtained by computing the equivalent daily return from the forecasting horizon return in the report.\n
        The value of omega that will be used in the Black-Litterman model is the found value for omega scaled to take values within the range of habitual omega values.
        ''')

    with st.expander("Machine learning prediction (Low priority)"):
        st.markdown('''
        A Machine learning algorithm is used to create a view automatically. This algorithm is designed to forecast stocks, and therefore is not used to create views for the bonds and cryptocurrencies.\n
        As stated, the views are created automatically. These views have the lowest priority, which implies that they will be replaced if any other type of view is used for a given stock.\n
        This model is able to create a prediction for the next day return, as well as a confidence interval for this return. Thus, the mean return and the omega are directly given thanks to this algorithm.
        ''')
        st.image("app/io/prediction_vs_target_test.png")
        st.markdown('''
        This is an example of a forecasting using the model. The shaded blue area represent the confidence range, which is used to compute omega.
        Once again the value found using the model will then be scaled to acheive consistent value of omega.
        The computed value of omega will be made greater than for other type of views on purpose, as a machine learning prediction is considered less reliable than an analyst prediction.
        ''')

with tab_maths:

    with st.expander("Black-Litterman model"):
        st.markdown('Explanation of the Black-Litterman model')

    with st.expander("Machine learning algorithm"):
        st.markdown(r'''## The model''')
        st.markdown("We consider the following generative model to express the return forecasting problem:")
        st.latex(r'''
            p\left(\left\{r_{i, T+1}\right\}_{i=1}^{N_{T+1}} \mid \mathcal{F}_T\right)
        ''')
        st.markdown("The forecasting is can be adapted in the following factor model.")
        st.latex(r'''
        \int\left(\prod_{i=1}^{N_{T+1}} p\left(r_{i, T+1} \mid \mathbf{F}_{\mathbf{T}+1}, \mathcal{F}_T\right)\right) p\left(\mathbf{F}_{\mathbf{T}+1} \mid \mathcal{F}_T\right) d \mathbf{F}_{\mathbf{T}+1}
        ''')
        st.markdown(r'''
        Where:
        - $r_{i, t}$ is the return of index $i$ at time $t$
        - $F_t$ are the factors at time $t$
        ''')
        st.markdown(r'''The model is then composed of two parts. First, forecasting the factors using factors history. Then predicting the stock return using the forcasted factors and stock history.''')
        st.markdown(r'''### The factor model''')
        st.latex(r'''
        p\left(\hat{\mathbf{F}}_{\mathrm{T}+1} \mid\left\{\hat{\mathbf{F}}_{\mathrm{t}}\right\}_{t=1}^T\right)
        ''')
        st.markdown(r'''
        The factor model aims at describing the likelihood of the factors at time $T+1$ given information up to time $T$.
        These factors can be anything that could give information about the behavior of a stock. In this case, the factor variables are:
        - **CFE VIX**: The VIX index, also known as the "fear gauge," measures market expectations of near-term volatility conveyed by S&P 500 stock index option prices.
        - **CBOE SKEW**: Measures the perceived risk of extreme negative market moves, reflecting tail risk in S&P 500 options.
        - **ML MOVE**: Tracks the implied volatility of U.S. Treasury bonds, indicating expected fluctuations in bond prices.
        - **RUSSELL 3000**: Represents the performance of the 3,000 largest U.S. companies, encompassing approximately 98% of the investable U.S. equity market.
        - **RUSSELL 3000 GROWTH**: Measures the performance of the growth segment within the Russell 3000, focusing on companies with higher growth potential.
        - **ISHARE SMALL CAP**: Tracks the performance of small-capitalization U.S. stocks, often represented by the iShares Russell 2000 ETF.
        - **ISHARE MID CAP**: Tracks the performance of mid-capitalization U.S. stocks, often represented by the iShares Russell Mid-Cap ETF.
        - **International baselines**: A list of national currencies against the American dollar, As we predict return of S&P500 companies.
        - **Commodities baselines**: A list of commodities etf, such as copper, gold, gas or agriulture index.
        ''')
        st.markdown(r'''### The stock model''')
        st.markdown(r'''The stock model is defined as:''')
        st.latex(r'''
        p\left(r_{\mathbf{i}, \mathbf{T}+1} \mid \hat{\mathbf{F}}_{\mathrm{T}+1},\left\{r_{i, t}\right\}_{t=1}^T\right)
        ''')
        st.markdown('''
        We are looking at the probability of the stock retrun at $T+1$ given our previous prediction using the factor model an the historical returns.
        ''')
