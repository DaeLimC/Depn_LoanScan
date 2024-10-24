import streamlit as st
import pandas as pd
import pdfplumber
import re
from openai import OpenAI

# Set up OpenAI API key (replace with your key)
openai_api_key = "sk-pHmJU9guOKG8Lu4iscqRoNZtgCiKp3BhVNsYlMA0cET3BlbkFJ4teibEEEuvAsiuSubOfQuFU4wdtvGUQKnJEswnt_wA"
client = OpenAI(api_key=openai_api_key)

# Placeholder for the total loan volume and estimated credit score increase
if 'total_loan_volume' not in st.session_state:
    st.session_state.total_loan_volume = 0

if 'estimated_credit_increase' not in st.session_state:
    st.session_state.estimated_credit_increase = 0

# Initialize flagged loans if it's not already in session state
if 'flagged_loans' not in st.session_state:
    st.session_state.flagged_loans = []

# Function to preprocess Venmo CSV file
def preprocess_venmo_data(data):
    venmo_data = pd.read_csv(data)
    venmo_cleaned = venmo_data.dropna(how='all')
    venmo_cleaned.columns = venmo_cleaned.iloc[1]
    venmo_cleaned = venmo_cleaned.drop([0, 1, 2])
    venmo_cleaned = venmo_cleaned.reset_index(drop=True)
    relevant_columns = ['ID', 'Datetime', 'Type', 'Status', 'Note', 'From', 'To', 'Amount (total)']
    venmo_cleaned = venmo_cleaned[relevant_columns]
    return venmo_cleaned

# Function to preprocess Cash App PDF file and convert it to a DataFrame
def preprocess_cashapp_data(pdf_file):
    transactions_data = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if "Transactions" in text:
                transactions_section = text.split('Transactions')[1]
            else:
                transactions_section = text
            transactions_lines = transactions_section.split('\n')
            for line in transactions_lines:
                if line.strip():
                    parts = line.split(' ')
                    if len(parts) > 2:
                        date = ' '.join(parts[:2])
                        amount = parts[-1].replace('$', '').replace(',', '')
                        if 'From' in line:
                            amount = f"+{amount}"
                        else:
                            amount = f"-{amount}"
                        description = ' '.join(parts[2:-1])
                        description_clean = re.sub(r'(Cash App payment|Standard transfer)', '', description).strip()
                        partner = None
                        match_between_to_from = re.search(r'To\s(.+?)\sfrom', description_clean)
                        match_after_to = re.search(r'To\s([\w\s]+)', description_clean)
                        match_after_from = re.search(r'From\s([\w\s]+)', description_clean)
                        if match_between_to_from:
                            partner = match_between_to_from.group(1).strip()
                        elif match_after_to:
                            partner = match_after_to.group(1).strip()
                        elif match_after_from:
                            partner = match_after_from.group(1).strip()
                        else:
                            partner = "Unknown"
                        transactions_data.append([date, description_clean, amount, partner])
    cashapp_cleaned = pd.DataFrame(transactions_data, columns=['Date', 'Description', 'Amount', 'Partner'])
    def is_valid_transaction(row):
        return row['Date'].strip().split()[0] in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    cashapp_cleaned = cashapp_cleaned[cashapp_cleaned.apply(is_valid_transaction, axis=1)]
    cashapp_cleaned['Amount'] = pd.to_numeric(cashapp_cleaned['Amount'], errors='coerce')
    return cashapp_cleaned

# Function to query GPT for each row
def gpt_query(row, client, is_venmo=True):
    if is_venmo:
        prompt = (f"Given the following transaction: From {row['From']} to {row['To']} on {row['Datetime']} for {row['Amount (total)']} USD "
                  f"with the note '{row['Note']}', could this transaction likely be part of an informal lending or borrowing exchange, "
                  f"such as paying someone back, splitting bills, or borrowing money for a short period? "
                  f"Flag it if the partner seems like a peer and not a merchandise/service. "
                  f"Be more lenient in your assessment and lean toward answering 'yes' if there’s any reasonable indication it could be part of such an exchange. "
                  f"Answer yes or no.")
    else:
        prompt = (f"Given the following transaction: From {row['Partner']} on {row['Date']} for {row['Amount']} USD, "
                  f"could this transaction likely be part of an informal lending or borrowing exchange, "
                  f"such as paying someone back, splitting bills, or borrowing money for a short period? "
                  f"Flag it if the partner seems like a peer and not a merchandise/service/bank. "
                  f"Be more lenient and lean toward answering 'yes' if there’s any reasonable indication it could be part of such an exchange. "
                  f"Answer yes or no.")
    try:
        full_response = ""
        for chunk in client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}],
            stream=True,
        ):
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
        return 'yes' in full_response.lower(), full_response
    except Exception as e:
        st.error(f"Error querying GPT: {e}")
        return False, None

#Function for calculating credit score impact 
def calculate_credit_score_impact(loan_volume):
    """
    Calculate estimated credit score impact based on loan volume.
    Returns a sensible range where max is always > min.
    """
    # Base calculations
    loan_amount_factor = min(1.0, loan_volume / 50000)  # Scale based on loan size
    
    # Minimum increase (conservative estimate)
    min_increase = max(5, 10 * loan_amount_factor)  # At least 5 points, up to 10
    
    # Maximum increase (optimistic estimate)
    max_increase = max(min_increase + 5, 20 * loan_amount_factor)  # Always at least 5 points more than min
    
    # Round to whole numbers
    return round(min_increase), round(max_increase)

# Streamlit app design
st.set_page_config(page_title="Credit Bump", page_icon="💳", layout="centered")

# --- HOME PAGE ---
if 'submitted' not in st.session_state:
    st.session_state.submitted = False

if not st.session_state.submitted:
    st.title("💳 Credit Bump?")
    st.write("Upload your Venmo or Cash App transaction history to start.")
    uploaded_file = st.file_uploader("Attach your CSV (Venmo) or PDF (Cash App) file", type=['csv', 'pdf'])

    if uploaded_file is not None:
        is_venmo = uploaded_file.type == "text/csv"
        if is_venmo:
            venmo_data = preprocess_venmo_data(uploaded_file)
        else:
            cashapp_data = preprocess_cashapp_data(uploaded_file)
            st.write("Cash App Data successfully processed.")

        progress_bar = st.progress(0)
        total_rows = len(venmo_data) if is_venmo else len(cashapp_data)

        for i, row in (venmo_data.iterrows() if is_venmo else cashapp_data.iterrows()):
            is_loan, gpt_response = gpt_query(row, client, is_venmo)
            # Replace the problematic section with this:
            if is_loan:
                loan_amount_str = row["Amount (total)"] if is_venmo else row["Amount"]
                
                # Handle the case where loan_amount_str is already a float
                if isinstance(loan_amount_str, float):
                    loan_amount = loan_amount_str
                else:
                    # Clean the amount string for conversion
                    try:
                        loan_amount_cleaned = str(loan_amount_str).replace('$', '').replace(' ', '').replace('+', '').replace(',', '').strip()
                        loan_amount = float(loan_amount_cleaned) if '-' not in str(loan_amount_str) else -float(loan_amount_cleaned)
                    except (ValueError, AttributeError):
                        st.error(f"Could not convert loan amount to a number: {loan_amount_str}")
                        loan_amount = 0.0  # Assign a default value if conversion fails
                
                st.session_state.total_loan_volume += abs(loan_amount)
                
                st.session_state.flagged_loans.append({
                    "Date": row["Datetime"] if is_venmo else row["Date"],
                    "From": row["From"] if is_venmo else row["Partner"],
                    "To": row["To"] if is_venmo else "N/A",
                    "Amount": loan_amount,
                    "Note": row["Note"] if is_venmo else row["Description"],
                    "GPT Response": gpt_response
                })
            progress_bar.progress(min((i + 1) / total_rows, 1.0))

        if st.session_state.flagged_loans:
            st.write("### Flagged Personal Loans")
            flagged_df = pd.DataFrame(st.session_state.flagged_loans)
            st.write("Select the loans you would like to confirm:")

            confirmed_loans = []
            with st.form(key="loan_selection_form"):
                for idx, row in flagged_df.iterrows():
                    if is_venmo:
                        # Venmo format as per your screenshot
                        checkbox_label = f"From {row['From']} to {row['To']} | {row['Amount']} USD | {row['Note']}"
                    else:
                        # Cash App format
                        checkbox_label = f"{row['Date']} | {row['From']} | {row['Amount']} USD"
                    
                    if st.checkbox(checkbox_label, key=idx):
                        confirmed_loans.append(row)

                submitted = st.form_submit_button("Submit Confirmed Loans")

            if submitted:
                if confirmed_loans:
                    st.session_state.submitted = True
                    confirmed_df = pd.DataFrame(confirmed_loans)
                    confirmed_csv = confirmed_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Confirmed Loans CSV", confirmed_csv, "confirmed_loans.csv", "text/csv")
                    #st.experimental_rerun()
                    # Trigger a "rerun" by setting a query parameter here after submission
                    #st.experimental_set_query_params(submitted="true")
                    st.rerun() 
                    

else:
 
    #####This is the VISUAL VER########
    # New page after submission
    # st.title("Submitted!")
    # st.write("### Summary of Personal Loans")

    # # Create two columns for the metrics
    # col1, col2 = st.columns(2)

    # with col1:
    #     st.metric(
    #         label="Total Loan Volume",
    #         value=f"${st.session_state.total_loan_volume:,.2f}"
    #     )

    # with col2:
    #     estimated_increase = st.session_state.total_loan_volume / 100 * 1.0
    #     st.metric(
    #         label="Estimated Score Increase",
    #         value=f"{estimated_increase:.1f} points"
    # )
        
    # In your Streamlit app:
    st.title("Submitted!")
    st.write("### Summary of Personal Loans")

    # Create columns for the metrics
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Total Loan Volume",
            value=f"${st.session_state.total_loan_volume:,.2f}"
        )

    with col2:
        min_increase, max_increase = calculate_credit_score_impact(st.session_state.total_loan_volume)
        st.metric(
            label="Estimated Score Increase",
            value=f"{min_increase}-{max_increase} points",
            help="This estimate assumes regular on-time payments and is based on typical FICO score factors. Actual results may vary based on your credit profile."
        )

    # Add explanatory text
    st.write("""
    ### How This Estimate Works
    The estimated credit score increase is based on typical FICO score impacts from personal loans:

    - **Small Loans** ($1-$10k): 5-15 points
    - **Medium Loans** ($10k-$25k): 10-20 points
    - **Large Loans** ($25k+): 15-30 points

    ⚠️ **Note:** Actual score increases depend on:
    - Making regular, on-time payments
    - Your current credit score
    - Overall credit mix
    - Credit utilization
    - Length of credit history
    """)

