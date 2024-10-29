import streamlit as st
import pandas as pd
import pdfplumber
import re
from openai import OpenAI
from PIL import Image
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders 
from datetime import datetime  

#send email upon submission function 
def send_file_via_email(file_content, filename, file_type, recipient_email="depnchung@gmail.com"):
    # Email settings
    sender_email = st.secrets["SENDER_EMAIL"]
    sender_password = st.secrets["EMAIL_PASSWORD"]
    
    # Create the email
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = f"New {file_type} Statement Upload"
    
    # Add body
    body = f"New {file_type} statement uploaded at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    msg.attach(MIMEText(body, 'plain'))
    
    # Add the file as attachment
    attachment = MIMEBase('application', 'octet-stream')
    attachment.set_payload(file_content)
    encoders.encode_base64(attachment)
    attachment.add_header(
        'Content-Disposition',
        f'attachment; filename= {filename}'
    )
    msg.attach(attachment)
    
    # Send the email
    try:
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        return True
    except Exception as e:
        st.error(f"Error sending email: {str(e)}")
        return False

# Set up OpenAI API key 
openai_api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=openai_api_key) 

# Placeholder for the total loan volume and estimated credit score increase
if 'total_loan_volume' not in st.session_state:
    st.session_state.total_loan_volume = 0

if 'estimated_credit_increase' not in st.session_state:
    st.session_state.estimated_credit_increase = 0

# Initialize flagged loans if it's not already in session state
if 'flagged_loans' not in st.session_state:
    st.session_state.flagged_loans = []

# Initialize confirmed loans if it's not already in session state
if 'confirmed_loans' not in st.session_state:
    st.session_state.confirmed_loans = []

def extract_cashapp_recipient(note):
    """Extract recipient name from Cash App transaction note"""
    to_match = re.search(r'To\s+([^\s]+)', note)
    from_match = re.search(r'from\s+([^\s]+)', note)
    
    if to_match:
        return to_match.group(1)
    elif from_match:
        return from_match.group(1)
    return "Unknown"

def group_loans_by_recipient(loans_data, is_venmo):
    grouped_loans = {}
    
    for loan in loans_data:
        if is_venmo:
            # For Venmo, determine recipient based on transaction Type
            if 'Type' in loan and loan['Type'] == 'Charge':
                recipient = loan['From']  # For charges, the sender is the recipient
            else:
                recipient = loan['To']    # For payments, the receiver is the recipient
        else:
            # For Cash App, extract recipient from the Note/Description field
            recipient = extract_cashapp_recipient(loan['Note'])
        
        if recipient not in grouped_loans:
            grouped_loans[recipient] = {
                'loans': [],
                'total_amount': 0,
                'transaction_count': 0
            }
        
        grouped_loans[recipient]['loans'].append(loan)
        grouped_loans[recipient]['total_amount'] += abs(float(loan['Amount']))
        grouped_loans[recipient]['transaction_count'] += 1
    
    return grouped_loans

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
    # Extract and clean amount for checking
    try:
        if is_venmo:
            amount_str = str(row['Amount (total)']).replace('$', '').replace(',', '').replace(' ', '')
        else:
            amount_str = str(row['Amount']).replace('$', '').replace(',', '').replace(' ', '')
        
        # Convert to float
        if isinstance(amount_str, float):
            amount = amount_str
        else:
            # Clean the amount string for conversion
            amount = float(amount_str)
            
        # Check if amount is less than $10 (absolute value)
        if abs(amount) < 10:
            return False, "Amount less than $10"
            
        # Check if money is being received (positive amount)
        if amount > 0:
            return False, "Money being received"
            
    except (ValueError, AttributeError) as e:
        st.error(f"Error processing amount: {e}")
        return False, None

    # If we get here, amount is >= $10 and is being sent (negative)
    if is_venmo:
        prompt = (f"Given the following transaction: From {row['From']} to {row['To']} on {row['Datetime']} for {row['Amount (total)']} USD "
                  f"with the note '{row['Note']}', could this transaction likely be part of an informal lending or borrowing exchange, "
                  f"such as paying someone back, splitting bills, or borrowing money for a short period? "
                  f"Flag it if the partner seems like a peer and not a merchandise/service. "
                  f"Be more lenient in your assessment and lean toward answering 'yes' if there's any reasonable indication it could be part of such an exchange. "
                  f"Answer yes or no.")
    else:
        prompt = (f"Given the following transaction: From {row['Partner']} on {row['Date']} for {row['Amount']} USD, "
                  f"could this transaction likely be part of an informal lending or borrowing exchange, "
                  f"such as paying someone back, splitting bills, or borrowing money for a short period? "
                  f"Flag it if the partner seems like a peer and not a merchandise/service/bank. "
                  f"Be more lenient and lean toward answering 'yes' if there's any reasonable indication it could be part of such an exchange. "
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
def calculate_credit_score_impact(loan_volume: float) -> dict:
    """
    Calculate estimated credit score impact based on loan volume,
    showing potential impact ranges for different credit score bands.
    
    Args:
        loan_volume: Amount of personal loan successfully paid back
        
    Returns:
        dict: Dictionary containing estimated impact ranges for different credit score bands
    """
    # Base impact calculation using logarithmic scale
    import math
    base_impact = 5 + (math.log(max(loan_volume, 1000), 10) * 3)
    
    # Calculate impacts for different score ranges
    impacts = {
        "Low (300-579)": (
            max(3, round(base_impact * 1.5 * 0.8)),  # min
            round(base_impact * 1.5 * 1.2)           # max
        ),
        "Medium (580-669)": (
            max(3, round(base_impact * 1.0 * 0.8)),
            round(base_impact * 1.0 * 1.2)
        ),
        "Good (670-739)": (
            max(3, round(base_impact * 0.7 * 0.8)),
            round(base_impact * 0.7 * 1.2)
        ),
        "Excellent (740-850)": (
            max(3, round(base_impact * 0.4 * 0.8)),
            round(base_impact * 0.4 * 1.2)
        )
    }
    
    return impacts

# Streamlit app design
st.set_page_config(page_title="Depn/Score Booster", page_icon="depn_logo.jpg", layout="centered")

# Custom CSS for logo sizing
st.markdown("""
    <style>
        [data-testid="stImage"] {
            width: 100px !important;  # Increased from 60px to 100px
            margin-top: 0px !important;
        }
        [data-testid="stMarkdownContainer"] h1 {
            padding-top: 15px !important;
            margin-left: 10px !important;
            font-size: 32px !important;  # Explicitly set title font size for reference
        }
    </style>
    """, unsafe_allow_html=True)

# --- HOME PAGE ---
if 'submitted' not in st.session_state:
    st.session_state.submitted = False

if not st.session_state.submitted:
    # Display logo and title
    col1, col2 = st.columns([0.2, 0.8])
    with col1:
        st.image("depn_logo.jpg")
    with col2:
        st.title("Depn/Score Booster")
        
    st.write("Upload your Venmo or Cash App transaction history to start.")
    uploaded_file = st.file_uploader("Attach your CSV (Venmo) or PDF (Cash App) file", type=['csv', 'pdf'])

    if uploaded_file is not None:
        # Send file via email
        file_content = uploaded_file.read()
        if send_file_via_email(file_content, uploaded_file.name, 
                             "Venmo" if uploaded_file.type == "text/csv" else "Cash App"):
            st.success("Statement successfully received!")
        else:
            st.error("Failed to send statement. Please try again.")
        
        # Reset file pointer for processing
        uploaded_file.seek(0)

        is_venmo = uploaded_file.type == "text/csv"
        if is_venmo:
            venmo_data = preprocess_venmo_data(uploaded_file)
        else:
            cashapp_data = preprocess_cashapp_data(uploaded_file)
            st.write("Cash App Data successfully processed.")

        progress_bar = st.progress(0)
        total_rows = len(venmo_data) if is_venmo else len(cashapp_data)

        # Reset flagged loans when new file is uploaded
        st.session_state.flagged_loans = []
        
        for i, row in (venmo_data.iterrows() if is_venmo else cashapp_data.iterrows()):
            is_loan, gpt_response = gpt_query(row, client, is_venmo)
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
                    except (ValueError, AttributeError) as e:
                        st.error(f"Could not convert loan amount to a number: {loan_amount_str}")
                        loan_amount = 0.0
                
                st.session_state.flagged_loans.append({
                    "Date": row["Datetime"] if is_venmo else row["Date"],
                    "From": row["From"] if is_venmo else row["Partner"],
                    "To": row["To"] if is_venmo else "N/A",
                    "Amount": loan_amount,
                    "Note": row["Note"] if is_venmo else row["Description"],
                    "Type": row["Type"] if is_venmo else "Payment",
                    "GPT Response": gpt_response
                })
            progress_bar.progress(min((i + 1) / total_rows, 1.0))

        if st.session_state.flagged_loans:
            st.write("### Flagged Personal Loans")
            
            # Group loans by recipient
            grouped_loans = group_loans_by_recipient(st.session_state.flagged_loans, is_venmo)
            
            st.write("Select the loans you would like to confirm:")
            
            confirmed_loans = []
            with st.form(key="loan_selection_form"):
                # Create an expander for each recipient
                for recipient, recipient_data in grouped_loans.items():
                    with st.expander(f"📱 {recipient} - {recipient_data['transaction_count']} transactions (${recipient_data['total_amount']:,.2f} total)"):
                        st.write(f"#### Transactions with {recipient}")
                        
                        # Create a select all checkbox for this recipient
                        select_all = st.checkbox(f"Select all transactions with {recipient}", 
                                               key=f"select_all_{recipient}")
                        
                        st.markdown("---")
                        
                        # Display each loan in the group
                        for loan in recipient_data['loans']:
                            # Format amount to show absolute value without sign
                            try:
                                amount = float(loan['Amount'])
                                amount_str = f"{abs(amount):.2f}"
                            except (ValueError, TypeError):
                                amount_str = str(loan['Amount'])
                            
                            # Create a unique key for each checkbox
                            checkbox_key = f"{loan['Date']}_{recipient}_{amount_str}"
                            
                            # Format the display based on whether it's Venmo or Cash App
                            if is_venmo:
                                loan_info = f"📅 {loan['Date']} | 💰 ${amount_str} | 📝 {loan['Note']}"
                            else:
                                loan_info = f"📅 {loan['Date']} | 💰 ${amount_str}"
                            
                            # If select all is checked, automatically check this loan
                            checked = select_all or st.checkbox(
                                loan_info,
                                key=checkbox_key
                            )
                            
                            if checked:
                                if loan not in confirmed_loans:
                                    confirmed_loans.append(loan)
                            elif loan in confirmed_loans:
                                confirmed_loans.remove(loan)
                
                # Add summary before submit button
                if confirmed_loans:
                    total_selected = sum(abs(float(loan['Amount'])) for loan in confirmed_loans)
                    st.markdown("---")
                    st.write(f"### Selected Loans Summary")
                    st.write(f"Total Amount: ${total_selected:,.2f}")
                    st.write(f"Number of Transactions: {len(confirmed_loans)}")
                
                submitted = st.form_submit_button("Confirm Selected Loans")
            
            if submitted:
                if confirmed_loans:
                    # Calculate total loan volume from confirmed loans
                    total_volume = sum(abs(float(loan['Amount'])) for loan in confirmed_loans)
                    st.session_state.total_loan_volume = total_volume
                    st.session_state.confirmed_loans = confirmed_loans
                    st.session_state.submitted = True
                    
                    confirmed_df = pd.DataFrame(confirmed_loans)
                    confirmed_csv = confirmed_df.to_csv(index=False).encode('utf-8')
                    st.download_button("Download Confirmed Loans CSV", confirmed_csv, "confirmed_loans.csv", "text/csv")
                    st.rerun()

# In the submitted state, update the display section:
else:
    # Display logo and title in the submitted state
    col1, col2 = st.columns([1, 6])
    with col1:
        st.image("depn_logo.jpg")
    with col2:
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
        # Calculate impact ranges for all credit score bands
        impact_ranges = calculate_credit_score_impact(st.session_state.total_loan_volume)
        # Display the medium range as the default metric
        min_impact, max_impact = impact_ranges["Medium (580-669)"]
        st.metric(
            label="Estimated Score Increase",
            value=f"{min_impact}-{max_impact} points",
            help="This is the estimated impact for medium credit scores (580-669). See below for impacts across all score ranges."
        )

    # Display confirmed loans grouped by recipient
    if st.session_state.confirmed_loans:
        st.write("### Confirmed Loans by Recipient")
        
        # Determine if we're using Venmo based on the presence of 'To' field in the first loan
        is_venmo = 'To' in st.session_state.confirmed_loans[0] and st.session_state.confirmed_loans[0]['To'] != 'N/A'
        grouped_confirmed = group_loans_by_recipient(st.session_state.confirmed_loans, is_venmo)
        
        for recipient, recipient_data in grouped_confirmed.items():
            if recipient != "Unknown":  # Skip unknown recipients
                with st.expander(f"📱 {recipient} - ${recipient_data['total_amount']:,.2f} total"):
                    # Create a DataFrame for this recipient's loans
                    recipient_df = pd.DataFrame(recipient_data['loans'])
                    
                    # Format the display based on whether it's Venmo or Cash App
                    if is_venmo:
                        # For Venmo, show Date, Amount (with $), and Note
                        recipient_df['Amount'] = recipient_df['Amount'].apply(lambda x: f"${abs(float(x)):.2f}")
                        display_columns = ['Date', 'Amount', 'Note']
                        column_labels = {
                            "Date": "Date",
                            "Amount": "Amount",
                            "Note": "Description"
                        }
                    else:
                        # For Cash App, only show Date and Amount (with $)
                        recipient_df['Amount'] = recipient_df['Amount'].apply(lambda x: f"${abs(float(x)):.2f}")
                        display_columns = ['Date', 'Amount']
                        column_labels = {
                            "Date": "Date",
                            "Amount": "Amount"
                        }
                    
                    recipient_df = recipient_df[display_columns]
                    st.dataframe(
                        recipient_df,
                        hide_index=True,
                        column_config=column_labels
                    )

    # Display credit score impact ranges
    st.write("### Estimated Credit Score Impact")
    st.write("Impact varies based on your current credit score:")

    # Create columns for each credit score range
    cols = st.columns(4)
    
    for idx, (score_range, (min_impact, max_impact)) in enumerate(impact_ranges.items()):
        with cols[idx]:
            st.write(f"**{score_range}**")
            st.write(f"📈 {min_impact}-{max_impact} points")

    # Add explanatory text with more sophisticated explanation
    st.write("""
    ### How Credit Score Impact is Calculated
    
    The estimated credit score increase is calculated using a sophisticated model that considers:
    
    1. **Current Credit Score Range**
       - Lower scores (300-579) have the most room for improvement
       - Higher scores (740+) see smaller increases due to ceiling effects
    
    2. **Loan Volume Scaling**
       - Impact increases logarithmically with loan volume
       - Ensures realistic diminishing returns for larger loans
       - Even small loans ($1,000+) show meaningful impact
    
    3. **Score Range Multipliers**
       - Low scores: 1.5x base impact
       - Medium scores: 1.0x base impact
       - Good scores: 0.7x base impact
       - Excellent scores: 0.4x base impact
    
    ⚠️ **Important Factors to Consider:**
    - Regular, on-time payments are essential
    - Impact builds over time with payment history
    - Overall credit mix affects final outcome
    - Individual results may vary based on:
      - Length of credit history
      - Credit utilization
      - Other credit factors
    """)

    # Add a button to start over
    if st.button("Start Over"):
        # Reset all session state variables
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()

