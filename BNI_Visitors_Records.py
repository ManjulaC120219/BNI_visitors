import streamlit as st

# Configure Streamlit page - MUST be first Streamlit command
st.set_page_config(
    page_title="BNI Brilliance Visitors Register",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Safe imports (these don't contain Streamlit commands)
import pandas as pd
from supabase import create_client, Client
from datetime import datetime, timedelta, date
import time
import logging
from typing import Optional, Dict, Any, Tuple
import traceback
from datetime import date
import calendar

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import modules that might contain Streamlit commands AFTER set_page_config
from BNI_Visitors_data_extraction import extract_data_from_image_v2

# Supabase configuration
supabase_url = st.secrets["SUPABASE_URL"]
supabase_key = st.secrets["SUPABASE_KEY"]


# Error handling utilities
class AppError(Exception):
    """Base exception for application errors"""
    pass

class DatabaseError(AppError):
    """Database related errors"""
    pass

class ValidationError(AppError):
    """Input validation errors"""
    pass

class AuthenticationError(AppError):
    """Authentication related errors"""
    pass
# Enhanced error tracking
def increment_error_count():
    """Track errors for monitoring"""
    if 'error_count' not in st.session_state:
        st.session_state.error_count = 0
    st.session_state.error_count += 1
    st.session_state.last_error = datetime.now()

def init_session_state():
    """Initialize session state variables"""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'error_count' not in st.session_state:
        st.session_state.error_count = 0


def show_error_details(error_message):
    """Display error details in a user-friendly way"""
    with st.expander("🔍 Error Details"):
        st.code(error_message)

def upload_and_process_image(uploaded_file):
    """Upload image and process it to extract table data."""
    # Save uploaded file to a temporary location
    with open("temp_image.png", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract data using the function from data_extraction.py
    extracted_data = extract_data_from_image_v2("temp_image.png")

    return extracted_data

def handle_error(func):
    """Decorator for error handling"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except AuthenticationError as e:
            st.error(f"Authentication failed: {str(e)}")
            logger.error(f"Authentication error in {func.__name__}: {e}")
            return None
        except ValidationError as e:
            st.error(f"Invalid input: {str(e)}")
            logger.error(f"Validation error in {func.__name__}: {e}")
            return None
        except DatabaseError as e:
            st.error(f"Database error: {str(e)}")
            logger.error(f"Database error in {func.__name__}: {e}")
            return None
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            return None
    return wrapper


def validate_input(field_name: str, value: Any, required: bool = False, min_length: int = 0) -> bool:
    """Validate input fields"""
    if required and (value is None or str(value).strip() == ""):
        raise ValidationError(f"{field_name} is required")

    if isinstance(value, str) and len(value.strip()) < min_length:
        raise ValidationError(f"{field_name} must be at least {min_length} characters")

    return True

# Initialize Supabase client with error handling
@st.cache_resource
def init_supabase():
    try:

        client = create_client(supabase_url, supabase_key)
        # Test connection
        client.table('bni_visitors_details').select('id').limit(1).execute()
        return client
    except Exception as e:
        logger.error(f"Supabase initialization failed: {e}")
        raise DatabaseError(f"Failed to connect to database: {str(e)}")

try:
    supabase: Client = init_supabase()
except DatabaseError:
    st.error("❌ Database Connection Failed")
    st.info("Please check your Supabase credentials and network connection.")
    st.code("""
    Troubleshooting steps:
    1. Verify Supabase URL and API key
    2. Check network connectivity
    3. Ensure database tables exist
    4. Check Supabase service status
    """)
    st.stop()

# Custom CSS
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }

    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        text-align: center;
        margin-bottom: 1rem;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }

    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }

    .delete-confirmation {
        background-color: #fee;
        border: 1px solid #fcc;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }

    .success-message {
        background-color: #efe;
        border: 1px solid #cfc;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
    }

    .error-container {
        background-color: #fee;
        border-left: 4px solid #f56565;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        transition: transform 0.2s;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
    }

    .sidebar .stSelectbox > div > div {
        background-color: #f0f2f6;
    }
</style>
""", unsafe_allow_html=True)


def save_data_to_supabase(dataframe: pd.DataFrame,selected_date=None):
    """Save the edited dataframe back to Supabase."""
    success_count = 0
    error_count = 0
    errors = []

    if selected_date is None:
        selected_date = date.today()

    # Combine selected date with current time for timestamp
    save_datetime = datetime.combine(selected_date, datetime.now().time())

    try:
        # Normalize column names (handle both "Name" and "name")
        dataframe.columns = dataframe.columns.str.lower().str.strip()

        for _, row in dataframe.iterrows():
            # Clean and convert data types
            def safe_convert_to_int(value):
                """Convert value to int, return None if empty or invalid"""
                if pd.isna(value) or value == "" or value is None:
                    return None
                try:
                    return int(float(str(value)))  # Handle float strings like "123.0"
                except (ValueError, TypeError):
                    return None

            def safe_convert_to_string(value):
                """Convert value to string, return empty string if None/NaN"""
                if pd.isna(value) or value is None:
                    return None
                return str(value).strip()

            def safe_convert_to_date_string(value, date_format=None):
                """Convert a value to a date string in YYYY-MM-DD format"""
                if pd.isna(value) or value in ("", None):
                    return None

                # If it's already a datetime object or pandas Timestamp
                if isinstance(value, (datetime, pd.Timestamp)):
                    return value.strftime('%Y-%m-%d')

                # If it's a date object
                if isinstance(value, date):
                    return value.strftime('%Y-%m-%d')

                # If it's already a string in the right format
                if isinstance(value, str) and len(value) == 10 and '-' in value:
                    try:
                        datetime.strptime(value, '%Y-%m-%d')
                        return value
                    except ValueError:
                        pass

                try:
                    if date_format:
                        parsed_date = datetime.strptime(str(value), date_format)
                        return parsed_date.strftime('%Y-%m-%d')
                    else:
                        parsed_date = pd.to_datetime(value, errors='coerce')
                        if pd.isna(parsed_date):
                            return None
                        return parsed_date.strftime('%Y-%m-%d')
                except (ValueError, TypeError):
                    return None

            # Use normalized lowercase column names
            member_data = {
                "name": safe_convert_to_string(row.get("name", "")),
                "company_name": safe_convert_to_string(row.get("company name", "")),
                "invited_by": safe_convert_to_string(row.get("invited by", "")),
                "fees": safe_convert_to_int(row.get("fees", 0)),
                "payment_mode": safe_convert_to_string(row.get("payment mode", "")),
                "note": safe_convert_to_string(row.get("note", "")),
                "created_at": save_datetime.isoformat(),
            }

            # Skip rows with no name
            if not member_data["name"]:
                continue

            # Check if record already exists
            existing = supabase.table("bni_visitors_details").select("*").eq("name", member_data["name"]).eq(
                "created_at", member_data["created_at"]).execute()

            if existing.data:
                # Update existing record
                response = supabase.table("bni_visitors_details").update(member_data).eq("id", existing.data[0][
                    'id']).execute()
            else:
                # Insert new record
                response = supabase.table("bni_visitors_details").insert(member_data).execute()

            # Check if the response contains data (success)
            if response.data:
                success_count += 1
            else:
                error_count += 1
                errors.append(f"No data returned for {member_data['name']}")

        # Show summary message after all operations
        if success_count > 0:
            st.success(f"✅ Successfully saved {success_count} visitor(s) to database!")

        if error_count > 0:
            st.error(f"❌ Failed to save {error_count} visitor(s). Errors: {'; '.join(errors)}")

    except Exception as e:
        st.error(f"❌ Database operation failed: {str(e)}")
        print(f"Debug - Full error: {e}")
        import traceback
        print(f"Debug - Traceback: {traceback.format_exc()}")

def debug_supabase_connection():
    """Debug function to test Supabase connection"""
    try:
        supabase = init_supabase()
        if not supabase:
            return "❌ Supabase client is None"

        # Test basic connection
        response = supabase.table("bni_visitors_details").select("*").limit(1).execute()
        return f"✅ Connection OK. Sample data: {response.data}"

    except Exception as e:
        return f"❌ Connection failed: {str(e)}"

# Alternative simplified save function for testing
def save_data_simple_test(dataframe: pd.DataFrame):
    """Simplified version for testing with correct schema"""
    try:
        supabase = init_supabase()

        # Helper functions
        def safe_int_convert(value):
            if pd.isna(value) or value == "" or value is None:
                return None
            try:
                return int(str(value))
            except (ValueError, TypeError):
                return None

        def safe_date_convert(value, date_format=None):
            """
            Convert a value to a datetime object.

            - If `date_format` is provided, it will use that format to parse the date.
            - If the value is empty, None, or not a valid date, return None.
            - Handles strings, pandas Timestamps, and datetime objects.
            """
            if pd.isna(value) or value in ("", None):
                return None

            # If it's already a datetime object or pandas Timestamp, return as datetime
            if isinstance(value, (datetime, pd.Timestamp)):
                return value.to_pydatetime()

            try:
                if date_format:
                    return datetime.strptime(str(value), date_format)
                else:
                    return pd.to_datetime(value, errors='coerce').to_pydatetime()
            except (ValueError, TypeError):
                return None

        # Test with just one record
        first_row = dataframe.iloc[0]
        test_data = {
            "name": str(first_row.get("Name", "")).strip(),
            "company_name": str(first_row.get("Company Name", "")).strip(),

            "invited_by": str(first_row.get("Invited by", "")).strip(),
            "fees": safe_int_convert(first_row.get("Fees", "")),  # int8
            "payment_mode": str(first_row.get("Payment Mode", "")).strip(),
            'note' : str(first_row.get("Note", "")).strip(),
            "created_at": datetime.now().isoformat()
        }

        st.write("Testing with data:", test_data)
        st.write("Payment value type:", type(test_data["fees"]))

        response = supabase.table("bni_visitors_details").insert(test_data).execute()

        st.write("Response:", response.data)
        st.write("Error:", getattr(response, 'error', 'None'))

        return response.data is not None

    except Exception as e:
        st.error(f"Test failed: {str(e)}")
        st.write(traceback.format_exc())
        return False
# Database helper functions with enhanced error handling
#def hash_password(password: str) -> str:
    #"""Hash password using SHA-256 with input validation"""
    #try:
        #validate_input("Password", password, required=True, min_length=1)
        #return hashlib.sha256(password.encode()).hexdigest()
    #except Exception as e:
        #logger.error(f"Password hashing failed: {e}")
        #raise ValidationError("Failed to process password")


#@handle_error
#def authenticate_user(username: str, password: str) -> bool:
    #"""Authenticate admin user with comprehensive error handling"""
    #try:
        #validate_input("Username", username, required=True)
        #validate_input("Password", password, required=True)

        #password_hash = hash_password(password)

        # Enhanced query with better error handling
        #response = supabase.table('admins').select('*').eq('username', username.strip()).execute()

        #if not response.data:
            #logger.warning(f"No admin found with username: {username}")
            #raise AuthenticationError("Invalid credentials")

        #stored_user = response.data[0]
        #stored_hash = stored_user.get('password_hash', '')

        #if not stored_hash:
            #logger.error("No password hash found in database")
            #raise DatabaseError("User account corrupted - no password hash")

        #if stored_hash == password_hash:
            #logger.info(f"Successful login for user: {username}")
            #return True
        #else:
            #logger.warning(f"Password mismatch for user: {username}")
            #raise AuthenticationError("Invalid credentials")

    #except (ValidationError, AuthenticationError, DatabaseError):
        #raise
    #except Exception as e:
        #logger.error(f"Unexpected authentication error: {e}")
        #raise AuthenticationError("Authentication service temporarily unavailable")


#@handle_error
#def ensure_admin_exists() -> bool:
    #"""Create default admin if not exists with proper error handling"""
    #try:
        # Check if admin table exists and is accessible
        #response = supabase.table('admins').select('username').limit(1).execute()

        # Check for existing admin
        #admin_response = supabase.table('admins').select('*').eq('username', 'admin').execute()

        #if len(admin_response.data) == 0:
            # Create new admin with the correct password hash
            #default_password = 'admin123'
            #admin_data = {
                #'username': 'admin',
                #'password_hash': hash_password(default_password),
                #'created_at': datetime.now().isoformat()
            #}

            #result = supabase.table('admins').insert(admin_data).execute()

            #if result.data:
                #logger.info("Default admin created successfully")
                #st.success("✅ Default admin created successfully!")
                #st.info(f"Username: admin | Password: {default_password}")
                #return True
            #else:
                #raise DatabaseError("Failed to create admin account")

        #logger.info("Admin account already exists")
        #return False

    #except Exception as e:
        #logger.error(f"Admin creation failed: {e}")
        #raise DatabaseError(f"Could not create/verify admin account: {str(e)}")



@handle_error
def get_all_members() -> pd.DataFrame:
    """Fetch all members from database with error handling"""
    try:
        response = supabase.table('bni_visitors_details').select('*').order('created_at', desc=True).execute()

        if response.data is None:
            logger.warning("No data returned from members query")
            return pd.DataFrame()

        df = pd.DataFrame(response.data)
        logger.info(f"Successfully fetched {len(df)} members")
        return df

    except Exception as e:
        logger.error(f"Failed to fetch members: {e}")
        raise DatabaseError(f"Could not retrieve members: {str(e)}")

@handle_error
def add_member(name: str,  company_name: str, invited_by: str, fees: int, payment_mode: str, note: str) -> bool:
    """Add new member with comprehensive validation and error handling"""
    try:
        # Input validation
        validate_input("Name", name, required=True, min_length=2)

        if fees <= 0:
            raise ValidationError("Payment amount must be greater than 0")

        # Prepare member data
        member_data = {
            'name': name.strip(),
            'company_name': company_name.strip() if company_name else '',

            'invited_by': invited_by.strip(),
            'fees': fees,
            'payment_mode':payment_mode.strip(),
            'note': note.strip(),
            'created_at': datetime.now().isoformat()
        }


        # Check for duplicate names (optional warning)
        existing_response = supabase.table('bni_visitors_details').select('name').eq('name', name.strip()).execute()
        if existing_response.data:
            st.warning(f"⚠️ A member named '{name}' already exists. Adding anyway...")

        # Insert member
        response = supabase.table('bni_visitors_details').insert(member_data).execute()

        if response.data:
            logger.info(f"Successfully added visitor: {name}")
            return True
        else:
            raise DatabaseError("Failed to insert visitor data")

    except (ValidationError, DatabaseError):
        raise
    except Exception as e:
        logger.error(f"Unexpected error adding visitor: {e}")
        raise DatabaseError(f"Could not add visitor: {str(e)}")


@handle_error
def update_member(member_id: int, name : str, company_name: str,  invited_by: str, fees: int, payment_mode: str,  note:str) -> bool:
    """Update member with validation and error handling"""
    try:
        # Input validation
        validate_input("Name", name, required=True, min_length=2)

        if fees <= 0:
            raise ValidationError("Payment amount must be greater than 0")

        # Check if member exists
        check_response = supabase.table('bni_visitors_details').select('id').eq('id', member_id).execute()
        if not check_response.data:
            raise ValidationError(f"Member with ID {member_id} not found")

        # Prepare update data
        member_data = {
            'name': name.strip(),
            'company_name': company_name.strip() if company_name else '',

            'invited_by': invited_by.strip(),
            'fees': fees,
            'payment_mode': payment_mode.strip(),
            'note': note.strip()
        }
        # Update member
        response = supabase.table('bni_visitors_details').update(member_data).eq('id', member_id).execute()

        if response.data:
            logger.info(f"Successfully updated member ID: {member_id}")
            return True
        else:
            raise DatabaseError("Failed to update member data")

    except (ValidationError, DatabaseError):
        raise
    except Exception as e:
        logger.error(f"Unexpected error updating member: {e}")
        raise DatabaseError(f"Could not update member: {str(e)}")

@handle_error
def delete_member(member_id: int) -> bool:
    """Delete member with error handling"""
    try:
        # Check if member exists
        check_response = supabase.table('bni_visitors_details').select('id, name').eq('id', member_id).execute()
        if not check_response.data:
            raise ValidationError(f"Member with ID {member_id} not found")

        member_name = check_response.data[0]['name']

        # Delete member
        response = supabase.table('bni_visitors_details').delete().eq('id', member_id).execute()

        if response.data is not None:  # Supabase delete returns data on success
            logger.info(f"Successfully deleted member: {member_name} (ID: {member_id})")
            return True
        else:
            raise DatabaseError("Failed to delete member")

    except (ValidationError, DatabaseError):
        raise
    except Exception as e:
        logger.error(f"Unexpected error deleting member: {e}")
        raise DatabaseError(f"Could not delete member: {str(e)}")

def get_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate member statistics with error handling"""
    try:
        if df.empty:
            return {'total_members': 0, 'total_payment': 0}

        # Ensure payment column exists and is numeric
        if 'fees' not in df.columns:
            logger.warning("Fees column missing from dataframe")
            return {'total_members': len(df), 'total_payment': 0}

        # Handle non-numeric payment values
        numeric_payments = pd.to_numeric(df['fees'], errors='coerce').fillna(0)

        return {
            'total_members': len(df),
            'total_payment': numeric_payments.sum(),
        }
    except Exception as e:
        logger.error(f"Stats calculation error: {e}")
        return {'total_members': 0, 'total_payment': 0}

@handle_error
#def reset_admin_password() -> bool:
    #"""Reset admin password to default with proper hashing"""
    #try:
        #default_password = 'admin123'
        #new_hash = hash_password(default_password)

        # First, delete any existing admin to avoid conflicts
        #supabase.table('admins').delete().eq('username', 'admin').execute()

        # Create fresh admin with correct hash
        #admin_data = {
            #'username': 'admin',
            #'password_hash': new_hash,
            #'created_at': datetime.now().isoformat()
        ##}

        #create_response = supabase.table('admins').insert(admin_data).execute()

        #if create_response.data:
            #logger.info("Admin password reset successfully")
            #st.success("✅ Admin account reset successfully!")
            #st.info("Username: admin | Password: admin123")
            #return True
        #else:
            #raise DatabaseError("Failed to create admin account")

    #except Exception as e:
        #logger.error(f"Password reset failed: {e}")
        #raise DatabaseError(f"Could not reset password: {str(e)}")

def init_session_state():
    """Initialize session state variables with error recovery"""
    try:
        default_states = {
            'logged_in': False,
            'show_add_form': False,
            'edit_member': None,
            'delete_confirmation': None,
            'search_query': "",
            'error_count': 0,
            'last_error': None,
            'selected_dates_calendar': set(),
            'amount': 0.0,
            'selected_thursdays': []
        }
        for key, default_value in default_states.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    except Exception as e:
        logger.error(f"Session state initialization failed: {e}")
        # Force reset session state on critical failure
        for key in list(st.session_state.keys()):
            del st.session_state[key]


# Session state management with error recovery
def show_error_details(error_message: str):
    """Show detailed error information for debugging"""
    with st.expander("🔧 Error Details (Click to expand)"):
        st.error(error_message)
        st.code(f"""
        Error Count: {st.session_state.get('error_count', 0)}
        Last Error Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

        Troubleshooting Tips:
        1. Refresh the page and try again
        2. Check your internet connection
        3. Verify database tables exist in Supabase
        4. Clear browser cache if issues persist
        """)

@handle_error
def search_members(query: str) -> pd.DataFrame:
    """Search members with error handling"""
    try:
        validate_input("Search query", query, required=True, min_length=1)

        # Sanitize search query
        clean_query = query.strip().replace("'", "''")  # Basic SQL injection prevention

        response = supabase.table('bni_visitors_details').select('*').or_(

            f'name.ilike.%{clean_query}%,toa.ilike.%{clean_query}%'
        ).order('created_at', desc=True).execute()

        df = pd.DataFrame(response.data) if response.data else pd.DataFrame()
        logger.info(f"Search for '{query}' returned {len(df)} results")
        return df

    except (ValidationError, DatabaseError):
        raise
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise DatabaseError(f"Search failed: {str(e)}")

def process_supabase_data_simple(raw_data):
    """Process raw Supabase data and calculate total fee per individual (no Thursday logic)"""
    try:
        from datetime import datetime

        st.write(f"🔍 **Processing {len(raw_data)} records from Supabase...**")

        payment_by_person = {}

        for record in raw_data:
            name = record.get('name', 'Unknown').strip().title()
            fees = record.get('fees', 0)


            # Accumulate fee per person
            if name in payment_by_person:
                payment_by_person[name] += fees
            else:
                payment_by_person[name] = fees

        # Format as list of dicts
        processed_data = [
            {"name": name, "total_payment": total}
            for name, total in payment_by_person.items()
        ]

        st.write(f"📊 **Successfully processed {len(processed_data)} individuals**")
        return processed_data

    except Exception as e:
        st.error(f"❌ Error processing Supabase data: {str(e)}")
        return []


def display_payment_summary(selected_thursday, weekly_amount):
    """Display payment summary for a single Thursday"""
    try:
        # ✅ Check if data is loaded
        if 'individuals_data' not in st.session_state or st.session_state.individuals_data is None:
            st.info("No payment data loaded.")
            return

        individuals = st.session_state.individuals_data

        # ✅ If DataFrame, convert to list of dicts
        if isinstance(individuals, pd.DataFrame):
            if individuals.empty:
                st.info("No payment data available.")
                return
            individuals = individuals.to_dict(orient="records")
        elif not individuals:  # Check if list is empty
            st.info("No payment data available.")
            return

        #st.write(f"### 📅 Payment Summary for: **{selected_thursday.strftime('%B %d, %Y')}**")
        #st.write(f"🧾 Weekly Payment Amount: Rs. {weekly_amount:,.2f}")

        summary_data = []
        total_collected = 0
        attendees_count = 0
        payment_mode_totals = {}  # e.g., {'Cash': 3000, 'UPI': 4500}

        # Debug: Print data structure
        #st.write(f"Debug: Processing {len(individuals)} individuals")

        for individual in individuals:
            # Handle different possible name fields
            name = individual.get("name") if individual.get("name") else " "
            company_name = individual.get("company_name") if individual.get("company_name") else " "
            invited_by = individual.get("invited_by") if individual.get("invited_by") else " "
            fees = individual.get("fees", 0)
            payment_mode = individual.get("payment_mode") if individual.get("payment_mode") else " "
            note = individual.get("note") if individual.get("note") else " "


            # Handle different fee structures
            if isinstance(fees, list):
                total_paid = sum(
                    float(fee) if isinstance(fee, (int, float, str)) and str(fee).replace('.', '').isdigit() else 0 for
                    fee in fees)
            elif isinstance(fees, (int, float)):
                total_paid = float(fees)
            elif isinstance(fees, str):
                try:
                    total_paid = float(fees)
                except (ValueError, TypeError):
                    total_paid = 0
            else:
                total_paid = 0
            # Normalize payment mode (handle None or unexpected values)
            mode = str(payment_mode).strip().title() if payment_mode else "Unknown"

            # Update totals per payment mode
            if total_paid > 0:
                payment_mode_totals[mode] = payment_mode_totals.get(mode, 0) + total_paid

            # Count attendees (anyone with payment data, even if 0)
            attendees_count += 1

            # Only add to total collected if payment > 0
            if total_paid > 0:
                total_collected += total_paid

            summary_data.append({
                "Name": name,
                'Company_Name':company_name,

                'Invited_By': invited_by,
                "Fees": total_paid,
                'Payment Mode':payment_mode,
                "Note": note,
                #"Status": "✅ Paid" if total_paid >= 1400 else "❌ Not Paid"
            })

            # Display summary metrics
        st.markdown("---")
        col1, col2, col3,col4 = st.columns(4)

        with col1:
            st.metric("Total Attendees", attendees_count)

        with col2:
            normalized_totals = {k.lower(): v for k, v in payment_mode_totals.items()}
            cash_total = normalized_totals.get("cash", 0)
            st.write("💵 Cash Collected", f"Rs. {cash_total:,.2f}")

        with col3:
            normalized_totals = {k.lower(): v for k, v in payment_mode_totals.items()}
            # Safely get 'cash' regardless of original casing
            online_total = normalized_totals.get("online", 0)
            st.write("💵 Online Collected", f"Rs. {online_total:,.2f}")

        with col4:
            st.write("💵 total amount", f"Rs. {total_collected:,.2f}")

        # Display the data table
        if summary_data:
            df = pd.DataFrame(summary_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.warning("No payment data to display.")

        # Back to Dashboard button
        st.divider()
        if st.button("🏠 Back to Dashboard", key="back_to_dashboard_payment_summary", use_container_width=True):
            st.session_state.show_payment_summary = False
            st.session_state.show_add_form = False
            st.session_state.show_visitors_list = False
            st.rerun()

    except Exception as e:
        st.error(f"❌ Error displaying payment summary: {str(e)}")
        # Additional debugging info
        st.write("Debug Information:")
        st.write(f"- Session state keys: {list(st.session_state.keys()) if 'st' in globals() else 'N/A'}")
        if 'individuals_data' in st.session_state:
            st.write(f"- Data type: {type(st.session_state.individuals_data)}")
            if hasattr(st.session_state.individuals_data, 'shape'):
                st.write(f"- Data shape: {st.session_state.individuals_data.shape}")
        import traceback
        st.code(traceback.format_exc())

# Solution 2: Force sidebar visibility in show_sidebar function
def show_sidebar():
    """Show sidebar with file upload and other options"""
    try:
        # Force sidebar to be visible
        st.sidebar.markdown("")  # This ensures sidebar is created

        with st.sidebar:
            # Error monitoring
            if st.session_state.get('error_count', 0) > 0:
                st.warning(f"⚠️ {st.session_state.error_count} error(s) occurred this session")

            # Image Processing Section
            st.header("📂 Image Processing")

            # Option selector for upload or capture
            capture_option = st.radio(
                "Choose input method:",
                options=["Upload File", "Capture Image"],
                key="input_method"
            )

            uploaded_file = None
            captured_image = None

            if capture_option == "Upload File":
                # File upload option
                uploaded_files = st.file_uploader(
                    "Upload image file(s)",
                    type=["jpg", "jpeg", "png"],
                    accept_multiple_files=True,
                    help="Select one or more image files from your device"
                )

                if uploaded_files:
                    st.success(f"✅ {len(uploaded_files)} file(s) uploaded!")

                    # Store in session state
                    st.session_state.image_to_process = uploaded_files
                    st.session_state.image_source = "upload"


            else:  # Capture Image
                # Camera capture option
                st.info("📷 Use camera to capture")
                captured_image = st.camera_input(
                    "Take a picture",
                    help="Click to take a picture using your device's camera"
                )

                if captured_image is not None:
                    st.success("✅ Image captured!")
                    # Store in session state for dashboard processing
                    st.session_state.image_to_process = captured_image
                    st.session_state.image_source = "capture"

            # Process button
            if st.session_state.get('image_to_process') is not None:
                if st.button("🔄 Process Image", use_container_width=True, type="primary"):
                    with st.spinner("Processing..."):
                        try:
                            # Process the image and extract the data
                            all_data = []

                            # Process single or multiple images
                            images_to_process = st.session_state.image_to_process
                            if not isinstance(images_to_process, list):
                                images_to_process = [images_to_process]

                            for img in images_to_process:
                                try:
                                    data = upload_and_process_image(img)
                                    if isinstance(data, pd.DataFrame):
                                        all_data.append(data)
                                    elif isinstance(data, list):  # In case it returns list of dicts
                                        all_data.append(pd.DataFrame(data))
                                except Exception as e:
                                    st.warning(
                                        f"Failed to process {img.name if hasattr(img, 'name') else 'captured image'}: {e}")

                            # Combine all results
                            if all_data:
                                combined_data = pd.concat(all_data, ignore_index=True)
                                st.session_state.extracted_data = combined_data
                                st.session_state.data_processed = True
                                st.success("✅ All images processed successfully!")
                                st.rerun()  # Refresh to show data in main area
                            else:
                                st.error("No data extracted from uploaded images.")


                        except Exception as e:
                            st.error(f"❌ Processing failed: {str(e)}")
                            st.session_state.extracted_data = None
                            st.session_state.data_processed = False

            # Clear processed data button
            if st.session_state.get('data_processed'):
                if st.button("🗑️ Clear Data", use_container_width=True):
                    st.session_state.image_to_process = None
                    st.session_state.extracted_data = None
                    st.session_state.data_processed = False
                    st.rerun()

            st.divider()


            #if st.button("🚪 Logout", use_container_width=True):
                #for key in list(st.session_state.keys()):
                    #del st.session_state[key]
                #st.rerun()

    except Exception as e:
        st.sidebar.error(f"Sidebar error: {str(e)}")
        logger.error(f"Sidebar error: {e}")

# Solution 4: Quick Debug - Add this to your current main execution
def debug_sidebar():
    """Quick debug function to force sidebar visibility"""
    with st.sidebar:
        st.write("🔧 Debug: Sidebar is working!")
        st.write("If you see this, sidebar is functional")


def load_all_visitors_data():
    """Load all visitors from database regardless of date"""
    try:
        response = supabase.table("bni_visitors_details").select("*").order(
            "created_at", desc=True
        ).execute()

        if response.data:
            st.session_state.individuals_data = response.data
        else:
            st.session_state.individuals_data = []

    except Exception as e:
        st.error(f"Error fetching all visitors: {e}")
        st.session_state.individuals_data = []


def show_all_members_list():
    """Display ALL members from database (no date filter)"""

    load_all_visitors_data()
    members_data = st.session_state.individuals_data

    if members_data is None or (isinstance(members_data, list) and not members_data):
        st.info("📭 No visitors found in database")
        return

    # Convert to DataFrame if it's a list
    if isinstance(members_data, list):
        if not members_data:
            st.info("📭 No visitors found in database")
            return
        df = pd.DataFrame(members_data)
    elif isinstance(members_data, pd.DataFrame):
        if members_data.empty:
            st.info("📭 No visitors found in database")
            return
        df = members_data.copy()
    else:
        st.error(f"❌ Unsupported data format: {type(members_data)}")
        return

    st.success(f"✅ Found {len(df)} total visitors in database")

    # Store the original IDs for updating later
    original_ids = df['id'].tolist() if 'id' in df.columns else []

    # Define display columns (exclude id and created_at from editing)
    display_columns = ['name', 'company_name', 'invited_by', 'fees', 'payment_mode', 'note']

    # Create display DataFrame with only the columns we want to show
    display_df = df.copy()

    # Keep only display columns that exist
    existing_display_cols = [col for col in display_columns if col in display_df.columns]
    display_df = display_df[existing_display_cols]

    # Rename columns for display
    column_mapping = {
        'name': 'Name',
        'company_name': 'Company Name',
        'invited_by': 'Invited By',
        'fees': 'Fees',
        'payment_mode': 'Payment Mode',
        'note': 'Note'
    }

    display_df = display_df.rename(columns=column_mapping)

    # Fill empty values
    for col in display_df.columns:
        if col != 'Fees':
            display_df[col] = display_df[col].fillna('').replace('', ' ')

    # Handle Fees column
    if 'Fees' in display_df.columns:
        display_df['Fees'] = pd.to_numeric(display_df['Fees'], errors='coerce').fillna(0).astype(int)

    # Add Date column for display
    #if 'created_at' in df.columns:
    #    dates = pd.to_datetime(df['created_at'], errors='coerce').dt.strftime('%Y-%m-%d')
     #   display_df.insert(0, 'Date', dates)

    # Add row selection column at the beginning
    display_df.insert(0, 'Select', False)

    # Configure column settings
    column_config = {
        'Select': st.column_config.CheckboxColumn(
            'Select',
            help='Select rows to delete',
            width='small'
        ),

        'Name': st.column_config.TextColumn(
            'Name',
            required=True,
            width='medium'
        ),
        'Company Name': st.column_config.TextColumn(
            'Company Name',
            width='medium'
        ),
        'Invited By': st.column_config.TextColumn(
            'Invited By',
            width='medium'
        ),
        'Fees': st.column_config.NumberColumn(
            'Fees',
            format='₹%d',
            width='small'
        ),
        'Payment Mode': st.column_config.SelectboxColumn(
            'Payment Mode',
            options=['Cash', 'Online'],
            width='small'
        ),
        'Note': st.column_config.TextColumn(
            'Note',
            width='large'
        )
    }

    # Display editable dataframe
    edited_df = st.data_editor(
        display_df,
        column_config=column_config,
        num_rows="fixed",
        use_container_width=True,
        hide_index=True,
        key="all_visitors_editor"
    )

    # Show summary statistics
    #st.divider()
    #col1, col2, col3, col4 = st.columns(4)

    #with col1:
        #st.metric("👥 Total Visitors", len(edited_df))

    #with col2:
        #if 'Fees' in edited_df.columns and 'Payment Mode' in edited_df.columns:
            #cash_total = edited_df[edited_df['Payment Mode'].str.lower() == 'cash']['Fees'].sum()
            #st.metric("💵 Cash Collected", f"₹{cash_total:,.0f}")

    #with col3:
        #if 'Fees' in edited_df.columns and 'Payment Mode' in edited_df.columns:
            #online_total = edited_df[edited_df['Payment Mode'].str.lower() == 'online']['Fees'].sum()
            #st.metric("💳 Online Collected", f"₹{online_total:,.0f}")

    #with col4:
        #if 'Fees' in edited_df.columns:
            #total_fees = edited_df['Fees'].sum()
            #st.metric("💰 Total Collected", f"₹{total_fees:,.0f}")

    # Action buttons
    st.divider()
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    with col1:
        if st.button("💾 Save All Changes", type="primary", key="save_all_changes_all_visitors_btn",
                     use_container_width=True):
            # Validate data
            empty_names = edited_df[edited_df['Name'].str.strip() == ''].index
            if len(empty_names) > 0:
                st.error(f"⚠️ Please fill in names for all rows")
            else:
                with st.spinner("Updating all records..."):
                    success_count = 0
                    error_count = 0

                    # Remove Select and Date columns for saving
                    save_df = edited_df.drop(['Select', 'Date'], axis=1, errors='ignore')

                    # Convert back to original column names
                    reverse_mapping = {v: k for k, v in column_mapping.items()}
                    save_df = save_df.rename(columns=reverse_mapping)

                    # Update each record
                    for idx, (_, row) in enumerate(save_df.iterrows()):
                        if idx < len(original_ids):
                            record_id = original_ids[idx]
                            record_data = row.to_dict()

                            if update_visitor_record(record_id, record_data):
                                success_count += 1
                            else:
                                error_count += 1

                    if success_count > 0:
                        st.success(f"✅ Successfully updated {success_count} record(s)!")
                    if error_count > 0:
                        st.error(f"❌ Failed to update {error_count} record(s)")

                    if success_count > 0:
                        time.sleep(1)
                        st.rerun()

    with col2:
        # Check if any rows are selected for deletion
        selected_rows = edited_df[edited_df['Select'] == True]
        if len(selected_rows) > 0:
            if st.button(f"🗑️ Delete Selected ({len(selected_rows)})", key="delete_selected_all_btn",
                         use_container_width=True):
                # Get IDs of selected rows
                selected_indices = selected_rows.index.tolist()
                ids_to_delete = [original_ids[idx] for idx in selected_indices if idx < len(original_ids)]

                # Get names for confirmation
                names_to_delete = selected_rows['Name'].tolist()

                st.session_state.bulk_delete_confirmation = {
                    'ids': ids_to_delete,
                    'names': names_to_delete,
                    'count': len(ids_to_delete)
                }
                st.rerun()

    with col3:
        # Download as CSV
        download_df = edited_df.drop('Select', axis=1, errors='ignore')
        csv = download_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"all_visitors_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            key="download_all_visitors_csv",
            use_container_width=True
        )

    with col4:
        if st.button("🏠 Back to Dashboard", key="back_to_dashboard_from_all", use_container_width=True):
            st.session_state.show_visitors_list = False
            st.session_state.view_all_dates = False
            st.session_state.show_add_form = False
            st.session_state.show_payment_summary = False
            st.rerun()

    # Handle bulk delete confirmation (same as in show_members_list_for_date)
    if st.session_state.get('bulk_delete_confirmation'):
        delete_info = st.session_state.bulk_delete_confirmation

        st.markdown(f"""
        <div style="background-color: #fff3cd; padding: 20px; border-radius: 10px; border-left: 5px solid #ffc107; margin: 20px 0;">
            <h4 style="color: #856404; margin: 0 0 10px 0;">⚠️ Confirm Deletion</h4>
            <p style="color: #856404; margin: 0;">Are you sure you want to delete <strong>{delete_info['count']}</strong> visitor(s)?</p>
            <p style="color: #856404; margin: 10px 0 0 0;"><strong>Visitors to be deleted:</strong></p>
            <ul style="color: #856404;">
                {''.join([f'<li>{name}</li>' for name in delete_info['names'][:5]])}
            </ul>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            if st.button("❌ Yes, Delete All", type="primary", key="confirm_bulk_delete_all_final"):
                with st.spinner(f"Deleting {delete_info['count']} visitor(s)..."):
                    success_count = 0
                    error_count = 0

                    for record_id in delete_info['ids']:
                        result = delete_member(record_id)
                        if result:
                            success_count += 1
                        else:
                            error_count += 1

                    if success_count > 0:
                        st.success(f"✅ Successfully deleted {success_count} visitor(s)!")
                    if error_count > 0:
                        st.error(f"❌ Failed to delete {error_count} visitor(s)")

                    st.session_state.bulk_delete_confirmation = None
                    time.sleep(1)
                    # Reload all data after deletion
                    load_all_visitors_data()
                    st.rerun()

        with col2:
            if st.button("↩️ Cancel", key="cancel_bulk_delete_all_final"):
                st.session_state.bulk_delete_confirmation = None
                st.rerun()

def show_dashboard():
    """Main dashboard function - shows calendar and date-based operations"""
    st.title("🏢 BNI Brilliance Visitors Register")

    # Initialize session state variables
    if "show_calendar" not in st.session_state:
        st.session_state.show_calendar = True

    if "show_add_form" not in st.session_state:
        st.session_state.show_add_form = False

    if "show_payment_summary" not in st.session_state:
        st.session_state.show_payment_summary = False

    if "show_visitors_list" not in st.session_state:
        st.session_state.show_visitors_list = False

    if "edit_member" not in st.session_state:
        st.session_state.edit_member = None

    if "search_query" not in st.session_state:
        st.session_state.search_query = ""

    if "selected_dates_calendar" not in st.session_state:
        st.session_state.selected_dates_calendar = set()

    if "delete_confirmation" not in st.session_state:
        st.session_state.delete_confirmation = None

    if "extracted_data" not in st.session_state:
        st.session_state.extracted_data = None

    if "data_processed" not in st.session_state:
        st.session_state.data_processed = False

    if "data_loaded_for_date" not in st.session_state:
        st.session_state.data_loaded_for_date = None

    # Always show calendar widget at the top
    selected_date = show_calendar_widget()

    # Show action buttons immediately after date selection
    if selected_date:
        #st.divider()

        # Display data summary for the selected date
        if st.session_state.get('individuals_data'):
            data_count = len(st.session_state.individuals_data)
            if data_count > 0:
                st.info(f"📊 Found **{data_count}** visitor(s) for this date")
            else:
                st.info(f"📭 No visitors found for this date")

        # Action buttons
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            if st.button("➕ Add New Visitors", key="add_new_btn", use_container_width=True):
                st.session_state.show_add_form = True
                st.session_state.show_payment_summary = False
                st.session_state.show_visitors_list = False
                st.session_state.edit_member = None
                st.rerun()

        with col2:
            if st.button("📊 Payment Summary", key="payment_btn", use_container_width=True):
                st.session_state.show_add_form = False
                st.session_state.show_payment_summary = True
                st.session_state.show_visitors_list = False
                st.session_state.edit_member = None
                st.rerun()

        with col3:
            if st.button("📋 View All Visitors", key="view_all_btn", use_container_width=True):
                st.session_state.show_add_form = False
                st.session_state.show_payment_summary = False
                st.session_state.show_visitors_list = True
                st.session_state.edit_member = None
                st.rerun()

    # Display extracted data section (only if data exists)
    if st.session_state.get('data_processed') and st.session_state.get('extracted_data') is not None:
        st.divider()
        st.header("📝 Extracted Data - Review & Edit")

        # Show source information
        source_info = st.session_state.get('image_source', 'unknown')
        if source_info == "upload":
            st.info("📁 Data extracted from uploaded file")
        else:
            st.info("📷 Data extracted from captured image")

        extracted_data = st.session_state.extracted_data

        # Ensure extracted_data is a pandas DataFrame
        if isinstance(extracted_data, list):
            extracted_data = pd.DataFrame(extracted_data)

        def parse_date_for_display(input_date):
            """Parse date for display in data editor"""
            if isinstance(input_date, date):
                return input_date
            elif isinstance(input_date, pd.Timestamp):
                return input_date.date()
            elif isinstance(input_date, str):
                if not input_date or input_date.strip() == '' or input_date == "Not Available":
                    return selected_date if selected_date else date.today()

                formats = ["%d/%m/%Y", "%Y-%m-%d", "%d-%b-%Y", "%d-%b"]

                for fmt in formats:
                    try:
                        parsed = datetime.strptime(input_date, fmt)
                        if fmt == "%d-%b":
                            parsed = parsed.replace(year=datetime.now().year)
                        return parsed.date()
                    except ValueError:
                        continue

                try:
                    cleaned = input_date.replace('+', '').replace('-', ' ').strip()
                    parts = cleaned.split()
                    if len(parts) >= 2:
                        day = parts[0]
                        month = parts[1]
                        year = parts[2] if len(parts) > 2 else str(datetime.now().year)

                        if len(year) == 3:
                            year = year + '5'
                        elif len(year) == 2:
                            year = '20' + year

                        date_str = f"{day}-{month}-{year}"
                        return datetime.strptime(date_str, "%d-%b-%Y").date()
                except:
                    pass

                return selected_date if selected_date else date.today()
            else:
                return selected_date if selected_date else date.today()

        # Prepare data for editing
        edit_df = extracted_data.copy()

        # Parse dates in the Date column
        if 'Date' in edit_df.columns:
            edit_df['Date'] = edit_df['Date'].apply(parse_date_for_display)

        # Ensure all required columns exist
        required_columns = ['Name', 'Company Name', 'Invited by', 'Fees', 'Payment Mode']
        for col in required_columns:
            if col not in edit_df.columns:
                edit_df[col] = ''

        # Convert Fees column to numeric
        if 'Fees' in edit_df.columns:
            edit_df['Fees'] = pd.to_numeric(edit_df['Fees'], errors='coerce').fillna(0).astype(int)

        # Add row numbers
        edit_df.insert(0, 'Row', range(1, len(edit_df) + 1))

        # Configure column types
        column_config = {
            'Row': st.column_config.NumberColumn('Row #', disabled=True, width='small'),
            'Name': st.column_config.TextColumn('Name', required=True, width='medium'),
            'Company Name': st.column_config.TextColumn('Company Name', width='medium'),
            'Invited by': st.column_config.TextColumn('Invited by', width='medium'),
            'Fees': st.column_config.NumberColumn('Fees', format='%d', width='small'),
            'Payment Mode': st.column_config.SelectboxColumn('Payment Mode', options=['Cash', 'Online'], width='small'),
            'Note': st.column_config.TextColumn('Note', width='large')
        }

        edited_df = st.data_editor(
            edit_df,
            column_config=column_config,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            key="extracted_data_editor"
        )

        # Show summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Visitors", len(edited_df))
        with col2:
            total_fees = edited_df['Fees'].apply(
                lambda x: float(x) if str(x).replace('.', '', 1).isdigit() else 0).sum()
            st.metric("Total Fees", f"₹{total_fees:,.0f}")
        with col3:
            valid_names = edited_df['Name'].apply(lambda x: bool(str(x).strip())).sum()
            st.metric("Valid Entries", valid_names)

        # Action buttons
        st.divider()
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

        with col1:
            if st.button("💾 Save to Database", type="primary", key="save_to_db_btn"):
                save_df = edited_df.drop('Row', axis=1)
                empty_names = save_df[save_df['Name'].str.strip() == ''].index
                if len(empty_names) > 0:
                    st.error(f"⚠️ Please fill in names for all rows")
                else:
                    with st.spinner(f"Saving to database for {selected_date.strftime('%B %d, %Y')}..."):
                        # Pass the selected_date to save function
                        save_data_to_supabase(save_df, selected_date=selected_date)
                        st.success(f"✅ Data saved successfully for {selected_date.strftime('%B %d, %Y')}!")
                        # Clear extracted data and reload data for the selected date
                        st.session_state.extracted_data = None
                        st.session_state.data_processed = False
                        st.session_state.image_to_process = None
                        time.sleep(1)
                        # Reload data for the current selected date
                        load_payment_data_for_date(selected_date)
                        st.rerun()

        with col2:
            if st.button("🔄 Reset Changes", key="reset_changes_btn"):
                st.session_state.extracted_data = None
                st.session_state.data_processed = False
                st.rerun()

        with col3:
            csv_data = edited_df.to_csv(index=False).encode('utf-8')
            download_key = f"download_csv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=f"visitors_{selected_date.strftime('%Y%m%d') if selected_date else 'exported'}.csv",
                mime="text/csv",
                key=download_key
            )

        with col4:
            if st.button("🏠 Back to Dashboard", key="back_to_dashboard_extracted"):
                st.session_state.extracted_data = None
                st.session_state.data_processed = False
                st.session_state.image_to_process = None
                st.rerun()

    # Handle delete confirmation
    if st.session_state.delete_confirmation:
        member_to_delete = st.session_state.delete_confirmation

        st.markdown(f"""
                <div class="delete-confirmation">
                    <h4>⚠️ Confirm Deletion</h4>
                    <p>Are you sure you want to delete <strong>{member_to_delete.get('name', 'Unknown Member')}</strong>?</p>
                    <p><em>This action cannot be undone.</em></p>
                </div>
                """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            if st.button("❌ Yes, Delete", type="primary", key="confirm_delete_btn"):
                with st.spinner("Deleting member..."):
                    result = delete_member(member_to_delete['id'])
                    if result:
                        st.success(f"✅ Member '{member_to_delete['name']}' deleted successfully!")
                        st.session_state.delete_confirmation = None
                        time.sleep(1)
                        st.rerun()

        with col2:
            if st.button("↩️ Cancel", key="cancel_delete_btn"):
                st.session_state.delete_confirmation = None
                st.rerun()

    # Main content area based on current state
    try:
        if selected_date:
            st.divider()

            if st.session_state.get("show_add_form"):
                st.header(f"➕ Add New Visitors for {selected_date.strftime('%B %d, %Y')}")
                show_add_member_form(selected_date=selected_date)

            elif st.session_state.get("edit_member"):
                st.header(f"✏️ Edit Visitor")
                show_edit_member_form()

            elif st.session_state.get("show_payment_summary"):
                st.header(f"📊 Payment Summary for {selected_date.strftime('%B %d, %Y')}")
                weekly_amount = 500
                display_payment_summary(selected_date, weekly_amount)


            elif st.session_state.get("show_visitors_list"):
                st.header("ALL Visitors List")
                show_all_members_list()

    except Exception as e:
        st.error("❌ An error occurred in the main application")
        show_error_details(str(e))

        col1, col2 = st.columns(2)

        with col1:
            if st.button("🔄 Refresh Page"):
                st.rerun()
        with col2:
            if st.button("🏠 Return to Dashboard"):
                st.session_state.show_add_form = False
                st.session_state.show_payment_summary = False
                st.session_state.show_visitors_list = False
                st.session_state.edit_member = None
                st.session_state.show_calendar = True
                st.rerun()

def update_visitor_record(record_id, data_dict):
    """Update a visitor record in Supabase"""
    try:
        # Prepare data for update - remove None values and empty strings
        update_data = {}
        for key, value in data_dict.items():
            if key not in ['id', 'created_at']:  # Don't update ID and created_at
                # Handle different data types
                if pd.isna(value) or value == '' or value == ' ':
                    update_data[key] = None
                elif isinstance(value, (pd.Timestamp, date)):
                    update_data[key] = value.strftime('%Y-%m-%d')
                else:
                    update_data[key] = value

        # Update record in Supabase
        response = supabase.table('bni_visitors_details').update(update_data).eq('id', record_id).execute()

        if response.data:
            return True
        else:
            return False

    except Exception as e:
        st.error(f"Error updating record {record_id}: {str(e)}")
        return False


def show_members_list_for_date(selected_date):
    """Display members list for a specific date in editable table format"""

    if 'individuals_data' not in st.session_state or st.session_state.individuals_data is None:
        st.info(f"📭 No data loaded for {selected_date.strftime('%B %d, %Y')}")
        if st.button("🔄 Load Data", key="load_data_btn"):
            st.rerun()
        return

    members_data = st.session_state.individuals_data

    # Convert to DataFrame if it's a list
    if isinstance(members_data, list):
        if not members_data:
            st.info(f"📭 No visitors found for {selected_date.strftime('%B %d, %Y')}")
            return
        df = pd.DataFrame(members_data)
    elif isinstance(members_data, pd.DataFrame):
        if members_data.empty:
            st.info(f"📭 No visitors found for {selected_date.strftime('%B %d, %Y')}")
            return
        df = members_data.copy()
    else:
        st.error(f"❌ Unsupported data format: {type(members_data)}")
        return

    st.success(f"✅ Found {len(df)} visitors for this date")

    # Store the original IDs for updating later
    original_ids = df['id'].tolist() if 'id' in df.columns else []

    # Define display columns (exclude id and created_at)
    display_columns = ['name', 'company_name', 'invited_by', 'fees', 'payment_mode', 'note']

    # Create display DataFrame with only the columns we want to show
    display_df = df.copy()

    # Keep only display columns that exist
    existing_display_cols = [col for col in display_columns if col in display_df.columns]
    display_df = display_df[existing_display_cols]

    # Rename columns for display
    column_mapping = {
        'name': 'Name',
        'company_name': 'Company Name',
        'invited_by': 'Invited By',
        'fees': 'Fees',
        'payment_mode': 'Payment Mode',
        'note': 'Note'
    }

    display_df = display_df.rename(columns=column_mapping)

    # Fill empty values
    for col in display_df.columns:
        if col != 'Fees':
            display_df[col] = display_df[col].fillna('').replace('', ' ')

    # Handle Fees column
    if 'Fees' in display_df.columns:
        display_df['Fees'] = pd.to_numeric(display_df['Fees'], errors='coerce').fillna(0).astype(int)


    # Add row selection column at the beginning
    display_df.insert(0, 'Select', False)

    # Configure column settings
    column_config = {
        'Select': st.column_config.CheckboxColumn(
            'Select',
            help='Select rows to delete',
            width='small'
        ),
        'Name': st.column_config.TextColumn(
            'Name',
            required=True,
            width='medium'
        ),
        'Company Name': st.column_config.TextColumn(
            'Company Name',
            width='medium'
        ),
        'Invited By': st.column_config.TextColumn(
            'Invited By',
            width='medium'
        ),
        'Fees': st.column_config.NumberColumn(
            'Fees',
            format='₹%d',
            width='small'
        ),
        'Payment Mode': st.column_config.SelectboxColumn(
            'Payment Mode',
            options=['Cash', 'Online'],
            width='small'
        ),

        'Note': st.column_config.TextColumn(
            'Note',
            width='large'
        )
    }
    edited_df = st.data_editor(
        display_df,
        column_config=column_config,
        num_rows="fixed",
        use_container_width=True,
        hide_index=True,
        key=f"visitors_editor_{selected_date.strftime('%Y%m%d')}"
    )

    # Show summary statistics
    st.divider()
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("👥 Total Visitors", len(edited_df))

    with col2:
        if 'Fees' in edited_df.columns and 'Payment Mode' in edited_df.columns:
            cash_total = edited_df[edited_df['Payment Mode'].str.lower() == 'cash']['Fees'].sum()
            st.metric("💵 Cash Collected", f"₹{cash_total:,.0f}")

    with col3:
        if 'Fees' in edited_df.columns and 'Payment Mode' in edited_df.columns:
            online_total = edited_df[edited_df['Payment Mode'].str.lower() == 'online']['Fees'].sum()
            st.metric("💳 Online Collected", f"₹{online_total:,.0f}")

    with col4:
        if 'Fees' in edited_df.columns:
            total_fees = edited_df['Fees'].sum()
            st.metric("💰 Total Collected", f"₹{total_fees:,.0f}")

    # Action buttons
    st.divider()
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    with col1:
        if st.button("💾 Save All Changes", type="primary", key="save_all_changes_btn", use_container_width=True):
            # Validate data
            empty_names = edited_df[edited_df['Name'].str.strip() == ''].index
            if len(empty_names) > 0:
                st.error(f"⚠️ Please fill in names for all rows")
            else:
                with st.spinner("Updating all records..."):
                    success_count = 0
                    error_count = 0

                    # Remove Select column for saving
                    save_df = edited_df.drop('Select', axis=1)

                    # Convert back to original column names
                    reverse_mapping = {v: k for k, v in column_mapping.items()}
                    save_df = save_df.rename(columns=reverse_mapping)

                    # Update each record
                    for idx, (_, row) in enumerate(save_df.iterrows()):
                        if idx < len(original_ids):
                            record_id = original_ids[idx]
                            record_data = row.to_dict()

                            if update_visitor_record(record_id, record_data):
                                success_count += 1
                            else:
                                error_count += 1

                    if success_count > 0:
                        st.success(f"✅ Successfully updated {success_count} record(s)!")
                    if error_count > 0:
                        st.error(f"❌ Failed to update {error_count} record(s)")

                    if success_count > 0:
                        time.sleep(1)
                        st.rerun()

    with col2:
        # Check if any rows are selected for deletion
        selected_rows = edited_df[edited_df['Select'] == True]
        if len(selected_rows) > 0:
            if st.button(f"🗑️ Delete Selected ({len(selected_rows)})", key="delete_selected_btn", use_container_width=True):
                # Get IDs of selected rows
                selected_indices = selected_rows.index.tolist()
                ids_to_delete = [original_ids[idx] for idx in selected_indices if idx < len(original_ids)]

                # Get names for confirmation
                names_to_delete = selected_rows['Name'].tolist()

                st.session_state.bulk_delete_confirmation = {
                    'ids': ids_to_delete,
                    'names': names_to_delete,
                    'count': len(ids_to_delete)
                }
                st.rerun()


    with col3:
        # Download as CSV
        download_df = edited_df.drop('Select', axis=1)
        csv = download_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"visitors_{selected_date.strftime('%Y%m%d')}.csv",
            mime="text/csv",
            key="download_visitors_csv",
            use_container_width=True
        )

    with col4:
        if st.button("🏠 Back to Dashboard", key="back_to_dashboard_from_save", use_container_width=True):
            st.session_state.show_visitors_list = False
            st.session_state.show_add_form = False
            st.session_state.show_payment_summary = False
            st.rerun()

    # Handle bulk delete confirmation
    if st.session_state.get('bulk_delete_confirmation'):
        delete_info = st.session_state.bulk_delete_confirmation

        st.markdown(f"""
        <div style="background-color: #fff3cd; padding: 20px; border-radius: 10px; border-left: 5px solid #ffc107; margin: 20px 0;">
            <h4 style="color: #856404; margin: 0 0 10px 0;">⚠️ Confirm Deletion</h4>
            <p style="color: #856404; margin: 0;">Are you sure you want to delete <strong>{delete_info['count']}</strong> visitor(s)?</p>
            <p style="color: #856404; margin: 10px 0 0 0;"><strong>Visitors to be deleted:</strong></p>
            <ul style="color: #856404;">
                {''.join([f'<li>{name}</li>' for name in delete_info['names'][:5]])}
            </ul>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            if st.button("❌ Yes, Delete All", type="primary", key="confirm_bulk_delete_final"):
                with st.spinner(f"Deleting {delete_info['count']} visitor(s)..."):
                    success_count = 0
                    error_count = 0

                    for record_id in delete_info['ids']:
                        result = delete_member(record_id)
                        if result:
                            success_count += 1
                        else:
                            error_count += 1

                    if success_count > 0:
                        st.success(f"✅ Successfully deleted {success_count} visitor(s)!")
                    if error_count > 0:
                        st.error(f"❌ Failed to delete {error_count} visitor(s)")

                    st.session_state.bulk_delete_confirmation = None
                    time.sleep(1)
                    st.rerun()

        with col2:
            if st.button("↩️ Cancel", key="cancel_bulk_delete_final"):
                st.session_state.bulk_delete_confirmation = None
                st.rerun()

def display_member_row(member, idx):
    """Helper function to display a single member row"""
    try:
        with st.container():
            col1, col2, col3 = st.columns([3, 2, 1])

            with col1:
                # Handle both dictionary and Series objects
                name = get_member_value(member, 'name', 'Name')
                company = get_member_value(member, 'company_name', 'Company Name')

                note = get_member_value(member, 'note', 'Note')
                st.write(f"**{name}**")
                st.write(f"Company: {company}")

                st.write(f"Note: {note}")

            with col2:
                invited_by = get_member_value(member, 'invited_by', 'Invited by')
                fees = get_member_value(member, 'fees', 'Fees')
                payment_mode = get_member_value(member, 'payment_mode', 'Payment Mode')

                st.write(f"Invited by: {invited_by}")
                st.write(f"Fees: ₹{fees}")
                st.write(f"Payment: {payment_mode}")

            with col3:
                if st.button("✏️ Edit", key=f"edit_{idx}"):
                    # Convert to dict if it's a pandas Series
                    if hasattr(member, 'to_dict'):
                        st.session_state.edit_member = member.to_dict()
                    else:
                        st.session_state.edit_member = member
                    st.rerun()

                if st.button("🗑️ Delete", key=f"delete_{idx}"):
                    # Convert to dict if it's a pandas Series
                    if hasattr(member, 'to_dict'):
                        st.session_state.delete_confirmation = member.to_dict()
                    else:
                        st.session_state.delete_confirmation = member
                    st.rerun()

            st.divider()
    except Exception as e:
        st.error(f"❌ Error displaying member {idx + 1}: {str(e)}")
        st.write(f"Member data: {member}")
        st.write(f"Member type: {type(member)}")


def get_member_value(member, key1, key2=None):
    """Helper function to safely get values from member data"""
    try:
        # If it's a dictionary
        if isinstance(member, dict):
            return member.get(key1, member.get(key2, 'N/A') if key2 else 'N/A')
        # If it's a pandas Series
        elif hasattr(member, 'get'):
            value = member.get(key1)
            if value is None and key2:
                value = member.get(key2)
            return value if value is not None else 'N/A'
        # If it has attribute access (like pandas Series)
        elif hasattr(member, key1):
            return getattr(member, key1, 'N/A')
        elif key2 and hasattr(member, key2):
            return getattr(member, key2, 'N/A')
        else:
            return 'N/A'
    except Exception as e:
        st.error(f"Error getting value for key '{key1}': {str(e)}")
        return 'N/A'

def show_dashboard_with_sidebar():
    """Combined function that shows both sidebar and dashboard after login"""
    # Show sidebar first
    show_sidebar()

    # Then show dashboard content
    show_dashboard()


def show_add_member_form(selected_date=None):
    """Show add member form with pre-filled date"""


    with st.form("add_member_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            name = st.text_input("Full Name*")
            invited_by = st.text_input("Invited by*")


        with col2:
            company_name = st.text_input("Company Name*")
            fees = st.number_input("Fees*", min_value=1400, step=100, value=1400)
            note = st.text_area("Note (Optional)")

        with col3:

            # Payment mode dropdown
            payment_mode = st.selectbox("Payment Mode", ["Cash", "Online"], index=0)

        submitted = st.form_submit_button("💾 Save Visitor", use_container_width=True)

        if submitted:
            if name and company_name and invited_by:

                # Create DataFrame with single row for your existing save function
                visitor_data = pd.DataFrame([{
                    'Name': name,
                    'company name': company_name,

                    'invited by': invited_by,
                    'fees': fees,
                    'payment mode': payment_mode,

                    'note': note
                }])

                # Use your existing save function
                save_data_to_supabase(visitor_data, selected_date=selected_date)

                st.success("✅ Visitor added successfully!")
                st.session_state.show_add_form = False

                # Reload data for the current date
                load_payment_data_for_date(selected_date)
                st.rerun()
            else:
                st.error("❌ Please fill in all required fields marked with *")

        # Cancel button
        if st.form_submit_button("❌ Cancel"):
            st.session_state.show_add_form = False
            st.rerun()


def show_edit_member_form():
    """Display edit member form with proper null handling and dropdown for payment mode"""
    member_data = st.session_state.edit_member

    # Safety check for member_data
    if not member_data:
        st.error("❌ No member data found for editing")
        if st.button("⬅️ Back to Dashboard"):
            st.session_state.edit_member = None
            st.rerun()
        return

    # Helper function to safely get values
    def safe_get(data, key, default=''):
        """Safely get value from data, ensuring it's not None"""
        value = data.get(key, default) if isinstance(data, dict) else getattr(data, key, default)
        return value if value is not None else default

    st.header(f"✏️ Edit Member: {safe_get(member_data, 'name', 'Unknown Member')}")

    col1, col2 = st.columns([3, 1])

    with col2:
        if st.button("⬅️ Back to Dashboard", key="back_to_dashboard_btn"):
            st.session_state.edit_member = None
            st.rerun()

    # Display member metadata with error handling
    try:
        member_id = safe_get(member_data, 'id', 'Unknown')
        created_at = safe_get(member_data, 'created_at', 'Unknown')
        st.info(f"**Member ID:** {member_id} | **Created:** {created_at}")
    except Exception as e:
        st.warning(f"Could not display member metadata: {str(e)}")

    with st.form("edit_member_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            name = st.text_input("Full Name*", value=safe_get(member_data, 'name', ''))
            invited_by = st.text_input("Invited by", value=safe_get(member_data, 'invited_by', ''))


        with col2:
            company_name = st.text_input("Company name", value=safe_get(member_data, 'company_name', ''))

            # Handle fees safely
            fees_value = safe_get(member_data, 'fees', 0)
            try:
                fees_int = int(float(fees_value)) if fees_value not in [None, '', 'N/A'] else 1400
            except (ValueError, TypeError):
                fees_int = 1400
                st.warning(f"Invalid fees value '{fees_value}', using 1400")

            fees = int(st.number_input("Fees*", min_value=1400, step=100, value=fees_int, format="%d"))
            note = st.text_input("Note", value=safe_get(member_data, 'note', ''))

        with col3:


            # Payment mode dropdown with proper index handling
            payment_options = ["Cash", "Online"]
            current_payment_mode = safe_get(member_data, 'payment_mode', '').strip()

            # Find the index of current payment mode, default to 0 if not found
            try:
                payment_index = payment_options.index(
                    current_payment_mode) if current_payment_mode in payment_options else 0
            except (ValueError, AttributeError):
                payment_index = 0

            payment_mode = st.selectbox("Mode of payment", payment_options, index=payment_index)

        # Submit button
        submitted = st.form_submit_button("Update Member", use_container_width=True)

        if submitted:
            # Validation
            if not name.strip():
                st.error("❌ Name is required")
                return

            # Clean and validate all inputs before sending
            clean_data = {
                'name': name.strip() if name else '',
                'company_name': company_name.strip() if company_name else '',

                'invited_by': invited_by.strip() if invited_by else '',
                'fees': fees,
                'payment_mode': payment_mode,  # This is already clean from selectbox

                'note': note.strip() if note else ''
            }

            with st.spinner("Updating member..."):
                try:
                    member_id = safe_get(member_data, 'id', None)

                    if not member_id:
                        st.error("❌ Cannot update: Member ID not found")
                        return

                    result = update_member(
                        member_id,
                        clean_data['name'],
                        clean_data['company_name'],

                        clean_data['invited_by'],
                        clean_data['fees'],
                        clean_data['payment_mode'],
                        clean_data['note']
                    )

                    if result:
                        st.success(f"✅ Member '{clean_data['name']}' updated successfully!")
                        time.sleep(1)
                        st.session_state.edit_member = None
                        st.rerun()
                    else:
                        st.error("❌ Failed to update member. Check the logs for details.")

                except Exception as e:
                    st.error(f"❌ Database error: {str(e)}")


def load_payment_data_for_date(selected_date):
    """Query Supabase for visitors on the selected date"""
    try:
        # Query for records created on the selected date
        start_datetime = datetime.combine(selected_date, datetime.min.time())
        end_datetime = datetime.combine(selected_date, datetime.max.time())

        response = supabase.table("bni_visitors_details").select("*").gte(
            "created_at", start_datetime.isoformat()
        ).lte(
            "created_at", end_datetime.isoformat()
        ).execute()

        if response.data:
            st.session_state.individuals_data = response.data
            st.session_state.data_loaded_for_date = selected_date
        else:
            st.session_state.individuals_data = []
            st.session_state.data_loaded_for_date = selected_date

    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.session_state.individuals_data = []
        st.session_state.data_loaded_for_date = None


def show_calendar_widget():
    """Calendar widget that returns selected date immediately after user selects year, month, and day"""
    st.subheader("📅 Select a date to view and manage visitor data")

    # Get today's date
    today = date.today()

    # Columns for year, month, and day
    col1, col2, col3 = st.columns(3)

    with col1:
        year_options = [today.year - 1, today.year, today.year + 1]
        selected_year = st.selectbox(
            "Select Year",
            year_options,
            index=year_options.index(today.year),  # Default to today's year
            key="year_selector"
        )

    with col2:
        month_options = list(range(1, 13))
        selected_month = st.selectbox(
            "Select Month",
            month_options,
            index=month_options.index(today.month),  # Default to today's month
            format_func=lambda m: calendar.month_name[m],
            key="month_selector"
        )

    with col3:
        # Get correct number of days in selected month/year
        days_in_month = calendar.monthrange(selected_year, selected_month)[1]
        day_options = list(range(1, days_in_month + 1))

        # Default to today's day if in same month/year, otherwise day 1
        if selected_year == today.year and selected_month == today.month:
            default_day = min(today.day, days_in_month)  # Ensure day exists in month
        else:
            default_day = 1

        selected_day = st.selectbox(
            "Select Day",
            day_options,
            index=day_options.index(default_day),  # Default to calculated day
            key="day_selector"
        )

    # Construct the selected date
    selected_date = date(selected_year, selected_month, selected_day)

    # Display selected date prominently
    st.success(f"📅 Selected Date: **{selected_date.strftime('%A, %B %d, %Y')}**")

    # Load data for the selected date
    load_payment_data_for_date(selected_date)

    # Construct the selected date
    selected_date = date(selected_year, selected_month, selected_day)

    # Display the selected date prominently
   # st.success(f"📅 Selected Date: **{selected_date.strftime('%A, %B %d, %Y')}**")

    # Load data for the selected date
    #load_payment_data_for_date(selected_date)

    return selected_date

def show_members_list():
    # Get members data with error handling
    try:
        if st.session_state.search_query:
            df = search_members(st.session_state.search_query)
            if df is not None:
                st.subheader(f"🔍 Search Results for: '{st.session_state.search_query}'")
            else:
                df = pd.DataFrame()
        else:
            df = get_all_members()
            if df is not None:
                st.subheader("👥 All Visitors")
            else:
                df = pd.DataFrame()

    except Exception as e:
        st.error("Failed to load members data")
        show_error_details(str(e))
        return

    # Show statistics
    stats = get_stats(df)

    col1, col2, col3 = st.columns(3)

    # Show members table with error handling
    if df is not None and not df.empty:
        try:
            # Create display dataframe with error handling
            display_df = df.copy()

            # Safe payment formatting
            if 'fees' in display_df.columns:
                display_df['fees'] = display_df['fees'].apply(
                    lambda x: f"₹{(x):,.2f}" if pd.notnull(x) else "₹0.00"
                )

            # Safe date formatting
            if 'created_at' in display_df.columns:
                display_df['created_at'] = pd.to_datetime(
                    display_df['created_at'], errors='coerce'
                ).dt.strftime('%Y-%m-%d')

            # Show table headers
            st.markdown("---")
            col1, col2, col3, col4, col5, col6,col7,col8 = st.columns([1, 0.75, 0.75, 0.75, 0.75, 0.75,1,1])

            with col1:
                st.markdown("**Name**")
            with col2:
                st.markdown("**Company Name**")

            with col4:
                st.markdown("**Invited by**")
            with col5:
                st.markdown("**Fees**")
            with col6:
                st.markdown("**Payment Mode**")
            with col7:
                st.markdown("**Note**")

            st.markdown("---")

            # Show table with error boundaries for each row
            for idx, row in df.iterrows():
                try:
                    with st.container():
                        col1, col2, col3, col4, col5, col6,col7,col8 = st.columns([2, 1.5, 1, 1, 1.5, 1,1,1])

                        with col1:
                            st.write(f"**{row.get('name', 'Unknown')}**")
                            st.caption(f"ID: {row.get('id', 'Unknown')}")

                        with col2:
                            st.write(row.get('company_name', 'Not specified'))



                        with col3:
                            st.write(row.get('invited_by', 'Not specified'))

                        with col4:
                            fees_val = row.get('fees', 0)
                            try:
                                st.write(f"₹{(fees_val):,.2f}")
                            except (ValueError, TypeError):
                                st.write("₹0.00")

                        with col5:
                            #mode = row.get('payment_mode', 'Unknown')
                            st.write(row.get('payment_mode', 'Unknown'))

                        with col6:
                            #mode = row.get('payment_mode', 'Unknown')
                            st.write(row.get('note', 'Unknown'))

                        with col7:
                            col_edit, col_delete = st.columns(2)

                            with col_edit:
                                if st.button("✏️", key=f"edit_{row['id']}", help="Edit member"):
                                    st.session_state.edit_member = row.to_dict()
                                    st.rerun()

                            with col_delete:
                                if st.button("🗑️", key=f"delete_{row['id']}", help="Delete member"):
                                    st.session_state.delete_confirmation = row.to_dict()
                                    st.rerun()

                        st.divider()

                except Exception as e:
                    st.error(f"Error displaying member row: {str(e)}")
                    logger.error(f"Row display error for member {row.get('id', 'unknown')}: {e}")
                    continue

        except Exception as e:
            st.error("Error displaying members table")
            show_error_details(str(e))

    else:
        if st.session_state.search_query:
            st.info(f"No members found matching '{st.session_state.search_query}'")
            if st.button("🔄 Try Different Search", use_container_width=True):
                st.session_state.search_query = ""
                st.rerun()
        else:
            st.info("No members found. Add some members to get started!")
            if st.button("➕ Add First Member"):
                st.session_state.show_add_form = True
                st.rerun()

def main():
    init_session_state()
    show_dashboard_with_sidebar()

if __name__ == "__main__":
    main()
    init_session_state()