"""
Google Form Auto-Filler using Selenium
Opens Chrome, fills the form, but does NOT submit or close the window.
"""

import csv
import time
import os
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Get script directory for correct file paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(SCRIPT_DIR, "test.csv")

# Google Form URL (viewform, not formResponse)
FORM_URL = "https://docs.google.com/forms/d/e/1FAIpQLSeTYiQsUDHuYaTrLsZQMeqNrxjQOT70MhCoY9IgWmH7aDARzQ/viewform"

# Map CSV columns to Google Form entry IDs
FIELD_MAP = {
    "Gender": "entry.496492923",
    "Age Group": "entry.185868127",
    "Povince": "entry.432228246",
    "  Cancer Type  ": "entry.1922255376",
    "  Duration Since Diagnosis  ": "entry.1117262979",
    "  Current Cancer Status  ": "entry.41377587",
    "  Current Treatment Type  ": "entry.589886884",
    "Are you currently taking any medication for depression, anxiety, stress, or other emotional/psychological problems?  ": "entry.1157263309",
    "If yes, what type of medication are you taking?  ": "entry.1949124698",
    "How long have you been taking this medication?  ": "entry.253786981",
    "Was this medication prescribed by a healthcare professional? ": "entry.945982350",
    "Do you feel the medication has improved your emotional well-being? ": "entry.1770878008",
    "1.": "entry.2095530668",
    "2.": "entry.842994683",
    "3.": "entry.511464836",
    "4.": "entry.241954053",
    "5.": "entry.787638514",
    "6.": "entry.162559166",
    "7.": "entry.646096057",
    "8.": "entry.285396625",
    "9.": "entry.1491695199",
    "10.": "entry.1231304567",
    "11.": "entry.950513803",
    "12.": "entry.362517736",
    "13.": "entry.1191851353",
    "14.": "entry.1896929342",
    "15.": "entry.1818468961",
    "16.": "entry.914193838",
    "17.": "entry.876231520",
    "18.": "entry.27904361",
    "19.": "entry.2073658433",
    "20.": "entry.285811288",
    "21.": "entry.1829172516",
}


def escape_xpath_string(s):
    """Escape a string for use in XPath - handles both single and double quotes."""
    if "'" not in s:
        return f"'{s}'"
    elif '"' not in s:
        return f'"{s}"'
    else:
        # Contains both quotes - use concat()
        parts = s.split("'")
        return "concat('" + "', \"'\", '".join(parts) + "')"


def fill_form(driver, row_data):
    """Fill form fields with data from CSV row."""
    for csv_col, entry_id in FIELD_MAP.items():
        if csv_col not in row_data or not row_data[csv_col]:
            continue
        
        value = row_data[csv_col].strip()
        
        try:
            # Try to find radio button with matching value
            # Google Forms uses data-value attribute for radio options
            escaped_value = escape_xpath_string(value)
            xpath = f"//div[@data-value={escaped_value}]"
            elements = driver.find_elements(By.XPATH, xpath)
            
            if elements:
                for elem in elements:
                    try:
                        # Check if this element is within the correct entry group
                        parent = elem.find_element(By.XPATH, f"./ancestor::div[contains(@data-params, '{entry_id}')]")
                        if parent:
                            driver.execute_script("arguments[0].scrollIntoView(true);", elem)
                            time.sleep(0.1)
                            elem.click()
                            time.sleep(0.2)
                            print(f"  Filled: {csv_col[:30]}...")
                            break
                    except:
                        continue
            else:
                # Try text input field
                input_elem = driver.find_elements(By.CSS_SELECTOR, f"input[name='{entry_id}']")
                if input_elem:
                    input_elem[0].clear()
                    input_elem[0].send_keys(value)
                    print(f"  Filled: {csv_col[:30]}...")
                    time.sleep(0.1)
                    
        except Exception as e:
            print(f"  Could not fill {csv_col[:30]}: {str(e)[:50]}")


def main():
    # Read CSV data
    with open(CSV_FILE, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if not rows:
        print("No data found in CSV file.")
        return
    
    print(f"Found {len(rows)} row(s) in CSV file.")
    
    # Setup Chrome (NOT headless - visible browser)
    chrome_options = Options()
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_experimental_option("detach", True)  # Keep browser open after script ends
    
    for i, row in enumerate(rows, start=1):
        print(f"\nOpening browser for Row {i}...")
        
        # Create new browser instance for each row
        driver = webdriver.Chrome(options=chrome_options)
        
        try:
            # Load the form
            driver.get(FORM_URL)
            
            # Wait for form to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "form"))
            )
            
            print(f"  Form loaded. Filling data...")
            time.sleep(1)
            
            # Fill the form
            fill_form(driver, row)
            
            print(f"  Row {i}: Form filled - DO NOT CLOSE BROWSER")
            print(f"  Review the form and submit manually if correct.")
            
        except Exception as e:
            print(f"  Error: {e}")
        
        # DO NOT close the browser - user will review and submit manually
        # DO NOT submit the form
        
        if i < len(rows):
            input(f"\nPress Enter to open next row ({i+1}/{len(rows)})...")
    
    print("\n" + "=" * 50)
    print("All forms opened. Review and submit each manually.")
    print("=" * 50)


if __name__ == "__main__":
    main()