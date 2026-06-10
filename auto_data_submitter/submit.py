"""
Google Form Auto-Filler using Selenium
Fills multi-page forms by finding question containers and clicking radio buttons.
"""

import csv
import time
import os
import re
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains

# Get script directory for correct file paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(SCRIPT_DIR, "test.csv")

# Google Form base URL
FORM_URL = "https://docs.google.com/forms/d/e/1FAIpQLSeTYiQsUDHuYaTrLsZQMeqNrxjQOT70MhCoY9IgWmH7aDARzQ/viewform"

# Map CSV columns → Google Form entry IDs
FIELD_MAP = {
    # Personal Information (13 fields)
    "Name": "1837420799",
    "Gender": "496492923",
    "Age Group": "185868127",
    "Povince": "432228246",
    "Cancer Type": "1922255376",
    "Duration Since Diagnosis": "1117262979",
    "Current Cancer Status": "41377587",
    "Current Treatment Type": "589886884",
    "Taking Medication": "1157263309",
    "Medication Type": "1949124698",
    "Medication Duration": "253786981",
    "Prescribed by Professional": "945982350",
    "Improved Well-being": "1770878008",
    # BDI Questions 1-21
    "BDI_1": "2095530668",
    "BDI_2": "842994683",
    "BDI_3": "511464836",
    "BDI_4": "241954053",
    "BDI_5": "787638514",
    "BDI_6": "162559166",
    "BDI_7": "646096057",
    "BDI_8": "285396625",
    "BDI_9": "1491695199",
    "BDI_10": "1231304567",
    "BDI_11": "950513803",
    "BDI_12": "362517736",
    "BDI_13": "1191851353",
    "BDI_14": "1896929342",
    "BDI_15": "1818468961",
    "BDI_16": "914193838",
    "BDI_17": "876231520",
    "BDI_18": "27904361",
    "BDI_19": "2073658433",
    "BDI_20": "285811288",
    "BDI_21": "1829172516",
    # FCRI Questions 1-42 + Final
    "FCRI_1": "1773995706",
    "FCRI_2": "1430159885",
    "FCRI_3": "1470599540",
    "FCRI_4": "2099260474",
    "FCRI_5": "94745958",
    "FCRI_6": "576926841",
    "FCRI_7": "1445753609",
    "FCRI_8": "1377563770",
    "FCRI_9": "443204367",
    "FCRI_10": "670785642",
    "FCRI_11": "1363557411",
    "FCRI_12": "787206451",
    "FCRI_13": "1595117144",
    "FCRI_14": "1347525278",
    "FCRI_15": "1009510228",
    "FCRI_16": "1623100657",
    "FCRI_17": "2119174721",
    "FCRI_18": "440117541",
    "FCRI_19": "1039474799",
    "FCRI_20": "631122325",
    "FCRI_21": "88688485",
    "FCRI_22": "2089363953",
    "FCRI_23": "1017565378",
    "FCRI_24": "281134323",
    "FCRI_25": "304736930",
    "FCRI_26": "921352921",
    "FCRI_27": "1477524796",
    "FCRI_28": "1361935715",
    "FCRI_29": "1643296702",
    "FCRI_30": "1266210791",
    "FCRI_31": "448282085",
    "FCRI_32": "146738798",
    "FCRI_33": "1580988037",
    "FCRI_34": "1786780054",
    "FCRI_35": "1361457604",
    "FCRI_36": "1058445786",
    "FCRI_37": "210664541",
    "FCRI_38": "1062387799",
    "FCRI_39": "1231739582",
    "FCRI_40": "935540755",
    "FCRI_41": "799274049",
    "FCRI_42": "1329416249",
    "FCRI_Reassured": "948704729",
}


def find_question_container(driver, entry_id):
    """Find the question container that holds this entry ID in its data-params."""
    try:
        # Google Forms stores entry IDs in data-params attribute
        containers = driver.find_elements(By.CSS_SELECTOR, '[data-params]')
        for container in containers:
            params = container.get_attribute('data-params') or ''
            if entry_id in params:
                return container
    except:
        pass
    return None


def fill_text_field(driver, entry_id, value):
    """Fill a text input field."""
    try:
        # Find the question container first
        container = find_question_container(driver, entry_id)
        if container:
            # Find input/textarea within container
            inputs = container.find_elements(By.CSS_SELECTOR, 'input[type="text"], textarea')
            for inp in inputs:
                if inp.is_displayed():
                    inp.click()
                    inp.clear()
                    inp.send_keys(value)
                    return True
        
        # Fallback: direct search
        for selector in [f'input[name="entry.{entry_id}"]', f'textarea[name="entry.{entry_id}"]']:
            fields = driver.find_elements(By.CSS_SELECTOR, selector)
            for field in fields:
                if field.is_displayed():
                    field.click()
                    field.clear()
                    field.send_keys(value)
                    return True
        return False
    except Exception as e:
        print(f"      Text field error: {e}")
        return False


def normalize(text):
    """Normalize text for comparison: lowercase and remove periods."""
    return text.strip().lower().replace('.', '').replace(',', '')


def click_radio_in_container(driver, container, value):
    """Click a radio button within a specific container. If no match, use 'Other' option."""
    value_norm = normalize(value)
    
    # Find all radio options in this container
    radios = container.find_elements(By.CSS_SELECTOR, 'div[role="radio"]')
    other_radio = None
    
    for radio in radios:
        # Get the option text and normalize for comparison
        data_value = radio.get_attribute('data-value') or ''
        data_value_norm = normalize(data_value)
        label_text = normalize(radio.text)
        
        # Check for "Other" option
        if data_value == '__other_option__':
            other_radio = radio
            continue
        
        if data_value_norm == value_norm or label_text == value_norm:
            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", radio)
            time.sleep(0.15)
            ActionChains(driver).move_to_element(radio).click().perform()
            return True
    
    # If no match found and "Other" option exists, use it
    if other_radio:
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", other_radio)
        time.sleep(0.15)
        ActionChains(driver).move_to_element(other_radio).click().perform()
        time.sleep(0.2)
        
        # Find and fill the "Other" text input
        other_input = container.find_elements(By.CSS_SELECTOR, 'input[type="text"]')
        for inp in other_input:
            if inp.is_displayed():
                inp.click()
                inp.clear()
                inp.send_keys(value)
                return True
    
    return False


def click_radio_option(driver, entry_id, value):
    """Click a radio button option within the correct question."""
    try:
        container = find_question_container(driver, entry_id)
        if container:
            return click_radio_in_container(driver, container, value)
        return False
    except Exception as e:
        print(f"      Radio error: {e}")
        return False


def fill_current_page(driver, row_data, filled_fields):
    """Fill all visible fields on the current page."""
    filled_count = 0
    time.sleep(1)  # Let page fully render
    
    for csv_col, entry_id in FIELD_MAP.items():
        if entry_id in filled_fields:
            continue
            
        value = row_data.get(csv_col, "").strip()
        if not value:
            continue
        
        # Check if this entry ID is on the current page
        container = find_question_container(driver, entry_id)
        if not container:
            continue  # Not on this page
        
        # Try text field first (for Name field)
        if csv_col == "Name":
            if fill_text_field(driver, entry_id, value):
                filled_fields.add(entry_id)
                filled_count += 1
                print(f"      ✓ {csv_col.strip()}")
                continue
        
        # Try radio button
        if click_radio_option(driver, entry_id, value):
            filled_fields.add(entry_id)
            filled_count += 1
            print(f"      ✓ {csv_col.strip()}")
        else:
            print(f"      ✗ {csv_col.strip()} - could not find '{value[:30]}...'")
    
    return filled_count


def click_next_button(driver):
    """Click the Next button if present."""
    try:
        time.sleep(0.5)
        # Try multiple selectors for Next button
        selectors = [
            '//span[contains(text(),"Next")]/ancestor::div[@role="button"]',
            '//span[text()="Next"]/parent::*',
            '//div[@role="button"]//span[contains(text(),"Next")]/..',
        ]
        for xpath in selectors:
            buttons = driver.find_elements(By.XPATH, xpath)
            for btn in buttons:
                if btn.is_displayed():
                    driver.execute_script("arguments[0].click();", btn)
                    time.sleep(2)
                    return True
        return False
    except:
        return False


def click_submit_button(driver):
    """Click the Submit button to submit the form."""
    try:
        time.sleep(0.5)
        # Try multiple selectors for Submit button
        selectors = [
            '//span[contains(text(),"Submit")]/ancestor::div[@role="button"]',
            '//span[text()="Submit"]/parent::*',
            '//div[@role="button"]//span[contains(text(),"Submit")]/..',
        ]
        for xpath in selectors:
            buttons = driver.find_elements(By.XPATH, xpath)
            for btn in buttons:
                if btn.is_displayed():
                    driver.execute_script("arguments[0].click();", btn)
                    time.sleep(2)
                    return True
        return False
    except:
        return False


def main():
    with open(CSV_FILE, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("No data found in CSV file.")
        return

    print(f"Found {len(rows)} row(s) in CSV.")

    chrome_options = Options()
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_experimental_option("detach", True)

    for i, row in enumerate(rows, start=1):
        name = row.get("Name", f"Row {i}").strip()
        print(f"\n{'='*55}")
        print(f"  Row {i}/{len(rows)}: {name}")
        print(f"{'='*55}")

        driver = webdriver.Chrome(options=chrome_options)
        filled_fields = set()

        try:
            driver.get(FORM_URL)
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "form"))
            )
            time.sleep(2)

            page = 1
            max_pages = 10  # Safety limit
            while page <= max_pages:
                print(f"  Page {page}:")
                filled = fill_current_page(driver, row, filled_fields)
                print(f"    → Filled {filled} fields")
                
                if not click_next_button(driver):
                    print("    (No more pages)")
                    break
                page += 1

            # Submit the form
            if click_submit_button(driver):
                print(f"\n  ✓ Submitted! Filled {len(filled_fields)} total fields.")
            else:
                print(f"\n  ⚠ Filled {len(filled_fields)} fields but could not auto-submit. Please submit manually.")

        except Exception as e:
            print(f"  ✗ Error: {e}")

        if i < len(rows):
            input(f"\n  Press Enter to process Row {i+1} ({rows[i].get('Name','').strip()})…")

    print(f"\n{'='*55}")
    print("  All forms processed.")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()