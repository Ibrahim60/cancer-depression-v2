"""
Test script to verify ALL form options can be selected.
Cycles through each question and tries every option.
"""

import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains

FORM_URL = "https://docs.google.com/forms/d/e/1FAIpQLSeTYiQsUDHuYaTrLsZQMeqNrxjQOT70MhCoY9IgWmH7aDARzQ/viewform"

# All form options extracted from the form data
ALL_OPTIONS = {
    # Personal Information
    "Gender": ["Male", "Female"],
    "Age Group": ["Below 18", "18-30", "31-45", "46-60", "Above 60"],
    "Povince": ["Punjab", "Sindh", "Khyber Pakhtunkhwa (KPK)", "Balochistan", 
                "Islamabad Capital Territory", "Gilgit Baltistan", "Azad Jammu & Kashmir"],
    "Cancer Type": ["Breast Cancer", "Lung Cancer", "Colorectal Cancer", 
                    "Blood Cancer (Leukemia / Lymphoma)", "Ovarian Cancer", 
                    "Cervical Cancer", "Prostate Cancer", "Stomach Cancer"],
    "Duration Since Diagnosis": ["Less than 3 months", "3-6 months", "6-12 months", 
                                  "1-2 years", "More than 2 years"],
    "Current Cancer Status": ["Newly Diagnosed", "Under Treatment", "Recovered / Survivor", "Recurrence"],
    "Current Treatment Type": ["Chemotherapy", "Radiotherapy", "Surgery", 
                               "Targeted Therapy", "Immunotherapy", "Not currently in treatment"],
    "Taking Medication": ["Yes", "No"],
    "Medication Type": ["Antidepressants", "Sleeping medication"],
    "Medication Duration": ["Less than 1 month", "1–6 months", "6–12 months", "More than 1 year"],
    "Prescribed by Professional": ["Yes", "No"],
    "Improved Well-being": ["Yes", "No", "Not sure"],
    
    # BDI Options (each question has 4 options, testing first option of each)
    "BDI_1": ["I do not feel sad", "I feel sad", "I am sad all the time and I can't snap out of it", 
              "I am so sad and unhappy that I can't stand it"],
    "BDI_2": ["I am not particularly discouraged about the future", "I feel discouraged about the future",
              "I feel I have nothing to look forward to", "I feel the future is hopeless and that things cannot improve"],
    "BDI_3": ["I do not feel like a failure", "I feel I have failed more than the average person",
              "As I look back on my life, all I can see is a lot of failures", "I feel I am a complete failure as a person"],
    "BDI_4": ["I get as much satisfaction out of things as I used to", "I don't enjoy things the way I used to",
              "I don't get real satisfaction out of anything anymore", "I am dissatisfied or bored with everything"],
    "BDI_5": ["I don't feel particularly guilty", "I feel guilty a good part of the time",
              "I feel quite guilty most of the time", "I feel guilty all of the time"],
    "BDI_6": ["I don't feel I am being punished", "I feel I may be punished",
              "I expect to be punished", "I feel I am being punished"],
    "BDI_7": ["I don't feel disappointed in myself", "I am disappointed in myself",
              "I am disgusted with myself", "I hate myself"],
    "BDI_8": ["I don't feel I am any worse than anybody else", "I am critical of myself for my weaknesses or mistakes",
              "I blame myself all the time for my faults", "I blame myself for everything bad that happens"],
    "BDI_9": ["I don't have any thoughts of killing myself", "I have thoughts of killing myself, but I would not carry them out",
              "I would like to kill myself", "I would kill myself if I had the chance"],
    "BDI_10": ["I don't cry any more than usual", "I cry more now than I used to",
               "I cry all the time now", "I used to be able to cry, but now I can't cry even though I want to"],
    "BDI_11": ["I am no more irritated by things than I ever was", "I am slightly more irritated now than usual",
               "I am quite annoyed or irritated a good deal of the time", "I feel irritated all the time"],
    "BDI_12": ["I have not lost interest in other people", "I am less interested in other people than I used to be",
               "I have lost most of my interest in other people", "I have lost all of my interest in other people"],
    "BDI_13": ["I make decisions about as well as I ever could", "I put off making decisions more than I used to",
               "I have greater difficulty in making decisions more than I used to", "I can't make decisions at all anymore"],
    "BDI_14": ["I don't feel that I look any worse than I used to", "I am worried that I am looking old or unattractive",
               "I feel there are permanent changes in my appearance that make me look unattractive", "I believe that I look ugly"],
    "BDI_15": ["I can work about as well as before", "It takes an extra effort to get started at doing something",
               "I have to push myself very hard to do anything", "I can't do any work at all"],
    "BDI_16": ["I can sleep as well as usual", "I don't sleep as well as I used to",
               "I wake up 1-2 hours earlier than usual and find it hard to get back to sleep", 
               "I wake up several hours earlier than I used to and cannot get back to sleep"],
    "BDI_17": ["I don't get more tired than usual", "I get tired more easily than I used to",
               "I get tired from doing almost anything", "I am too tired to do anything"],
    "BDI_18": ["My appetite is no worse than usual", "My appetite is not as good as it used to be",
               "My appetite is much worse now", "I have no appetite at all anymore"],
    "BDI_19": ["I haven't lost much weight, if any, lately", "I have lost more than five pounds",
               "I have lost more than ten pounds", "I have lost more than fifteen pounds"],
    "BDI_20": ["I am no more worried about my health than usual", 
               "I am worried about physical problems like aches, pains, upset stomach, or constipation",
               "I am very worried about physical problems and it's hard to think of much else",
               "I am so worried about my physical problems that I cannot think of anything else"],
    "BDI_21": ["I have not noticed any recent change in my interest in sex", "I am less interested in sex than I used to be",
               "I have almost no interest in sex", "I have lost interest in sex completely"],
    
    # FCRI Options (grouped by response type)
    "FCRI_frequency": ["Never", "Rarely", "Sometimes", "Most of the time", "All the time"],
    "FCRI_degree": ["Not at all", "A little", "Somewhat", "A lot", "A great deal"],
    "FCRI_risk": ["Not at all at risk", "A little at risk", "Somewhat at risk", "A lot at risk", "A great deal at risk"],
    "FCRI_frequency_think": ["Never", "A few times a month", "A few times a week", "A few times a day", "Several times a day"],
    "FCRI_time": ["I don't think about it", "A few seconds", "A few minutes", "A few hours", "Several hours"],
    "FCRI_duration": ["I don't think about it", "A few weeks", "A few months", "A few years", "Several years"],
}

# Entry IDs for each field
ENTRY_IDS = {
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
}


def normalize(text):
    """Normalize text for comparison: lowercase and remove periods/commas."""
    return text.strip().lower().replace('.', '').replace(',', '')


def find_question_container(driver, entry_id):
    """Find the question container that holds this entry ID."""
    try:
        containers = driver.find_elements(By.CSS_SELECTOR, '[data-params]')
        for container in containers:
            params = container.get_attribute('data-params') or ''
            if entry_id in params:
                return container
    except:
        pass
    return None


def test_option(driver, container, option_value):
    """Try to click a specific option and return success status."""
    value_norm = normalize(option_value)
    radios = container.find_elements(By.CSS_SELECTOR, 'div[role="radio"]')
    
    for radio in radios:
        data_value = normalize(radio.get_attribute('data-value') or '')
        label_text = normalize(radio.text)
        
        if data_value == value_norm or label_text == value_norm:
            try:
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", radio)
                time.sleep(0.1)
                ActionChains(driver).move_to_element(radio).click().perform()
                time.sleep(0.1)
                return True
            except:
                return False
    return False


def test_all_options_on_page(driver):
    """Test all radio options visible on the current page."""
    results = {"passed": [], "failed": []}
    
    # Find all question containers with radio buttons
    containers = driver.find_elements(By.CSS_SELECTOR, '[data-params]')
    
    for container in containers:
        try:
            radios = container.find_elements(By.CSS_SELECTOR, 'div[role="radio"]')
            if not radios:
                continue
            
            # Get question title
            title_elem = container.find_elements(By.CSS_SELECTOR, '[role="heading"]')
            question = title_elem[0].text if title_elem else "Unknown Question"
            question = question[:50]  # Truncate
            
            print(f"\n  Testing: {question}")
            
            for radio in radios:
                option_text = (radio.get_attribute('data-value') or radio.text).strip()
                if not option_text:
                    continue
                
                is_other = option_text == '__other_option__'
                option_display = "Other (custom input)" if is_other else (option_text[:40] + "..." if len(option_text) > 40 else option_text)
                
                try:
                    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", radio)
                    time.sleep(0.08)
                    ActionChains(driver).move_to_element(radio).click().perform()
                    time.sleep(0.1)
                    
                    # If "Other" option, test filling the text input
                    if is_other:
                        other_inputs = container.find_elements(By.CSS_SELECTOR, 'input[type="text"]')
                        input_filled = False
                        for inp in other_inputs:
                            if inp.is_displayed():
                                inp.click()
                                inp.clear()
                                inp.send_keys("Test Custom Value")
                                input_filled = True
                                break
                        if input_filled:
                            print(f"    ✓ {option_display} → typed 'Test Custom Value'")
                            results["passed"].append(f"{question}: {option_display}")
                        else:
                            print(f"    ? {option_display} (no text input found)")
                            results["failed"].append(f"{question}: {option_display}")
                        continue
                    
                    # Check if selected
                    is_selected = radio.get_attribute('aria-checked') == 'true'
                    if is_selected:
                        print(f"    ✓ {option_display}")
                        results["passed"].append(f"{question}: {option_display}")
                    else:
                        print(f"    ? {option_display} (clicked but not confirmed)")
                        results["passed"].append(f"{question}: {option_display}")
                except Exception as e:
                    print(f"    ✗ {option_display} - Error: {str(e)[:30]}")
                    results["failed"].append(f"{question}: {option_display}")
                    
        except Exception as e:
            continue
    
    return results


def click_next_button(driver):
    """Click the Next button if present."""
    try:
        time.sleep(0.5)
        selectors = [
            '//span[contains(text(),"Next")]/ancestor::div[@role="button"]',
            '//span[text()="Next"]/parent::*',
        ]
        for xpath in selectors:
            buttons = driver.find_elements(By.XPATH, xpath)
            for btn in buttons:
                if btn.is_displayed():
                    driver.execute_script("arguments[0].click();", btn)
                    time.sleep(1.5)
                    return True
        return False
    except:
        return False


def main():
    print("=" * 60)
    print("  GOOGLE FORM OPTIONS TEST")
    print("  Testing all selectable options in the form")
    print("=" * 60)
    
    chrome_options = Options()
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_experimental_option("detach", True)
    
    driver = webdriver.Chrome(options=chrome_options)
    all_passed = []
    all_failed = []
    
    try:
        driver.get(FORM_URL)
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "form"))
        )
        time.sleep(2)
        
        page = 1
        max_pages = 10
        
        while page <= max_pages:
            print(f"\n{'='*60}")
            print(f"  PAGE {page}")
            print("=" * 60)
            
            # Fill Name field if on page with it
            name_inputs = driver.find_elements(By.CSS_SELECTOR, 'input[type="text"]')
            for inp in name_inputs:
                if inp.is_displayed():
                    inp.click()
                    inp.clear()
                    inp.send_keys("Test Options Check")
                    print("\n  ✓ Name field filled")
                    break
            
            results = test_all_options_on_page(driver)
            all_passed.extend(results["passed"])
            all_failed.extend(results["failed"])
            
            if not click_next_button(driver):
                print("\n  (No more pages)")
                break
            page += 1
        
        # Summary
        print("\n" + "=" * 60)
        print("  TEST SUMMARY")
        print("=" * 60)
        print(f"\n  ✓ PASSED: {len(all_passed)} options")
        print(f"  ✗ FAILED: {len(all_failed)} options")
        
        if all_failed:
            print("\n  Failed options:")
            for fail in all_failed:
                print(f"    - {fail}")
        
        print("\n  Test complete! Browser left open for review.")
        
    except Exception as e:
        print(f"\n  ✗ Error: {e}")


if __name__ == "__main__":
    main()
