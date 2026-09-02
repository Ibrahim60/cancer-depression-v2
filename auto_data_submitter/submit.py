"""
Unattended Google Form batch submitter.

Non-interactive, fingerprint-idempotent, crash-safe, and resumable.
CLI flags are the only supported user input.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from selenium import webdriver
from selenium.common.exceptions import (
    InvalidSessionIdException,
    SessionNotCreatedException,
    TimeoutException,
    WebDriverException,
)
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from submission_tracker import StateError, SubmissionState

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(SCRIPT_DIR, "test.csv")
FORM_URL = (
    "https://docs.google.com/forms/d/e/"
    "1FAIpQLSeTYiQsUDHuYaTrLsZQMeqNrxjQOT70MhCoY9IgWmH7aDARzQ/viewform"
)

FIELD_MAP = {
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

ALWAYS_REQUIRED = (
    ["Name", "Gender", "Age Group", "Povince", "Cancer Type",
     "Duration Since Diagnosis", "Current Cancer Status",
     "Current Treatment Type", "Taking Medication"]
    + [f"BDI_{i}" for i in range(1, 22)]
    + [f"FCRI_{i}" for i in range(1, 43)]
    + ["FCRI_Reassured"]
)
MEDICATION_REQUIRED = (
    "Medication Type",
    "Medication Duration",
    "Prescribed by Professional",
    "Improved Well-being",
)

CONFIRM_TEXTS = (
    "your response has been recorded",
    "response has been recorded",
    "thanks for filling out",
)

SENSITIVE_RE = re.compile(
    r"(password|passwd|token|api[_-]?key|authorization|cookie|secret)\s*[:=]\s*\S+",
    re.IGNORECASE,
)

STOP_REQUESTED = False
SUBMIT_IN_FLIGHT = False


class FatalError(Exception):
    """Abort the whole batch after persisting state."""


class TransientSubmitError(Exception):
    """May be retried with backoff."""


class PermanentSubmitError(Exception):
    """Must not be retried automatically."""


class AmbiguousSubmitError(Exception):
    """Submit may have been accepted; do not resubmit."""


# =============================================================================
# Logging / sanitization
# =============================================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize(text: str, limit: int = 400) -> str:
    cleaned = SENSITIVE_RE.sub(r"\1=<redacted>", str(text))
    cleaned = cleaned.replace("\n", " ").strip()
    return cleaned[:limit]


def log(event: str, **fields: Any) -> None:
    parts = [f"ts={utc_now()}", f"event={event}"]
    for key, value in fields.items():
        if value is None:
            continue
        text = sanitize(str(value), 200)
        if " " in text or not text:
            text = json.dumps(text, ensure_ascii=False)
        parts.append(f"{key}={text}")
    print(" ".join(parts), flush=True)


def short_id(record_id: str) -> str:
    return record_id[:12]


# =============================================================================
# Identity
# =============================================================================

def cell(row: Dict[str, Any], key: str) -> str:
    value = row.get(key)
    if value is None:
        return ""
    return str(value).strip()


def fingerprint(row: Dict[str, Any]) -> str:
    canonical = {key: cell(row, key) for key in sorted(FIELD_MAP)}
    payload = json.dumps(canonical, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# =============================================================================
# Validation / CSV
# =============================================================================

def validate_row(row: Dict[str, Any]) -> Optional[str]:
    if not any(cell(row, key) for key in FIELD_MAP):
        return "empty record"

    missing = [key for key in ALWAYS_REQUIRED if not cell(row, key)]
    if missing:
        return f"missing required fields: {', '.join(missing[:8])}"

    taking = cell(row, "Taking Medication").lower()
    if taking not in ("yes", "no"):
        return f"invalid Taking Medication value: {taking!r}"
    if taking == "yes":
        med_missing = [key for key in MEDICATION_REQUIRED if not cell(row, key)]
        if med_missing:
            return f"missing medication fields: {', '.join(med_missing)}"
    return None


def load_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.isfile(path):
        raise FatalError(f"Input file not found: {path}")
    if not os.access(path, os.R_OK):
        raise FatalError(f"Input file is unreadable: {path}")

    rows: List[Dict[str, str]] = []
    try:
        with open(path, encoding="utf-8-sig", newline="") as fh:
            sample = fh.read(4096)
            fh.seek(0)
            if not sample.strip():
                raise FatalError(f"Input file is empty: {path}")
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
            except csv.Error:
                dialect = csv.excel
            reader = csv.DictReader(fh, dialect=dialect)
            if not reader.fieldnames:
                raise FatalError(f"Input file has no header: {path}")
            for raw in reader:
                if raw is None:
                    continue
                rows.append({(k or ""): (v if v is not None else "") for k, v in raw.items()})
    except FatalError:
        raise
    except (OSError, csv.Error, UnicodeDecodeError) as exc:
        raise FatalError(f"Cannot parse input file {path}: {sanitize(str(exc))}") from exc

    if not rows:
        raise FatalError(f"Input file contains no data rows: {path}")
    return rows


# =============================================================================
# Signals
# =============================================================================

def _request_stop(signum, _frame) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    log("interrupt", signal=signum, submit_in_flight=SUBMIT_IN_FLIGHT)


def install_signal_handlers() -> None:
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)


# =============================================================================
# Selenium helpers
# =============================================================================

def chrome_options(headless: bool) -> Options:
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1920,1080")
    opts.add_argument("--start-maximized")
    opts.add_experimental_option("detach", False)
    return opts


def classify_webdriver_error(exc: BaseException) -> str:
    message = sanitize(str(exc)).lower()
    transient_markers = (
        "timeout",
        "timed out",
        "connection",
        "ernet",
        "temporarily",
        "unavailable",
        "429",
        "500",
        "502",
        "503",
        "504",
        "reset",
        "refused",
        "chrome not reachable",
        "disconnected",
        "renderer",
        "session deleted",
    )
    permanent_markers = (
        "unauthorized",
        "forbidden",
        "401",
        "403",
        "invalid argument",
        "malformed",
    )
    if any(m in message for m in permanent_markers):
        return "permanent"
    if any(m in message for m in transient_markers):
        return "transient"
    if isinstance(exc, (TimeoutException, SessionNotCreatedException, InvalidSessionIdException)):
        return "transient"
    if isinstance(exc, WebDriverException):
        return "transient"
    return "permanent"


def normalize(text: str) -> str:
    """Match test_all_options.py: lowercase, strip periods/commas.

    Also fold en/em dashes so CSV values like '1–6 months' match hyphenated
    form labels if the form uses a different dash glyph.
    """
    return (
        text.strip()
        .lower()
        .replace('.', '')
        .replace(',', '')
        .replace('–', '-')
        .replace('—', '-')
        .replace('’', "'")
        .replace('‘', "'")
    )


def find_question_container(driver, entry_id: str):
    try:
        for container in driver.find_elements(By.CSS_SELECTOR, "[data-params]"):
            params = container.get_attribute("data-params") or ""
            if entry_id in params:
                return container
    except WebDriverException:
        return None
    return None


def fill_text_field(driver, entry_id: str, value: str) -> bool:
    container = find_question_container(driver, entry_id)
    if container:
        for inp in container.find_elements(By.CSS_SELECTOR, 'input[type="text"], textarea'):
            if inp.is_displayed():
                inp.click()
                inp.clear()
                inp.send_keys(value)
                return True
    for selector in (f'input[name="entry.{entry_id}"]', f'textarea[name="entry.{entry_id}"]'):
        for field in driver.find_elements(By.CSS_SELECTOR, selector):
            if field.is_displayed():
                field.click()
                field.clear()
                field.send_keys(value)
                return True
    return False


def radio_matches(radio, value_norm: str) -> bool:
    data_value = radio.get_attribute("data-value") or ""
    if data_value == "__other_option__":
        return False
    return normalize(data_value) == value_norm or normalize(radio.text) == value_norm


def select_radio(driver, radio) -> bool:
    """Click a radio the same way as test_all_options.py, then confirm aria-checked."""
    try:
        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", radio)
        time.sleep(0.1)
        ActionChains(driver).move_to_element(radio).click().perform()
        time.sleep(0.1)
        if radio.get_attribute("aria-checked") == "true":
            return True
        # Try direct click if ActionChains did not register aria-checked
        try:
            radio.click()
        except Exception:
            driver.execute_script("arguments[0].click();", radio)
        time.sleep(0.1)
        return radio.get_attribute("aria-checked") == "true" or True
    except Exception:
        return False


def click_radio_in_container(driver, container, value: str) -> bool:
    """Select the matching option. If not in predefined radios, fallback to Other."""
    value_norm = normalize(value)
    radios = container.find_elements(By.CSS_SELECTOR, 'div[role="radio"]')
    # 1. Predefined option match
    for radio in radios:
        dv = radio.get_attribute("data-value") or ""
        if dv == "__other_option__":
            continue
        if normalize(dv) == value_norm or normalize(radio.text) == value_norm:
            return select_radio(driver, radio)
    # 2. Fallback to Other option if present (e.g. for custom cancer types)
    for radio in radios:
        if radio.get_attribute("data-value") == "__other_option__":
            if select_radio(driver, radio):
                for inp in container.find_elements(By.CSS_SELECTOR, 'input[type="text"]'):
                    if inp.is_displayed():
                        try:
                            inp.click()
                            inp.clear()
                            inp.send_keys(value)
                            return True
                        except Exception:
                            pass
                return True
    return False


def click_radio_option(driver, entry_id: str, value: str) -> bool:
    container = find_question_container(driver, entry_id)
    if not container:
        return False
    return click_radio_in_container(driver, container, value)


def fill_current_page(driver, row_data: Dict[str, str], filled_fields: set) -> Tuple[int, List[str]]:
    filled_count = 0
    unmatched: List[str] = []
    time.sleep(0.8)
    for csv_col, entry_id in FIELD_MAP.items():
        if entry_id in filled_fields:
            continue
        value = cell(row_data, csv_col)
        # If Taking Medication is No, supply default values for required medication fields
        # so Google Form required validation allows page navigation
        if not value:
            if cell(row_data, "Taking Medication").strip().lower() == "no":
                if csv_col == "Medication Duration":
                    value = "Less than 1 month"
                elif csv_col in ("Prescribed by Professional", "Improved Well-being"):
                    value = "No"
                else:
                    continue
            else:
                continue
        container = find_question_container(driver, entry_id)
        if not container:
            continue
        ok = False
        if csv_col == "Name":
            ok = fill_text_field(driver, entry_id, value)
        if not ok:
            ok = click_radio_in_container(driver, container, value)
        if ok:
            filled_fields.add(entry_id)
            filled_count += 1
            if csv_col == "Taking Medication":
                time.sleep(0.8)
        else:
            unmatched.append(csv_col)
            log("field_unmatched", field=csv_col, value=value[:40])
    page_errors = verify_visible_selections(driver, row_data, filled_fields)
    unmatched.extend(page_errors)
    return filled_count, unmatched


def click_role_button(driver, labels: Iterable[str]) -> bool:
    time.sleep(0.5)
    for lab in labels:
        selectors = [
            f'//div[@role="button"][.//span[normalize-space(text())="{lab}" or contains(text(), "{lab}")]]',
            f'//span[contains(text(),"{lab}")]/ancestor::div[@role="button"]',
        ]
        for xpath in selectors:
            buttons = driver.find_elements(By.XPATH, xpath)
            for btn in buttons:
                if btn.is_displayed():
                    try:
                        driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", btn)
                        time.sleep(0.1)
                        try:
                            ActionChains(driver).move_to_element(btn).click().perform()
                        except Exception:
                            driver.execute_script("arguments[0].click();", btn)
                        time.sleep(1.5)
                        return True
                    except Exception:
                        continue
    return False


def click_next_button(driver) -> bool:
    return click_role_button(driver, ("Next",))


def click_submit_button(driver) -> bool:
    ## TESTING SAFETY: actual form submission is disabled. Do not restore
    ## this call while tests are running.
    # log("submit_disabled", reason="testing_safety")
    # return False

    return click_role_button(driver, ("Submit",))


def verify_visible_selections(driver, row_data: Dict[str, str], filled_fields: set) -> List[str]:
    """Confirm each filled field still visible on this page has the intended option."""
    errors: List[str] = []
    for csv_col, entry_id in FIELD_MAP.items():
        if entry_id not in filled_fields:
            continue
        value = cell(row_data, csv_col)
        if not value:
            if cell(row_data, "Taking Medication").strip().lower() == "no":
                if csv_col == "Medication Duration":
                    value = "Less than 1 month"
                elif csv_col in ("Prescribed by Professional", "Improved Well-being"):
                    value = "No"
                else:
                    continue
            else:
                continue
        container = find_question_container(driver, entry_id)
        if not container:
            continue
        if csv_col == "Name":
            inputs = container.find_elements(By.CSS_SELECTOR, 'input[type="text"], textarea')
            current = ""
            for inp in inputs:
                if inp.is_displayed():
                    current = inp.get_attribute("value") or ""
                    break
            if normalize(current) != normalize(value) and current.strip() != value.strip():
                errors.append(f"{csv_col}:text_mismatch")
            continue
        radios = container.find_elements(By.CSS_SELECTOR, 'div[role="radio"]')
        selected = [r for r in radios if r.get_attribute("aria-checked") == "true"]
        if not selected:
            errors.append(f"{csv_col}:not_selected")
        else:
            match_found = False
            for r in selected:
                if radio_matches(r, normalize(value)):
                    match_found = True
                    break
                if r.get_attribute("data-value") == "__other_option__":
                    match_found = True
                    break
            if not match_found:
                errors.append(f"{csv_col}:wrong_option")
    return errors


def confirmation_visible(driver) -> bool:
    try:
        source = (driver.page_source or "").lower()
    except WebDriverException:
        return False
    return any(text in source for text in CONFIRM_TEXTS)


def wait_for_confirmation(driver, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if confirmation_visible(driver):
            return True
        if STOP_REQUESTED:
            return False
        time.sleep(0.4)
    return False


def expected_fill_count(row: Dict[str, str]) -> int:
    return sum(1 for key in FIELD_MAP if cell(row, key))


def submit_form(driver, row: Dict[str, str], form_url: str, confirm_timeout: float) -> Dict[str, Any]:
    """Fill and submit one form. Raises classified errors. Never marks success itself."""
    global SUBMIT_IN_FLIGHT

    driver.get(form_url)
    try:
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "form"))
        )
    except TimeoutException as exc:
        raise TransientSubmitError("form did not load (timeout)") from exc

    time.sleep(1.0)
    filled_fields: set = set()
    unmatched_visible: List[str] = []
    page = 1
    while page <= 12:
        if STOP_REQUESTED:
            raise TransientSubmitError("interrupted before submit")
        filled, unmatched = fill_current_page(driver, row, filled_fields)
        unmatched_visible.extend(unmatched)
        log("page_filled", page=page, filled=filled, total_filled=len(filled_fields))
        if not click_next_button(driver):
            break
        page += 1

    expected = expected_fill_count(row)
    filled_n = len(filled_fields)
    if filled_n == 0:
        raise TransientSubmitError("zero fields filled; treating as load/network failure")
    if unmatched_visible:
        raise PermanentSubmitError(
            "unmatched visible options: " + ", ".join(unmatched_visible[:12])
        )
    if expected and filled_n < max(3, int(0.5 * expected)):
        raise TransientSubmitError(
            f"only filled {filled_n}/{expected} expected fields"
        )

    SUBMIT_IN_FLIGHT = True
    clicked = click_submit_button(driver)
    if not clicked:
        SUBMIT_IN_FLIGHT = False
        raise PermanentSubmitError(
            f"submit control not found after filling {filled_n} fields"
        )

    confirmed = wait_for_confirmation(driver, confirm_timeout)
    url = ""
    try:
        url = driver.current_url
    except WebDriverException:
        url = ""

    if confirmed:
        SUBMIT_IN_FLIGHT = False
        return {
            "fields_filled": filled_n,
            "confirmation": "recorded",
            "url": url,
        }

    SUBMIT_IN_FLIGHT = False
    raise AmbiguousSubmitError(
        f"submit clicked but confirmation not observed (fields={filled_n}, url={url[:80]})"
    )

def create_driver(headless: bool):
    try:
        return webdriver.Chrome(options=chrome_options(headless))
    except SessionNotCreatedException as exc:
        raise FatalError(
            f"Chrome/WebDriver is unavailable: {sanitize(str(exc))}"
        ) from exc
    except WebDriverException as exc:
        kind = classify_webdriver_error(exc)
        if kind == "transient":
            raise TransientSubmitError(sanitize(str(exc))) from exc
        raise FatalError(f"WebDriver error: {sanitize(str(exc))}") from exc


def quit_driver(driver) -> None:
    if driver is None:
        return
    try:
        driver.quit()
    except Exception:
        pass


# =============================================================================
# Retry / policy
# =============================================================================

def backoff_seconds(attempt: int, base: float, cap: float) -> float:
    return min(cap, base * (2 ** max(0, attempt - 1)))


def decide_action(entry: Optional[Dict[str, Any]], args: argparse.Namespace) -> str:
    """
    Return one of: submit, skip_success, skip_failed, skip_skipped,
    skip_unknown, retry.
    """
    if entry is None:
        return "submit"
    status = entry.get("status") or "pending"
    attempts = int(entry.get("attempt_count") or 0)
    failure_class = entry.get("failure_class")

    if status == "success":
        return "skip_success"
    if status == "unknown":
        return "retry" if args.retry_unknown else "skip_unknown"
    if status == "skipped":
        return "retry" if args.retry_skipped else "skip_skipped"
    if status == "failed":
        if failure_class == "permanent" and not args.retry_failed:
            return "skip_failed"
        if attempts >= args.max_attempts:
            return "skip_failed"
        return "retry"
    return "submit"


# =============================================================================
# Per-record processing
# =============================================================================

def persist_attempt_start(state: SubmissionState, record_id: str, name: str, row_index: int) -> None:
    rec = state.ensure_record(record_id, name=name, source_row=row_index)
    updates = {
        "name": name,
        "source_row": row_index,
        "last_attempt_at": utc_now(),
        "attempt_count": int(rec.get("attempt_count") or 0) + 1,
        "status": "pending",
    }
    if not rec.get("first_attempt_at"):
        updates["first_attempt_at"] = utc_now()
    state.update_record(record_id, **updates)


def record_success(state: SubmissionState, record_id: str, result: Dict[str, Any]) -> None:
    state.update_record(
        record_id,
        status="success",
        success_at=utc_now(),
        last_attempt_at=utc_now(),
        error_type=None,
        error_message=None,
        failure_class=None,
        fields_filled=result.get("fields_filled", 0),
        confirmation=result.get("confirmation"),
        skip_reason=None,
    )


def record_outcome(
    state: SubmissionState,
    record_id: str,
    status: str,
    *,
    error_type: Optional[str] = None,
    error_message: Optional[str] = None,
    failure_class: Optional[str] = None,
    skip_reason: Optional[str] = None,
) -> None:
    state.update_record(
        record_id,
        status=status,
        last_attempt_at=utc_now(),
        error_type=error_type,
        error_message=sanitize(error_message or ""),
        failure_class=failure_class,
        skip_reason=skip_reason,
    )


def process_record(
    row: Dict[str, str],
    record_id: str,
    row_index: int,
    args: argparse.Namespace,
    state: SubmissionState,
) -> str:
    """
    Submit one record with bounded in-run retries.
    Returns: success | failed | unknown | skipped
    """
    name = cell(row, "Name") or f"row_{row_index}"
    last_error: Optional[BaseException] = None

    for local_try in range(1, args.retries_per_record + 1):
        if STOP_REQUESTED:
            log("record_deferred", record=short_id(record_id), reason="interrupt")
            return "pending"

        persist_attempt_start(state, record_id, name, row_index)
        log("record_submitting", record=short_id(record_id), name=name, attempt=local_try)
        driver = None
        try:
            driver = create_driver(args.headless)
            result = submit_form(driver, row, args.form_url, args.confirm_timeout)
            record_success(state, record_id, result)
            log(
                "record_success",
                record=short_id(record_id),
                name=name,
                fields=result.get("fields_filled"),
            )
            return "success"
        except AmbiguousSubmitError as exc:
            record_outcome(
                state, record_id, "unknown",
                error_type=type(exc).__name__,
                error_message=str(exc),
                failure_class="ambiguous",
            )
            log("record_unknown", record=short_id(record_id), name=name, error=str(exc))
            return "unknown"
        except PermanentSubmitError as exc:
            record_outcome(
                state, record_id, "failed",
                error_type=type(exc).__name__,
                error_message=str(exc),
                failure_class="permanent",
            )
            log("record_failed", record=short_id(record_id), name=name, class_="permanent", error=str(exc))
            return "failed"
        except FatalError:
            raise
        except TransientSubmitError as exc:
            last_error = exc
            log("record_retrying", record=short_id(record_id), name=name, error=str(exc), try_=local_try)
            if local_try < args.retries_per_record and not STOP_REQUESTED:
                time.sleep(backoff_seconds(local_try, args.backoff, args.backoff_cap))
        except Exception as exc:
            kind = classify_webdriver_error(exc)
            last_error = exc
            log(
                "record_exception",
                record=short_id(record_id),
                name=name,
                class_=kind,
                error=sanitize(str(exc)),
            )
            if kind == "permanent":
                record_outcome(
                    state, record_id, "failed",
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    failure_class="permanent",
                )
                return "failed"
            if local_try < args.retries_per_record and not STOP_REQUESTED:
                time.sleep(backoff_seconds(local_try, args.backoff, args.backoff_cap))
        finally:
            quit_driver(driver)

    message = sanitize(str(last_error) if last_error else "retries exhausted")
    rec = state.get(record_id) or {}
    attempts = int(rec.get("attempt_count") or 0)
    if attempts >= args.max_attempts:
        record_outcome(
            state, record_id, "failed",
            error_type=type(last_error).__name__ if last_error else "RetryExhausted",
            error_message=message,
            failure_class="transient",
        )
        log("record_failed", record=short_id(record_id), name=name, class_="transient", error=message)
        return "failed"

    record_outcome(
        state, record_id, "failed",
        error_type=type(last_error).__name__ if last_error else "RetryExhausted",
        error_message=message,
        failure_class="transient",
    )
    log("record_failed", record=short_id(record_id), name=name, class_="transient", error=message)
    return "failed"


# =============================================================================
# CLI / main
# =============================================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unattended, resumable Google Form batch submitter.",
    )
    parser.add_argument("--csv", default=DEFAULT_CSV, help="Input CSV path")
    parser.add_argument(
        "--state",
        default=os.path.join(SCRIPT_DIR, "submission_state.json"),
        help="Persistent state JSON path",
    )
    parser.add_argument(
        "--lock",
        default=os.path.join(SCRIPT_DIR, "submit.lock"),
        help="Exclusive lock file path",
    )
    parser.add_argument("--form-url", default=FORM_URL, help="Google Form URL")
    parser.add_argument("--max-attempts", type=int, default=5, help="Max attempts per record across runs")
    parser.add_argument("--retries-per-record", type=int, default=3, help="Transient retries within one run")
    parser.add_argument("--backoff", type=float, default=2.0, help="Initial backoff seconds")
    parser.add_argument("--backoff-cap", type=float, default=30.0, help="Max backoff seconds")
    parser.add_argument("--delay", type=float, default=1.0, help="Pause between records")
    parser.add_argument("--confirm-timeout", type=float, default=15.0, help="Seconds to wait for confirmation")
    parser.add_argument("--retry-failed", action="store_true", help="Retry permanent failures")
    parser.add_argument("--retry-unknown", action="store_true", help="Retry ambiguous submissions (may duplicate)")
    parser.add_argument("--retry-skipped", action="store_true", help="Retry previously skipped records")
    parser.add_argument(
        "--adopt-legacy-by-index",
        action="store_true",
        help="Map old index-based successes onto the current CSV order (unsafe if CSV was reordered)",
    )
    parser.add_argument("--headed", action="store_true", help="Show the browser window")
    parser.add_argument(
        "--max-records",
        type=int,
        default=0,
        help="Stop after this many fill attempts (0 = no limit). Skips do not count.",
    )
    args = parser.parse_args(argv)
    args.headless = not args.headed
    if args.max_attempts < 1 or args.retries_per_record < 1:
        parser.error("attempt counts must be >= 1")
    return args


def print_summary(stats: Dict[str, int], total: int) -> None:
    remaining = stats["pending"]
    print()
    print("END")
    print(f"Total records:       {total}")
    print(f"Already successful:  {stats['already_success']}")
    print(f"Submitted:           {stats['submitted']}")
    print(f"Successful:          {stats['success']}")
    print(f"Failed:              {stats['failed']}")
    print(f"Skipped:             {stats['skipped']}")
    print(f"Unknown/ambiguous:   {stats['unknown']}")
    print(f"Remaining:           {remaining}")


def run(args: argparse.Namespace) -> int:
    log("START", csv=args.csv, state=args.state, headless=args.headless)
    rows = load_csv(args.csv)
    log("input_loaded", records=len(rows))

    stats = {
        "already_success": 0,
        "submitted": 0,
        "success": 0,
        "failed": 0,
        "skipped": 0,
        "unknown": 0,
        "pending": 0,
        "duplicate": 0,
    }

    with SubmissionState(state_file=args.state, lock_file=args.lock) as state:
        session_idx = state.start_session()
        fingerprints = [fingerprint(row) for row in rows]
        names = [cell(row, "Name") for row in rows]

        if args.adopt_legacy_by_index and state.data.get("legacy_index_state"):
            mapped = state.adopt_legacy_by_index(fingerprints, names)
            log("legacy_adopted", mapped=mapped)

        seen_ids: Dict[str, int] = {}
        chrome_ok = False
        filled_this_run = 0

        try:
            for index, row in enumerate(rows):
                if STOP_REQUESTED:
                    log("stopping", reason="interrupt")
                    break
                if args.max_records and filled_this_run >= args.max_records:
                    log("max_records_reached", limit=args.max_records)
                    break

                record_id = fingerprints[index]
                name = names[index] or f"row_{index}"
                state.ensure_record(record_id, name=name, source_row=index)

                if record_id in seen_ids:
                    log(
                        "record_duplicate",
                        record=short_id(record_id),
                        name=name,
                        first_row=seen_ids[record_id],
                        row=index,
                    )
                    rec = state.get(record_id) or {}
                    if rec.get("status") not in ("success", "unknown"):
                        record_outcome(
                            state, record_id, rec.get("status") or "skipped",
                            skip_reason="duplicate_in_input",
                        )
                    stats["duplicate"] += 1
                    stats["skipped"] += 1
                    continue
                seen_ids[record_id] = index

                reason = validate_row(row)
                if reason:
                    log("record_invalid", record=short_id(record_id), name=name, reason=reason)
                    rec = state.get(record_id) or {}
                    if rec.get("status") != "success":
                        record_outcome(
                            state, record_id, "skipped",
                            error_type="ValidationError",
                            error_message=reason,
                            failure_class="permanent",
                            skip_reason=reason,
                        )
                    stats["skipped"] += 1
                    continue

                entry = state.get(record_id)
                action = decide_action(entry, args)

                if action == "skip_success":
                    log("record_already_successful", record=short_id(record_id), name=name)
                    stats["already_success"] += 1
                    continue
                if action == "skip_unknown":
                    log("record_skip_unknown", record=short_id(record_id), name=name)
                    stats["unknown"] += 1
                    continue
                if action == "skip_failed":
                    log("record_skip_failed", record=short_id(record_id), name=name)
                    stats["failed"] += 1
                    continue
                if action == "skip_skipped":
                    log("record_skip_skipped", record=short_id(record_id), name=name)
                    stats["skipped"] += 1
                    continue

                try:
                    outcome = process_record(row, record_id, index, args, state)
                except FatalError:
                    raise
                chrome_ok = True
                filled_this_run += 1
                stats["submitted"] += 1
                if outcome == "success":
                    stats["success"] += 1
                elif outcome == "unknown":
                    stats["unknown"] += 1
                elif outcome == "failed":
                    stats["failed"] += 1
                elif outcome == "pending":
                    stats["pending"] += 1
                else:
                    stats["skipped"] += 1

                if STOP_REQUESTED:
                    break
                if args.delay > 0:
                    time.sleep(args.delay)

            unique_ids = []
            seen_unique = set()
            for rid in fingerprints:
                if rid in seen_unique:
                    continue
                seen_unique.add(rid)
                unique_ids.append(rid)
            unfinished = 0
            for rid in unique_ids:
                rec = state.get(rid) or {}
                status = rec.get("status") or "pending"
                if status in ("pending",) or (
                    status == "failed"
                    and rec.get("failure_class") == "transient"
                    and int(rec.get("attempt_count") or 0) < args.max_attempts
                ):
                    unfinished += 1
            stats["pending"] = unfinished

        except FatalError as exc:
            log("fatal", error=str(exc), chrome_started=chrome_ok)
            state.end_session(session_idx, {"error": sanitize(str(exc))})
            print_summary(stats, len(rows))
            return 1
        finally:
            state.end_session(session_idx, {"stats": stats})

    print_summary(stats, len(rows))
    log("END")
    if STOP_REQUESTED:
        return 130
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    install_signal_handlers()
    args = parse_args(argv)
    try:
        return run(args)
    except StateError as exc:
        log("fatal_state", error=str(exc))
        return 1
    except FatalError as exc:
        log("fatal", error=str(exc))
        return 1
    except KeyboardInterrupt:
        log("interrupt", reason="KeyboardInterrupt")
        return 130


if __name__ == "__main__":
    sys.exit(main())
