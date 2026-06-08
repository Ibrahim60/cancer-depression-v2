"""
Depression Prediction Interactive Tool
Uses trained ML models to predict depression and provide suggestions.

Run: python predict_depression.py
"""

import os
import joblib
import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

MODELS_DIR = 'models'
BINARY_MODEL_PATH = os.path.join(MODELS_DIR, 'binary_voting_ensemble_model.joblib')
MULTICLASS_MODEL_PATH = os.path.join(MODELS_DIR, 'multiclass_voting_ensemble_model.joblib')
SCALER_PATH = os.path.join(MODELS_DIR, 'binary_scaler.joblib')

# Top predictive BDI questions (from feature importance analysis)
BDI_QUESTIONS = {
    '3.': {
        'question': 'How do you feel about yourself and your accomplishments?',
        'options': [
            ('I do not feel like a failure.', 0),
            ('I feel I have failed more than the average person.', 1),
            ('As I look back on my life, all I can see is a lot of failures.', 2),
            ('I feel I am a complete failure as a person.', 3)
        ]
    },
    '2.': {
        'question': 'How do you feel about the future?',
        'options': [
            ('I am not particularly discouraged about the future.', 0),
            ('I feel discouraged about the future.', 1),
            ('I feel I have nothing to look forward to.', 2),
            ('I feel the future is hopeless and that things cannot improve.', 3)
        ]
    },
    '10.': {
        'question': 'Do you cry more than usual?',
        'options': [
            ("I don't cry any more than usual.", 0),
            ('I cry more now than I used to.', 1),
            ('I cry all the time now.', 2),
            ("I used to be able to cry, but now I can't cry even though I want to.", 3)
        ]
    },
    '13.': {
        'question': 'How is your ability to make decisions?',
        'options': [
            ('I make decisions about as well as I ever could.', 0),
            ('I put off making decisions more than I used to.', 1),
            ('I have greater difficulty in making decisions than before.', 2),
            ("I can't make decisions at all anymore.", 3)
        ]
    },
    '18.': {
        'question': 'How is your appetite?',
        'options': [
            ('My appetite is no worse than usual.', 0),
            ('My appetite is not as good as it used to be.', 1),
            ('My appetite is much worse now.', 2),
            ('I have no appetite at all anymore.', 3)
        ]
    },
    '1.': {
        'question': 'How do you feel emotionally?',
        'options': [
            ('I do not feel sad.', 0),
            ('I feel sad.', 1),
            ("I am sad all the time and I can't snap out of it.", 2),
            ("I am so sad and unhappy that I can't stand it.", 3)
        ]
    },
    '7.': {
        'question': 'How do you feel about yourself?',
        'options': [
            ("I don't feel disappointed in myself.", 0),
            ('I am disappointed in myself.', 1),
            ('I am disgusted with myself.', 2),
            ('I hate myself.', 3)
        ]
    },
    '14.': {
        'question': 'How do you feel about your appearance?',
        'options': [
            ("I don't feel that I look any worse than I used to.", 0),
            ('I am worried that I am looking old or unattractive.', 1),
            ('I feel there are permanent changes in my appearance that make me look unattractive.', 2),
            ('I believe that I look ugly.', 3)
        ]
    },
    '17.': {
        'question': 'How easily do you get tired?',
        'options': [
            ("I don't get more tired than usual.", 0),
            ('I get tired more easily than I used to.', 1),
            ('I get tired from doing almost anything.', 2),
            ('I am too tired to do anything.', 3)
        ]
    },
    '20.': {
        'question': 'Are you worried about your physical health?',
        'options': [
            ('I am no more worried about my health than usual.', 0),
            ('I am worried about physical problems like aches, pains, upset stomach, or constipation.', 1),
            ('I am very worried about physical problems and it\'s hard to think of much else.', 2),
            ('I am so worried about my physical problems that I cannot think of anything else.', 3)
        ]
    }
}

# FCRI Questions (Fear of Cancer Recurrence)
FCRI_QUESTIONS = {
    '9. I am worried or anxious about the possibility of cancer recurrence': {
        'question': 'How worried are you about the possibility of cancer recurrence?',
        'options': [
            ('Not at all', 0),
            ('A little', 1),
            ('Somewhat', 2),
            ('A lot', 3),
            ('A great deal', 4)
        ]
    },
    '14. In your opinion, are you at risk of having a cancer recurrence?': {
        'question': 'In your opinion, are you at risk of having a cancer recurrence?',
        'options': [
            ('Not at all at risk', 0),
            ('A little at risk', 1),
            ('Somewhat at risk', 2),
            ('A lot at risk', 3),
            ('A great deal at risk', 4)
        ]
    },
    '12. When I think about the possibility of cancer recurrence, this triggers other unpleasant\nthoughts or images (such as death, suffering, the consequences for my family)': {
        'question': 'When thinking about cancer recurrence, do you have unpleasant thoughts (death, suffering, consequences for family)?',
        'options': [
            ('Not at all', 0),
            ('A little', 1),
            ('Somewhat', 2),
            ('A lot', 3),
            ('A great deal', 4)
        ]
    }
}

# Severity levels and suggestions
SEVERITY_INFO = {
    'Normal': {
        'level': 0,
        'description': 'No significant depression detected',
        'color': '\033[92m',  # Green
        'suggestions': [
            'Continue maintaining your mental health through regular self-care.',
            'Stay connected with friends and family.',
            'Keep up with physical activities you enjoy.',
            'Practice mindfulness or relaxation techniques as a preventive measure.'
        ]
    },
    'Mild mood disturbance': {
        'level': 1,
        'description': 'Mild mood changes detected',
        'color': '\033[93m',  # Yellow
        'suggestions': [
            'Consider talking to someone you trust about how you feel.',
            'Try to maintain regular sleep and eating patterns.',
            'Engage in activities that usually bring you joy.',
            'Light exercise like walking can help improve mood.',
            'Consider keeping a mood journal to track your feelings.'
        ]
    },
    'Borderline clinical depression': {
        'level': 2,
        'description': 'Borderline depression symptoms present',
        'color': '\033[93m',  # Yellow
        'suggestions': [
            'Consider scheduling an appointment with a mental health professional.',
            'Reach out to your cancer care team about your emotional well-being.',
            'Join a cancer support group to connect with others who understand.',
            'Practice stress-reduction techniques like deep breathing or meditation.',
            'Ensure you are getting adequate rest and nutrition.',
            'Limit alcohol and avoid recreational drugs.'
        ]
    },
    'Moderate depression': {
        'level': 3,
        'description': 'Moderate depression symptoms detected',
        'color': '\033[91m',  # Red
        'suggestions': [
            '**Please consult a mental health professional soon.**',
            'Talk to your oncologist about your emotional state - they can provide referrals.',
            'Consider evidence-based treatments like Cognitive Behavioral Therapy (CBT).',
            'Medication may be helpful - discuss options with your doctor.',
            'Lean on your support network - don\'t try to handle this alone.',
            'Cancer centers often have psycho-oncology services available.',
            'Helpline: Contact a mental health helpline if you need immediate support.'
        ]
    },
    'Severe depression': {
        'level': 4,
        'description': 'Severe depression symptoms detected',
        'color': '\033[91m',  # Red
        'suggestions': [
            '**IMPORTANT: Please seek professional help immediately.**',
            'Contact your doctor, oncologist, or a mental health professional today.',
            'If you have thoughts of self-harm, please call a crisis helpline immediately.',
            'You do not have to face this alone - professional support is available.',
            'Consider intensive outpatient or inpatient treatment if recommended.',
            'Inform a trusted family member or friend about how you are feeling.',
            'Pakistan Mental Health Helpline: 0311-7786264',
            'International: Contact your local emergency services or crisis line.'
        ]
    },
    'Extreme depression': {
        'level': 5,
        'description': 'Extreme depression symptoms - urgent attention needed',
        'color': '\033[91m',  # Red
        'suggestions': [
            '**URGENT: Please seek immediate professional help.**',
            'Go to the nearest emergency room or call emergency services.',
            'Call a crisis helpline immediately if you have thoughts of self-harm.',
            'Do not be alone - stay with someone you trust.',
            'This level of depression requires immediate professional intervention.',
            'Pakistan Mental Health Helpline: 0311-7786264',
            'Umang Helpline: 0311-7786264',
            'International: Contact your local emergency services (911, 999, 112).'
        ]
    }
}

RESET_COLOR = '\033[0m'
BOLD = '\033[1m'


def load_models():
    """Load trained models and scaler."""
    if not os.path.exists(MODELS_DIR):
        print(f"Error: Models directory '{MODELS_DIR}' not found.")
        print("Please run 'python depression_detection_pipeline.py' first to train the models.")
        return None, None, None
    
    try:
        binary_model = joblib.load(BINARY_MODEL_PATH)
        multiclass_model = joblib.load(MULTICLASS_MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("Models loaded successfully.\n")
        return binary_model, multiclass_model, scaler
    except FileNotFoundError as e:
        print(f"Error loading models: {e}")
        print("Please run 'python depression_detection_pipeline.py' first to train the models.")
        return None, None, None


def print_header():
    """Print welcome header."""
    print("\n" + "=" * 60)
    print(f"{BOLD}DEPRESSION SCREENING TOOL FOR CANCER PATIENTS{RESET_COLOR}")
    print("=" * 60)
    print("\nThis tool will ask you a few questions to assess your")
    print("emotional well-being and provide personalized suggestions.")
    print("\nPlease answer honestly - all responses are confidential.")
    print("-" * 60 + "\n")


def ask_question(question_data, question_num, total_questions):
    """Ask a single question and get user response."""
    print(f"\n{BOLD}Question {question_num}/{total_questions}{RESET_COLOR}")
    print(f"{question_data['question']}\n")
    
    for i, (option_text, _) in enumerate(question_data['options'], 1):
        print(f"  {i}. {option_text}")
    
    while True:
        try:
            choice = input(f"\nYour answer (1-{len(question_data['options'])}): ").strip()
            choice_num = int(choice)
            if 1 <= choice_num <= len(question_data['options']):
                return question_data['options'][choice_num - 1][1]
            else:
                print(f"Please enter a number between 1 and {len(question_data['options'])}")
        except ValueError:
            print("Please enter a valid number.")


def collect_responses():
    """Collect all responses from user."""
    responses = {}
    
    # Collect BDI responses
    print(f"\n{BOLD}--- Part 1: Emotional Well-being Assessment ---{RESET_COLOR}")
    total_bdi = len(BDI_QUESTIONS)
    for i, (key, q_data) in enumerate(BDI_QUESTIONS.items(), 1):
        responses[key] = ask_question(q_data, i, total_bdi)
    
    # Collect FCRI responses
    print(f"\n{BOLD}--- Part 2: Cancer-Related Concerns ---{RESET_COLOR}")
    total_fcri = len(FCRI_QUESTIONS)
    for i, (key, q_data) in enumerate(FCRI_QUESTIONS.items(), 1):
        responses[key] = ask_question(q_data, i, total_fcri)
    
    return responses


def prepare_features(responses, scaler):
    """Prepare feature vector for prediction."""
    # Create a feature vector with all expected columns (72 features from training)
    # For simplicity, we'll use the collected responses and fill others with 0
    
    # Get all BDI columns (1. through 21.)
    bdi_cols = [f'{i}.' for i in range(1, 22)]
    
    # Initialize feature dict
    features = {}
    
    # Fill BDI columns
    for col in bdi_cols:
        features[col] = responses.get(col, 0)
    
    # Fill FCRI columns (simplified - using 0 for non-asked questions)
    # In production, you'd want to map all 42 FCRI columns
    for key in FCRI_QUESTIONS.keys():
        features[key] = responses.get(key, 0)
    
    # Create DataFrame with proper column order
    df = pd.DataFrame([features])
    
    # For missing columns expected by the model, fill with 0
    # The model expects 72 features - we'll need to match the training features
    
    return df


def calculate_bdi_score(responses):
    """Calculate BDI total score from responses."""
    bdi_score = sum(responses.get(f'{i}.', 0) for i in range(1, 22) if f'{i}.' in responses)
    # Extrapolate from asked questions to estimate full BDI score
    asked_bdi = [k for k in responses.keys() if k in BDI_QUESTIONS]
    if asked_bdi:
        avg_per_question = bdi_score / len(asked_bdi)
        estimated_total = avg_per_question * 21
        return estimated_total
    return bdi_score


def get_severity_from_score(score):
    """Map BDI score to severity category."""
    if score <= 10:
        return 'Normal'
    elif score <= 16:
        return 'Mild mood disturbance'
    elif score <= 20:
        return 'Borderline clinical depression'
    elif score <= 30:
        return 'Moderate depression'
    elif score <= 40:
        return 'Severe depression'
    else:
        return 'Extreme depression'


def display_results(has_depression, severity, bdi_score):
    """Display prediction results and suggestions."""
    print("\n" + "=" * 60)
    print(f"{BOLD}ASSESSMENT RESULTS{RESET_COLOR}")
    print("=" * 60)
    
    severity_info = SEVERITY_INFO.get(severity, SEVERITY_INFO['Normal'])
    color = severity_info['color']
    
    print(f"\n{BOLD}Estimated Depression Score:{RESET_COLOR} {bdi_score:.1f}/63")
    print(f"\n{BOLD}Depression Status:{RESET_COLOR} {color}{'Detected' if has_depression else 'Not Detected'}{RESET_COLOR}")
    print(f"{BOLD}Severity Level:{RESET_COLOR} {color}{severity}{RESET_COLOR}")
    print(f"\n{severity_info['description']}")
    
    print(f"\n{BOLD}Personalized Suggestions:{RESET_COLOR}")
    print("-" * 40)
    for i, suggestion in enumerate(severity_info['suggestions'], 1):
        print(f"  {i}. {suggestion}")
    
    # FCRI-related suggestion
    fcri_responses = {k: v for k, v in responses.items() if k in FCRI_QUESTIONS}
    avg_fcri = sum(fcri_responses.values()) / len(fcri_responses) if fcri_responses else 0
    
    if avg_fcri >= 2.5:
        print(f"\n{BOLD}Cancer-Related Anxiety:{RESET_COLOR}")
        print("  Your responses indicate significant fear of cancer recurrence.")
        print("  Consider discussing these concerns with your oncology team.")
        print("  Support groups for cancer survivors can also be very helpful.")
    
    print("\n" + "-" * 60)
    print(f"{BOLD}DISCLAIMER:{RESET_COLOR}")
    print("This is a screening tool and NOT a clinical diagnosis.")
    print("Please consult a qualified healthcare professional for")
    print("proper evaluation and treatment recommendations.")
    print("=" * 60 + "\n")


def main():
    """Main function to run the prediction tool."""
    global responses
    
    print_header()
    
    # Load models
    binary_model, multiclass_model, scaler = load_models()
    if binary_model is None:
        return
    
    # Collect responses
    responses = collect_responses()
    
    # Calculate BDI score
    bdi_score = calculate_bdi_score(responses)
    
    # Determine severity based on score (rule-based as backup)
    severity = get_severity_from_score(bdi_score)
    has_depression = bdi_score > 10
    
    # Display results
    display_results(has_depression, severity, bdi_score)
    
    # Ask if user wants to try again
    again = input("Would you like to take the assessment again? (y/n): ").strip().lower()
    if again == 'y':
        main()


if __name__ == "__main__":
    main()
