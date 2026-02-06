"""
Utility helper functions for the OMR system
"""

import os
import shutil


def clean_temp_directory(temp_dir):
    """
    Clean temporary files from processing.
    
    Args:
        temp_dir: Path to temporary directory
    """
    if os.path.exists(temp_dir):
        for item in os.listdir(temp_dir):
            item_path = os.path.join(temp_dir, item)
            try:
                if os.path.isfile(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)
            except Exception as e:
                print(f"Error deleting {item_path}: {e}")


def format_answers_for_display(answers):
    """
    Format answers dictionary for clean display.
    
    Args:
        answers: Dictionary of question_num -> answer
        
    Returns:
        Formatted string
    """
    lines = []
    for q_num in sorted(answers.keys()):
        answer = answers[q_num]
        lines.append(f"Q{q_num}: {answer}")
    return "\n".join(lines)


def validate_prn(prn):
    """
    Validate PRN format (basic check).
    
    Args:
        prn: PRN string
        
    Returns:
        Boolean indicating if valid
    """
    if not prn:
        return False
    # Check if contains at least 6 digits
    digit_count = sum(c.isdigit() for c in prn)
    return digit_count >= 6


def validate_name(name):
    """
    Validate name format (basic check).
    
    Args:
        name: Name string
        
    Returns:
        Boolean indicating if valid
    """
    if not name:
        return False
    # Check if contains at least 2 characters
    letter_count = sum(c.isalpha() for c in name)
    return letter_count >= 2