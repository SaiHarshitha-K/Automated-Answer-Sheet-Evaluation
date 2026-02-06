# utils/prn_utils.py

def normalize_prn(prn: str) -> str:
    """
    Convert PRN to 3-digit string for filenames.
    Rules:
    - 9   -> 009
    - 12  -> 012
    - 145 -> 145
    - 1234 -> 234 (last 3 digits)
    """
    if not prn:
        return "000"

    digits = "".join(c for c in prn if c.isdigit())

    if not digits:
        return "000"

    if len(digits) >= 3:
        return digits[-3:]

    return digits.zfill(3)
