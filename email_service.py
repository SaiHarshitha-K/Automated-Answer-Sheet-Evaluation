# email_service.py
import os
import smtplib
import time
from datetime import datetime
from email.message import EmailMessage
from typing import Optional, Dict, Any, List

from db_utils import get_subject_name, get_students_for_exam_with_emails

# ==========================================================
# ✅ HARD-CODE CONFIG (as you requested)
# ==========================================================
SENDER_EMAIL = "21h41a0597@bvcits.edu.in"     # <-- admin gmail
APP_PASSWORD = "xuyntvcgdgdmknel"     # <-- gmail app password

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587

EMAIL_ENABLED = True
EMAIL_SLEEP_SECONDS = 0.2

REPORTS_DIR = os.path.join("data", "email_reports")  # optional txt attachments


def _safe_filename(s: str) -> str:
    keep = []
    for ch in (s or ""):
        if ch.isalnum() or ch in ("_", "-", ".", " "):
            keep.append(ch)
    out = "".join(keep).strip().replace(" ", "_")
    return out if out else "report"


def build_student_email_html(
    *,
    student_name: str,
    prn: str,
    subject_name: str,
    published_at: str,
    theory_marks: float,
    theory_out_of: float,
    lab_marks: Optional[float],
    lab_out_of: float,
) -> str:
    total = float(theory_marks or 0.0) + float(lab_marks or 0.0)
    total_out_of = float(theory_out_of) + (float(lab_out_of) if lab_marks is not None else 0.0)
    status = "PASS" if total >= 16.0 else "FAIL"

    lab_line = "Not uploaded / Not applicable" if lab_marks is None else f"{lab_marks:.2f} / {lab_out_of:.0f}"

    html = f"""
    <div style="font-family: Arial, sans-serif; line-height: 1.6;">
      <h2 style="margin:0;">EXAM RESULT REPORT</h2>
      <p style="margin:4px 0; color:#555;">Published At: <b>{published_at}</b></p>

      <hr/>

      <h3 style="margin-bottom:6px;">Student</h3>
      <p style="margin:2px 0;"><b>Name:</b> {student_name}</p>
      <p style="margin:2px 0;"><b>PRN:</b> {prn}</p>

      <h3 style="margin-bottom:6px;">Subject</h3>
      <p style="margin:2px 0;"><b>{subject_name}</b></p>

      <h3 style="margin-bottom:6px;">Marks</h3>
      <table style="border-collapse: collapse; width: 420px;">
        <tr>
          <td style="border:1px solid #ddd; padding:8px;"><b>Theory</b></td>
          <td style="border:1px solid #ddd; padding:8px;">{theory_marks:.2f} / {theory_out_of:.0f}</td>
        </tr>
        <tr>
          <td style="border:1px solid #ddd; padding:8px;"><b>Lab</b></td>
          <td style="border:1px solid #ddd; padding:8px;">{lab_line}</td>
        </tr>
        <tr>
          <td style="border:1px solid #ddd; padding:8px;"><b>Total</b></td>
          <td style="border:1px solid #ddd; padding:8px;"><b>{total:.2f} / {total_out_of:.0f}</b></td>
        </tr>
      </table>

      <h3 style="margin-top:16px;">Status:
        <span style="color:{'#0a7d2c' if status=='PASS' else '#b00020'}; font-weight:bold;">
          {status}
        </span>
      </h3>

      <p style="margin-top:18px; color:#666; font-size: 13px;">
        This is an automated email from Exam Portal. If you find any issue, contact Exam Cell.
      </p>
    </div>
    """
    return html


def generate_student_report_txt(
    *,
    student_name: str,
    prn: str,
    email: str,
    subject_name: str,
    published_at: str,
    theory_marks: float,
    theory_out_of: float,
    lab_marks: Optional[float],
    lab_out_of: float,
) -> str:
    os.makedirs(REPORTS_DIR, exist_ok=True)
    safe_stu = _safe_filename(student_name)
    safe_sub = _safe_filename(subject_name)
    path = os.path.join(REPORTS_DIR, f"{safe_stu}_{prn}_{safe_sub}_report.txt")

    total = float(theory_marks or 0.0) + float(lab_marks or 0.0)
    total_out_of = float(theory_out_of) + (float(lab_out_of) if lab_marks is not None else 0.0)
    status = "PASS" if total >= 16.0 else "FAIL"

    with open(path, "w", encoding="utf-8") as f:
        f.write("EXAM RESULT REPORT\n")
        f.write(f"Published At: {published_at}\n\n")
        f.write(f"Student Name: {student_name}\n")
        f.write(f"PRN: {prn}\n")
        f.write(f"Email: {email}\n\n")
        f.write(f"Subject: {subject_name}\n\n")
        f.write(f"Theory: {theory_marks:.2f} / {theory_out_of:.0f}\n")
        if lab_marks is None:
            f.write("Lab: Not uploaded / Not applicable\n")
        else:
            f.write(f"Lab: {lab_marks:.2f} / {lab_out_of:.0f}\n")
        f.write(f"Total: {total:.2f} / {total_out_of:.0f}\n")
        f.write(f"Status: {status}\n")

    return path


def send_email_html_with_optional_attachment(
    *,
    to_email: str,
    subject: str,
    html_body: str,
    attachment_path: Optional[str] = None,
):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = SENDER_EMAIL
    msg["To"] = to_email

    # Plain text fallback + HTML
    msg.set_content("Your result is published. Please open this email in HTML view for formatted report.")
    msg.add_alternative(html_body, subtype="html")

    if attachment_path and os.path.exists(attachment_path):
        with open(attachment_path, "rb") as f:
            data = f.read()

        # attach as txt
        msg.add_attachment(
            data,
            maintype="text",
            subtype="plain",
            filename=os.path.basename(attachment_path),
        )

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(SENDER_EMAIL, APP_PASSWORD)
        server.send_message(msg)


def send_publish_emails_to_students(
    *,
    exam_id: int,
    subject_id: int,
    theory_out_of: float = 40.0,
    lab_out_of: float = 40.0,
    attach_txt: bool = True,
) -> Dict[str, Any]:

    if not EMAIL_ENABLED:
        return {"ok": False, "error": "Email sending disabled (EMAIL_ENABLED=False)"}

    if not SENDER_EMAIL or not APP_PASSWORD or "YOUR_16_CHAR_APP_PASSWORD" in APP_PASSWORD:
        return {"ok": False, "error": "SMTP credentials not set in email_service.py"}

    subject_name = get_subject_name(subject_id) or f"Subject-{subject_id}"
    published_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    rows = get_students_for_exam_with_emails(exam_id=exam_id, subject_id=subject_id)

    sent = 0
    skipped_no_email = 0
    failed = 0
    errors: List[str] = []

    for r in rows:
        prn = str(r.get("prn") or "").strip()
        student_name = (r.get("student_name") or r.get("exam_name") or "Student").strip()
        email = (r.get("email") or "").strip()

        if not email:
            skipped_no_email += 1
            continue

        score_raw = float(r.get("score") or 0.0)
        total_q = float(r.get("total_questions") or 0.0)

        # Scale to /40 like portal
        theory_marks = (score_raw / total_q * theory_out_of) if total_q > 0 else 0.0

        lab_marks = r.get("lab_marks")
        lab_marks = float(lab_marks) if lab_marks is not None else None

        try:
            html = build_student_email_html(
                student_name=student_name,
                prn=prn,
                subject_name=subject_name,
                published_at=published_at,
                theory_marks=theory_marks,
                theory_out_of=theory_out_of,
                lab_marks=lab_marks,
                lab_out_of=lab_out_of,
            )

            attachment_path = None
            if attach_txt:
                attachment_path = generate_student_report_txt(
                    student_name=student_name,
                    prn=prn,
                    email=email,
                    subject_name=subject_name,
                    published_at=published_at,
                    theory_marks=theory_marks,
                    theory_out_of=theory_out_of,
                    lab_marks=lab_marks,
                    lab_out_of=lab_out_of,
                )

            send_email_html_with_optional_attachment(
                to_email=email,
                subject=f"Results Published: {subject_name}",
                html_body=html,
                attachment_path=attachment_path,
            )

            sent += 1
            time.sleep(EMAIL_SLEEP_SECONDS)

        except Exception as e:
            failed += 1
            errors.append(f"{prn} -> {email} : {e}")

    return {
        "ok": True,
        "subject_id": subject_id,
        "exam_id": exam_id,
        "subject_name": subject_name,
        "sent": sent,
        "skipped_no_email": skipped_no_email,
        "failed": failed,
        "errors": errors[:10],
    }
