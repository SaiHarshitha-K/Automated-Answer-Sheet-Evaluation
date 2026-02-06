# course_report.py
import pandas as pd
import streamlit as st
from io import BytesIO

from db_utils import get_course_report


def render_course_report(*, course_id: int, pass_percent: float = 35.0):
    """
    Streamlit UI for Course Report:
    - subject-wise topper, min/max/avg
    - present/absent (based on students table PRNs)
    - fail count and fail rate
    - overall toppers across subjects (sum of scores)
    - Excel export
    """
    report = get_course_report(course_id=course_id, pass_percent=pass_percent)

    if not report or report.get("error"):
        st.error(report.get("error", "Could not generate report"))
        return

    course = report["course"]
    st.subheader(f"📊 Course Report — {course['course_name']} ({course.get('batch_name','')})")

    # --- quick summary cards ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Enrolled", report["total_enrolled"])
    c2.metric("Subjects", report["subjects_total"])
    c3.metric("Subjects with Exam", report["subjects_with_exam"])
    c4.metric("Pass % Threshold", f"{report['pass_percent_threshold']}%")

    st.markdown("---")

    # --- subject-wise table ---
    rows = []
    for s in report["subject_reports"]:
        if not s.get("has_exam"):
            rows.append({
                "Subject": s["subject_name"],
                "Exam": "No",
                "Enrolled": report["total_enrolled"],
                "Present": 0,
                "Absent": report["total_enrolled"],
                "Min": "",
                "Max": "",
                "Avg": "",
                "Fail Count": "",
                "Fail Rate %": "",
                "Topper(s)": "",
            })
            continue

        toppers = s.get("toppers") or []
        topper_txt = ", ".join([f"{t['prn']} ({t['score']})" for t in toppers])

        rows.append({
            "Subject": s["subject_name"],
            "Exam": "Yes",
            "Enrolled": s["enrolled"],
            "Present": s["present"],
            "Absent": s["absent"],
            "Min": s["min"],
            "Max": s["max"],
            "Avg": s["avg"],
            "Fail Count": s["fail_count"],
            "Fail Rate %": s["fail_rate_percent"],
            "Topper(s)": topper_txt,
        })

    df_subjects = pd.DataFrame(rows)
    st.markdown("### 📌 Subject-wise Summary")
    st.dataframe(df_subjects, use_container_width=True)

    # --- hardest subjects ---
    st.markdown("### 🔥 Hardest Subjects (highest fail rate)")
    hardest = report.get("hardest_subjects_by_fail_rate") or []
    if not hardest:
        st.info("No exam data to compute hardest subjects.")
    else:
        df_hard = pd.DataFrame([{
            "Subject": x["subject_name"],
            "Fail Rate %": x["fail_rate_percent"],
            "Fail Count": x["fail_count"],
            "Present": x["present"],
        } for x in hardest])
        st.dataframe(df_hard, use_container_width=True)

    # --- overall toppers ---
    st.markdown("### 🏆 Overall Top 10 (sum across subjects’ latest exams)")
    top = report.get("overall_topper_list") or []
    if not top:
        st.info("No data.")
    else:
        df_top = pd.DataFrame([{
            "PRN": x["prn"],
            "Name": x.get("name", ""),
            "Score Sum": x["score_sum"],
            "Total Sum": x["total_sum"],
            "Overall %": x["overall_percent"],
            "Attempted Subjects": x["attempted_subjects"],
        } for x in top])
        st.dataframe(df_top, use_container_width=True)

    # --- Export Excel ---
    st.markdown("---")
    st.markdown("### ⬇ Download Course Report (Excel)")

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df_subjects.to_excel(writer, index=False, sheet_name="Subject Summary")
        if hardest:
            df_hard.to_excel(writer, index=False, sheet_name="Hardest Subjects")
        if top:
            df_top.to_excel(writer, index=False, sheet_name="Overall Toppers")

    st.download_button(
        "Download Course Report Excel",
        data=buffer.getvalue(),
        file_name=f"{course['course_name']}_course_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
