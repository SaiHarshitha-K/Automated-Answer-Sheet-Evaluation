# student_wise_report.py
import pandas as pd
import streamlit as st
from io import BytesIO

from db_utils import get_student_wise_report


def render_student_wise_report(
    *,
    prn: str,
    course_id: int,
    theory_max: float = 40.0,
    lab_max: float = 40.0,
    pass_mark: float = 16.0,
):
    report = get_student_wise_report(
        prn=prn,
        course_id=course_id,
        theory_max=theory_max,
        lab_max=lab_max,
        pass_mark=pass_mark,
    )

    if not report or report.get("error"):
        st.error(report.get("error", "Could not generate student report"))
        return

    stu = report["student"]
    summ = report["summary"]

    st.subheader(f"📄 Student Report — {stu['name']} ({stu['prn']})")
    st.caption(f"{stu.get('course_name','')}  |  {stu.get('batch_name','')}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall %", f"{summ['overall_percent']}%")
    c2.metric("Overall Total", f"{summ['overall_total']} / {summ['overall_possible']}")
    c3.metric("Theory Total", summ["total_theory"])
    c4.metric("Lab Total", summ["total_lab"])

    if summ.get("failed_subjects"):
        st.error("❌ Failed Subjects (<16): " + ", ".join(summ["failed_subjects"]))

    if summ.get("below_20_subjects"):
        st.warning("⚠ Study Hard (Total < 20): " + ", ".join(summ["below_20_subjects"]))

    st.markdown("---")

    df = pd.DataFrame(report["subjects"])
    if df.empty:
        st.info("No subjects found for this course.")
        return

    df_view = df.rename(
        columns={
            "subject_name": "Subject",
            "exam_uploaded": "Exam Uploaded?",
            "theory_mark": f"Theory/{int(theory_max)}",
            "lab_mark": f"Lab/{int(lab_max)}",
            "total": "Total",
            "total_possible": "Out Of",
            "percent": "%",
            "status": "Status",
            "rank": "Rank",
            "class_avg_theory": "Class Avg (Theory)",
            "class_min_theory": "Class Min (Theory)",
            "class_max_theory": "Class Max (Theory)",
        }
    )[
        [
            "Subject",
            "Exam Uploaded?",
            f"Theory/{int(theory_max)}",
            f"Lab/{int(lab_max)}",
            "Total",
            "Out Of",
            "%",
            "Status",
            "Rank",
            "Class Avg (Theory)",
            "Class Min (Theory)",
            "Class Max (Theory)",
        ]
    ]

    st.markdown("### 📌 Subject-wise Performance")
    st.dataframe(df_view, use_container_width=True)

    # Quick info
    ab_subjects = df[df["lab_mark"] == "AB"]["subject_name"].tolist()
    if ab_subjects:
        st.info("🧪 Lab Absent (AB): " + ", ".join(ab_subjects))

    na_subjects = df[df["lab_mark"] == "NA"]["subject_name"].tolist()
    if na_subjects:
        st.info("🧪 No Lab Uploaded (NA): " + ", ".join(na_subjects))

    # Excel
    st.markdown("### ⬇ Download Student Report (Excel)")
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        df_view.to_excel(writer, index=False, sheet_name="Student Report")

    st.download_button(
        "Download Student Report Excel",
        data=buffer.getvalue(),
        file_name=f"{stu['prn']}_student_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
