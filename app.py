# app.py

import streamlit as st
import pandas as pd
import subprocess
import sys
import tempfile
import os

# 1. Page config + title
st.set_page_config(page_title="Time‑Series CLI Runner", layout="wide")
st.title("📁 Upload a CSV and Run moirai.py or Kronos.py")

st.markdown(
    """
    1. Upload a CSV file containing your time‑series.  
    2. Choose which script to run (moirai.py or Kronos.py).  
    3. The selected script will be invoked as if you ran `python <script> <uploaded.csv>`.  
    4. We capture stdout/stderr and display it below.
    """
)

# 2. File uploader widget
uploaded_file = st.file_uploader(
    "Step 1: Upload your CSV (e.g. time_series.csv)", 
    type=["csv"]
)

# 3. Dropdown to pick which CLI script to run
script_choice = st.selectbox(
    "Step 2: Select which script to run on the uploaded CSV",
    ("moirai.py", "Kronos.py")
)

# 4. A “Run” button
run_button = st.button("🚀 Run Selected Script")

# 5. Only proceed if a file is uploaded and Run is clicked
if uploaded_file is not None and run_button:
    try:
        # 5.1. Save uploaded File to a Temporary File on disk
        # We need a “real” file because subprocess cannot read an in‑memory BytesIO.
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp_path = tmp.name
            tmp.write(uploaded_file.read())
            tmp.flush()
        st.info(f"✔️ Saved uploaded CSV to `{tmp_path}`")

        # 5.2. Build the command that we’d run on the command line
        #    e.g. `python moirai.py /path/to/tmp.csv`
        python_executable = sys.executable  # ensures we call the same Python interpreter
        script_path = os.path.join(os.getcwd(), script_choice)
        cmd = [python_executable, script_path, tmp_path]

        st.text(f"Running command:\n```\n{cmd}\n```")

        # 5.3. Call the script via subprocess, capturing stdout and stderr
        #     We set text=True so stdout/stderr come back as strings
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False  # we’ll handle nonzero return codes manually
        )

        # 5.4. Show the return code, stdout, and stderr
        st.subheader("🔄 Script Return Code")
        st.code(str(result.returncode))

        st.subheader("📤 stdout (script print output)")
        if result.stdout.strip() == "":
            st.write("_No stdout was printed by the script._")
        else:
            st.code(result.stdout)

        st.subheader("🐛 stderr (any errors/warnings)")
        if result.stderr.strip() == "":
            st.write("_No stderr output._")
        else:
            # Display stderr in red to emphasize errors
            st.code(result.stderr)

    except Exception as e:
        st.error(f"❌ An exception occurred while running the script:\n```\n{e}\n```")
    finally:
        # 5.5. Clean up the temporary file if you want. 
        #       If you want to inspect it (e.g. in logs), you can comment this out.
        try:
            os.remove(tmp_path)
        except Exception:
            pass

else:
    # If no file is uploaded or button hasn't been clicked, show a hint
    if uploaded_file is None:
        st.info("⚠️ Please upload a CSV file first.")
    else:
        st.info("ℹ️ Click the `Run Selected Script` button after choosing your file and script.")
