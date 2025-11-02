
# Cross-BBT v4 — One-click Run (Windows)

## Quick Run
1) Double-click: **Run_CrossBBT_v4_quick.bat**
   - Uses your system Python.
   - Installs requirements, then launches Streamlit.
   - The app URL will appear in the terminal (usually http://localhost:8501).

## Isolated Run (recommended)
1) Double-click: **Run_CrossBBT_v4_venv.bat**
   - Creates `.venv` locally (first time only), installs requirements inside it.
   - Launches the app isolated from your system packages.

## Requirements
- Windows 10/11
- Python 3.10+ installed and added to PATH
- Files in the same folder:
  - cross_bbt_app_v4.py
  - requirements.txt
  - (Optional) branch Excel files e.g. `P0005 Sales.xlsx` + `P0005 Stock.xlsx`

## Notes
- If you deploy to Streamlit Cloud, choose `cross_bbt_app_v4.py` as **main file**.
- In the app sidebar you can switch between **Local files** (same folder) or **Upload files**.
