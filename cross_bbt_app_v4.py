
# cross_bbt_app_v4.py
# Streamlit Cross-Branch Balance & Transfer Planner (BBT)
# - Donor whitelist + donor priority by branch OverRatio
# - Recipient whitelist + taker priority (Need/NeedRatio/Demand)

from pathlib import Path
import re
import io
import pandas as pd
import numpy as np
import streamlit as st

ROOT = Path(__file__).parent

@st.cache_data(show_spinner=False)
def read_excel_safe(src) -> pd.DataFrame:
    name = getattr(src, "name", None) or (str(src) if isinstance(src, (str, Path)) else "uploaded_file")
    try:
        return pd.read_excel(src)
    except Exception as e:
        st.warning(f"⚠️ تعذرت قراءة الملف: {name} — {e}")
        return pd.DataFrame()

def pick(df: pd.DataFrame, options):
    for o in options:
        if o in df.columns:
            return o
    lower_map = {c.strip().lower(): c for c in df.columns}
    for o in options:
        key = o.strip().lower()
        if key in lower_map:
            return lower_map[key]
    return None

@st.cache_data(show_spinner=False)
def discover_branch_files():
    pattern_sales = re.compile(r"^(p\d{4})\s+sales\.xlsx$", re.IGNORECASE)
    pattern_stock = re.compile(r"^(p\d{4})\s+stock\.xlsx$", re.IGNORECASE)

    sales = {}
    stock = {}
    for p in ROOT.glob("*.xlsx"):
        m = pattern_sales.match(p.name)
        if m:
            sales[m.group(1).upper()] = p
        m = pattern_stock.match(p.name)
        if m:
            stock[m.group(1).upper()] = p

    branches = sorted(set(sales) & set(stock))
    pairs = {b: {"sales": sales[b], "stock": stock[b]} for b in branches}
    return branches, pairs

def parse_uploaded_files(files):
    if not files:
        return [], {}
    pattern_sales = re.compile(r"^(p\d{4})\s+sales\.xlsx$", re.IGNORECASE)
    pattern_stock = re.compile(r"^(p\d{4})\s+stock\.xlsx$", re.IGNORECASE)

    sales = {}
    stock = {}
    for f in files:
        if not hasattr(f, "name"): 
            continue
        name = f.name.strip()
        ms = pattern_sales.match(name)
        mt = pattern_stock.match(name)
        if ms:
            sales[ms.group(1).upper()] = f
        if mt:
            stock[mt.group(1).upper()] = f

    branches = sorted(set(sales) & set(stock))
    pairs = {b: {"sales": sales[b], "stock": stock[b]} for b in branches}
    return branches, pairs

# ---------------- Planner Core ----------------
def build_branch_stats(branch_id: str, paths: dict, sales_days: int) -> pd.DataFrame:
    sdf = read_excel_safe(paths["sales"])
    tdf = read_excel_safe(paths["stock"])

    if sdf.empty and tdf.empty:
        return pd.DataFrame()

    s_item_code = pick(sdf, ["Item Code","item_code","item code","ItemCode"])
    s_item_name = pick(sdf, ["Item Name","item_name","item name","ItemName","Item Name_x","item_name_x"])
    s_qty       = pick(sdf, ["SaleQty","saleqty","sale qty","Sale Qty","Sales Qty","sales qty","SaleQty_2024","SaleQty_2025"])

    t_item_code = pick(tdf, ["Item Code","item_code","item code","ItemCode"])
    t_item_name = pick(tdf, ["Item Name","item_name","item name","ItemName","Item Name_y","item_name_y"])
    t_qty       = pick(tdf, ["Quantity","quantity","qty","Stock Qty","stock qty","stock_qty","StockQty"])

    for need, dfname in [(s_item_code,"Sales:Item Code"),
                         (s_item_name,"Sales:Item Name"),
                         (s_qty,"Sales:SaleQty"),
                         (t_item_code,"Stock:Item Code"),
                         (t_item_name,"Stock:Item Name"),
                         (t_qty,"Stock:Quantity")]:
        if need is None:
            st.error(f"❌ العمود مفقود ({dfname}) في فرع {branch_id}")
            return pd.DataFrame()

    s = (sdf[[s_item_code, s_item_name, s_qty]]
         .rename(columns={s_item_code:"ItemCode", s_item_name:"ItemName", s_qty:"SalesQty"}))
    s["SalesQty"] = pd.to_numeric(s["SalesQty"], errors="coerce").fillna(0)
    s = s.groupby(["ItemCode","ItemName"], as_index=False).agg({"SalesQty":"sum"})

    t = (tdf[[t_item_code, t_item_name, t_qty]]
         .rename(columns={t_item_code:"ItemCode", t_item_name:"ItemName", t_qty:"StockQty"}))
    t["StockQty"] = pd.to_numeric(t["StockQty"], errors="coerce").fillna(0)
    t = t.groupby(["ItemCode","ItemName"], as_index=False).agg({"StockQty":"sum"})

    merged = s.merge(t, on=["ItemCode","ItemName"], how="outer").fillna(0)
    merged["SalesQty"] = merged["SalesQty"].astype(float)
    merged["StockQty"] = merged["StockQty"].astype(float)

    sales_days = max(1, int(sales_days))
    merged["AvgDaily"] = merged["SalesQty"] / sales_days
    merged.insert(0,"Branch", branch_id)
    return merged

def build_universe(pairs: dict, selected_branches, sales_days: int) -> pd.DataFrame:
    frames = []
    for b in selected_branches:
        dfb = build_branch_stats(b, pairs[b], sales_days)
        if not dfb.empty:
            frames.append(dfb)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).fillna(0)

def compute_need_overstock(df: pd.DataFrame, need_days: int, over_days: int):
    target_need = df["AvgDaily"] * need_days
    target_over = df["AvgDaily"] * over_days

    df["NeedQty"] = np.ceil(np.maximum(0, target_need - df["StockQty"])).astype(int)
    df["OverQty"] = np.floor(np.maximum(0, df["StockQty"] - target_over)).astype(int)
    return df

def make_transfers(df: pd.DataFrame,
                   donor_whitelist: set | None = None,
                   donor_over_ratio: dict | None = None,
                   recipient_whitelist: set | None = None,
                   taker_priority: str = "need",
                   taker_ratio_map: dict | None = None,
                   taker_demand_map: dict | None = None,
                   max_per_transfer: int | None = None):
    """
    Greedy per-item matching donors->takers.
    Donors sorted by: donor_over_ratio desc, OverQty desc.
    Takers sorted by: 
      - 'need'        -> NeedQty desc
      - 'need_ratio'  -> ratio desc (from taker_ratio_map)
      - 'demand'      -> demand desc (from taker_demand_map)
    Whitelists restrict who can donate/receive.
    """
    plans = []

    for (icode, iname), chunk in df.groupby(["ItemCode","ItemName"], dropna=False):
        donors_df = chunk[(chunk["OverQty"] > 0)].copy()
        if donor_whitelist is not None:
            donors_df = donors_df[donors_df["Branch"].isin(donor_whitelist)]

        if donors_df.empty:
            continue

        donors_df["DonorRatio"] = donors_df["Branch"].map(donor_over_ratio or {}).fillna(0.0)
        donors_df = donors_df.sort_values(["DonorRatio", "OverQty"], ascending=[False, False])
        donors = donors_df[["Branch","OverQty"]].values.tolist()

        takers_df = chunk[(chunk["NeedQty"] > 0)].copy()
        if recipient_whitelist is not None:
            takers_df = takers_df[takers_df["Branch"].isin(recipient_whitelist)]

        if takers_df.empty:
            continue

        if taker_priority == "need_ratio":
            takers_df["TakerScore"] = takers_df["Branch"].map(taker_ratio_map or {}).fillna(0.0)
        elif taker_priority == "demand":
            takers_df["TakerScore"] = takers_df["Branch"].map(taker_demand_map or {}).fillna(0.0)
        else:  # 'need'
            takers_df["TakerScore"] = takers_df["NeedQty"].astype(float)

        takers_df = takers_df.sort_values(["TakerScore","NeedQty"], ascending=[False, False])
        takers = takers_df[["Branch","NeedQty"]].values.tolist()

        d_i, t_i = 0, 0
        while d_i < len(donors) and t_i < len(takers):
            d_branch, d_qty = donors[d_i]
            t_branch, t_qty = takers[t_i]
            send = min(d_qty, t_qty)
            if max_per_transfer is not None and max_per_transfer > 0:
                send = min(send, int(max_per_transfer))
            if send <= 0:
                break
            plans.append({
                "ItemCode": icode,
                "ItemName": iname,
                "From": d_branch,
                "To": t_branch,
                "Qty": int(send)
            })
            d_qty -= send
            t_qty -= send
            donors[d_i][1] = d_qty
            takers[t_i][1] = t_qty
            if d_qty == 0: d_i += 1
            if t_qty == 0: t_i += 1

    detail = pd.DataFrame(plans)
    if detail.empty:
        return detail, detail

    consolidated = (detail.groupby(["To","ItemCode","ItemName"], as_index=False)
                    .agg({"Qty":"sum"}).sort_values(["To","ItemCode"]))

    detail = (detail.groupby(["From","To","ItemCode","ItemName"], as_index=False)
              .agg({"Qty":"sum"}).sort_values(["From","To","ItemCode"]))
    return detail, consolidated

def to_excel_bytes(**sheets):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        for name, df in sheets.items():
            (df if df is not None else pd.DataFrame()).to_excel(
                writer, sheet_name=name[:31], index=False
            )
    return output.getvalue()

# ---------------- UI ----------------
st.set_page_config(page_title="Cross-BBT Transfers", layout="wide")
st.title("🚚 Cross Branch Balance & Transfer Planner (Streamlit)")
st.caption("فلتر مانحين ومستلمين + أولوية توزيع مرنة (Need / Need Ratio / Demand).")

with st.sidebar:
    st.header("⚙️ الإعدادات")
    src_mode = st.radio("مصدر البيانات", ["ملفات داخل نفس المجلد (Local)", "رفع ملفات (Upload)"], index=0)
    sales_days = st.number_input("عدد أيام فترة المبيعات (لحساب المتوسط اليومي)", min_value=1, max_value=365, value=90, step=1)
    need_days  = st.number_input("مدة الاحتياج (بالأيام)", min_value=1, max_value=365, value=45, step=1)
    over_days  = st.number_input("عتبة الأوفر استوك (بالأيام)", min_value=1, max_value=365, value=60, step=1)
    max_transfer = st.number_input("حد أقصى للكمية لكل تحويل (اختياري)", min_value=0, value=0, step=1, help="0 = بدون حد")

    uploaded = None
    if src_mode == "رفع ملفات (Upload)":
        uploaded = st.file_uploader(
            "ارفع ملفات Excel بصيغة الأسماء: P0005 Sales.xlsx / P0005 Stock.xlsx ...",
            type=["xlsx"],
            accept_multiple_files=True
        )

    run_btn = st.button("🔄 احسب وولِّد خطة التحويلات", type="primary", use_container_width=True)

# determine branches & pairs
if src_mode == "رفع ملفات (Upload)":
    branches, pairs = parse_uploaded_files(uploaded or [])
else:
    branches, pairs = discover_branch_files()

if not branches:
    if src_mode == "رفع ملفات (Upload)":
        st.info("ارفع ملفّات بصيغة الأسماء الصحيحة لكل فرع: `P#### Sales.xlsx` و `P#### Stock.xlsx`.")
    else:
        st.error("لم يتم العثور على ملفات بالصيغة: `P#### Sales.xlsx` و `P#### Stock.xlsx` في نفس المجلد.")
    st.stop()

with st.expander("🔍 الفروع المكتشفة", expanded=False):
    st.write(branches)

with st.sidebar:
    selected = st.multiselect("اختيار الفروع (تحليل فقط)", branches, default=branches)
    donor_selected = st.multiselect("الفروع المسموح الأخذ منها (Donors)", branches, default=branches)
    recipient_selected = st.multiselect("الفروع المسموح الاستلام لها (Recipients)", branches, default=branches)

    prioritize_donors = st.toggle("أولوية المانحين: أعلى نسبة أوفرستوك فرعيًا", value=True)
    taker_priority_label = st.selectbox("أولوية الوجهات (المستلمين)",
                                        ["NeedQty (أكبر كمية)", "Need Ratio % (Need/Stock)", "Demand (AvgDaily)"],
                                        index=0)

if run_btn:
    if not selected:
        st.warning("اختر فرعًا واحدًا على الأقل.")
        st.stop()

    base = build_universe(pairs, selected, sales_days)
    if base.empty:
        st.error("لا توجد بيانات صالحة.")
        st.stop()

    stats = compute_need_overstock(base.copy(), need_days=int(need_days), over_days=int(over_days))

    # -------- Branch summary & ratios --------
    branch_summary = (stats.groupby("Branch", as_index=False)
                      .agg(TotalSalesQty=("SalesQty","sum"),
                           TotalStock=("StockQty","sum"),
                           AvgDaily=("AvgDaily","sum"),
                           Need=("NeedQty","sum"),
                           Over=("OverQty","sum")))

    for c in ["TotalSalesQty","TotalStock","Need","Over"]:
        branch_summary[c] = branch_summary[c].astype(int)

    branch_summary["OverRatio%"] = np.where(branch_summary["TotalStock"]>0,
                                            100 * branch_summary["Over"] / branch_summary["TotalStock"],
                                            0.0).round(2)
    branch_summary["NeedRatio%"] = np.where(branch_summary["TotalStock"]>0,
                                            100 * branch_summary["Need"] / branch_summary["TotalStock"],
                                            0.0).round(2)

    st.subheader("📊 ملخص الفروع")
    st.dataframe(branch_summary.sort_values("OverRatio%", ascending=False), use_container_width=True)

    # -------- View per item --------
    st.subheader("🧮 حسابات الصنف/الفرع")
    col1, col2 = st.columns(2)
    with col1:
        min_over = st.number_input("إظهار الأصناف ذات OverQty على الأقل", min_value=0, value=0, step=1)
    with col2:
        search_item = st.text_input("بحث بالاسم/الكود (اختياري)")
    view_df = stats.copy()
    if min_over > 0:
        view_df = view_df[view_df["OverQty"] >= int(min_over)]
    if search_item:
        s = search_item.strip().lower()
        view_df = view_df[(view_df["ItemName"].str.lower().str.contains(s, na=False)) | (view_df["ItemCode"].astype(str).str.contains(s, na=False))]
    st.dataframe(view_df[["Branch","ItemCode","ItemName","SalesQty","StockQty","AvgDaily","NeedQty","OverQty"]],
                 use_container_width=True)

    # -------- Build ratios/demands dicts --------
    tmp = branch_summary.set_index("Branch")
    donor_ratio = None
    if prioritize_donors:
        donor_ratio = (tmp["Over"] / tmp["TotalStock"].replace(0, np.nan)).to_dict()
        donor_ratio = {k: float(0 if pd.isna(v) else v) for k, v in donor_ratio.items()}

    taker_ratio_map = (tmp["Need"] / tmp["TotalStock"].replace(0, np.nan)).to_dict()
    taker_ratio_map = {k: float(0 if pd.isna(v) else v) for k, v in taker_ratio_map.items()}
    taker_demand_map = tmp["AvgDaily"].astype(float).to_dict()

    # map UI label -> internal code
    taker_priority_code = {"NeedQty (أكبر كمية)":"need",
                           "Need Ratio % (Need/Stock)":"need_ratio",
                           "Demand (AvgDaily)":"demand"}[taker_priority_label]

    cap = None if int(max_transfer) == 0 else int(max_transfer)
    detail, consolidated = make_transfers(stats,
                                          donor_whitelist=set(donor_selected),
                                          donor_over_ratio=donor_ratio,
                                          recipient_whitelist=set(recipient_selected),
                                          taker_priority=taker_priority_code,
                                          taker_ratio_map=taker_ratio_map,
                                          taker_demand_map=taker_demand_map,
                                          max_per_transfer=cap)

    st.subheader("🚚 خطة التحويلات (حسب المصدر→الوجهة→الصنف)")
    if detail.empty:
        st.info("لا توجد تحويلات مطلوبة وفق الإعدادات الحالية.")
    else:
        st.dataframe(detail, use_container_width=True)

    st.subheader("📦 تجميع للصنف لكل وجهة (بدون تكرار)")
    if not consolidated.empty:
        st.dataframe(consolidated, use_container_width=True)

    if not detail.empty or not consolidated.empty:
        xls = to_excel_bytes(Transfers=detail if not detail.empty else pd.DataFrame(),
                             Consolidated=consolidated if not consolidated.empty else pd.DataFrame(),
                             BranchSummary=branch_summary)
        st.download_button("⬇️ تنزيل الخطة Excel", data=xls, file_name="cross_bbt_transfer_plan.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
else:
    st.info("اختَر الإعدادات من الشريط الجانبي واضغط **احسب وولِّد خطة التحويلات**.")
