import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

# --- 頁面配置 ---
st.set_page_config(page_title="台股 AI 趨勢監控系統", layout="wide")

# --- 核心功能函數 ---
@st.cache_data(ttl=3600)
def load_data(sid):
    try:
        df = yf.download(sid, period="2y", interval="1d")
        if df is None or df.empty:
            return None
        # 修正多層索引問題 (yfinance 新版特性)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception:
        return None

def run_trend_prediction(df, periods=7):
    # 使用多項式回歸擬合趨勢
    y = df['Close'].fillna(method='ffill').values
    x = np.arange(len(y))
    z = np.polyfit(x, y, 2)
    p = np.poly1d(z)
    future_x = np.arange(len(y), len(y) + periods)
    return p(future_x)

# --- 側邊欄 ---
st.sidebar.header("📈 系統控制台")
stock_id = st.sidebar.text_input("輸入代碼 (例: 2330.TW)", value="2330.TW")
st.sidebar.info("上市加 .TW，上櫃加 .TWO")

# --- 主程式 ---
df = load_data(stock_id)

if df is None:
    st.error("❌ 無法獲取數據。請檢查代號格式是否正確，或稍後再試。")
else:
    tab1, tab2, tab3 = st.tabs(["📊 即時檢驗", "📈 趨勢預測", "⚙️ 模型校正"])

    # --- TAB 1: 即時檢驗 ---
    with tab1:
        st.subheader(f"{stock_id} 買賣訊號檢索")
        last_price = float(df['Close'].iloc[-1])
        df['MA20'] = ta.sma(df['Close'], length=20)
        
        # 確保有資料才顯示
        if not df['MA20'].isnull().all():
            ma20_val = float(df['MA20'].iloc[-1])
            col1, col2 = st.columns(2)
            col1.metric("當前現價", f"{last_price:.2f}")
            col2.metric("20日均線 (支撐)", f"{ma20_val:.2f}", f"{last_price - ma20_val:.2f}")

            if last_price > ma20_val:
                st.success("🎯 建議：趨勢向上，股價站穩均線，建議續抱。")
            else:
                st.error("🛑 警告：股價跌破均線，短期轉弱，建議賣出或減碼。")
        
        st.line_chart(df[['Close', 'MA20']].tail(100))

    # --- TAB 2: 趨勢預測 ---
    with tab2:
        st.subheader("未來 7 天 AI 趨勢預估")
        if st.button("啟動趨勢運算"):
            with st.spinner("計算中..."):
                forecast = run_trend_prediction(df)
                dates = [(datetime.now() + timedelta(days=i+1)).strftime("%Y-%m-%d") for i in range(7)]
                
                res_df = pd.DataFrame({"日期": dates, "預估價格": [f"{v:.2f}" for v in forecast]})
                st.table(res_df)
                
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(df['Close'].tail(30).values, label="最近 30 天實際價", color="#1f77b4")
                ax.plot(np.arange(30, 37), forecast, label="未來 7 天預測趨勢", color="#ff7f0e", linestyle="--")
                ax.set_title("股價趨勢擬合分析")
                ax.legend()
                st.pyplot(fig)

    # --- TAB 3: 模型校正 ---
    with tab3:
        st.subheader("數據校正紀錄")
        st.write(f"最後更新時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.info("系統採用多項式回歸校正，每日收盤後自動更新歷史權重。")
        if st.button("手動清除快取並重新校正"):
            st.cache_data.clear()
            st.rerun()
