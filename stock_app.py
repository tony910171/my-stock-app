import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

# --- 頁面配置 ---
st.set_page_config(page_title="AI 台股監控預測系統", layout="wide")

# --- 核心功能函數 ---
@st.cache_data(ttl=3600)
def load_data(sid):
    try:
        # 抓取資料
        data = yf.download(sid, period="2y", interval="1d")
        if data.empty:
            return None
        
        # 處理新版 yfinance 可能產生的 MultiIndex 欄位
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            
        return data
    except Exception as e:
        st.sidebar.error(f"資料抓取錯誤: {e}")
        return None

def run_trend_prediction(df, periods=7):
    # 使用多項式回歸擬合趨勢 (不需要額外安裝 Prophet)
    y = df['Close'].fillna(method='ffill').values
    x = np.arange(len(y))
    z = np.polyfit(x, y, 2) # 二次曲線擬合
    p = np.poly1d(z)
    future_x = np.arange(len(y), len(y) + periods)
    return p(future_x)

# --- 側邊欄 ---
st.sidebar.header("📈 系統控制台")
stock_id = st.sidebar.text_input("輸入代碼 (例: 2330.TW)", value="2330.TW")
st.sidebar.info("上市請加 .TW \n上櫃請加 .TWO")

# --- 主程式邏輯 ---
df = load_data(stock_id)

if df is None:
    st.error("❌ 無法獲取數據。請檢查：1.代號是否正確 2.網路環境 3.GitHub 配置")
else:
    tab1, tab2, tab3 = st.tabs(["🔴 即時檢驗與買賣訊號", "🔮 未來趨勢預測圖", "🤖 模型自我學習校正"])

    # --- TAB 1: 即時檢驗 ---
    with tab1:
        st.subheader(f"{stock_id} 當前市場檢驗")
        last_close = float(df['Close'].iloc[-1])
        
        # 計算技術指標
        df['MA20'] = ta.sma(df['Close'], length=20)
        
        col1, col2 = st.columns(2)
        if not df['MA20'].isnull().all():
            current_ma20 = float(df['MA20'].iloc[-1])
            col1.metric("當前股價", f"{last_close:.2f}")
            col2.metric("20日均線 (支撐線)", f"{current_ma20:.2f}", f"{last_close - current_ma20:.2f}")

            st.write("### 🔍 檢驗報告")
            if last_close > current_ma20:
                st.success("✅ 【趨勢偏多】股價位於均線上方，建議續抱。")
            else:
                st.error("❌ 【趨勢偏空】股價跌破均線，短期轉弱，建議減碼。")
        
        # 顯示近 100 天走勢
        st.line_chart(df[['Close', 'MA20']].tail(100))

    # --- TAB 2: 未來預測 ---
    with tab2:
        st.subheader("未來 7 天 AI 趨勢預估")
        if st.button("啟動趨勢運算"):
            with st.spinner("正在進行大數據擬合..."):
                forecast = run_trend_prediction(df)
                
                # 建立展示表格
                dates = [(datetime.now() + timedelta(days=i+1)).strftime("%Y-%m-%d") for i in range(7)]
                res_df = pd.DataFrame({"預測日期": dates, "預估價格": [f"{v:.2f}" for v in forecast]})
                st.table(res_df)
                
                # 繪圖
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(df['Close'].tail(30).values, label="歷史收盤價", color="#1f77b4")
                ax.plot(np.arange(30, 37), forecast, label="預測趨勢線", color="#ff7f0e", linestyle="--", marker='o')
                ax.set_title("股價動能預測分析")
                ax.legend()
                st.pyplot(fig)

    # --- TAB 3: 模型校正 ---
    with tab3:
        st.subheader("🤖 模型自動化管理")
        st.write(f"目前數據點總數: {len(df)}")
        st.write(f"最後校正時間: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        st.info("系統採用多項式回歸 (Polynomial Regression)，每天啟動時會自動將最新價格加入權重重新校正。")
        
        if st.button("手動清除快取並重新學習"):
            st.cache_data.clear()
            st.rerun()
