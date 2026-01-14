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
        # 使用更穩定的抓取參數
        ticker = yf.Ticker(sid)
        df = ticker.history(period="2y", interval="1d")
        
        if df.empty:
            return None
        
        # 處理新版 yfinance 可能產生的 MultiIndex 或欄位名稱問題
        df = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # 強制確保需要的欄位存在且為正確格式
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        return df
    except Exception as e:
        st.sidebar.error(f"連線異常: {e}")
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
stock_id = st.sidebar.text_input("輸入代碼 (例: 2330.TW)", value="2330.TW").upper()
st.sidebar.info("上市加 .TW，上櫃加 .TWO")

# --- 主程式 ---
df = load_data(stock_id)

if df is None:
    st.error(f"❌ 無法獲取 {stock_id} 的數據。")
    st.info("請檢查：\n1. 代號是否包含 .TW (如 2330.TW)\n2. 網路環境是否正常\n3. 嘗試在側邊欄手動輸入其他代號")
else:
    tab1, tab2, tab3 = st.tabs(["🔴 即時檢驗", "📈 趨勢預測", "⚙️ 模型校正"])

    # --- TAB 1: 即時檢驗 ---
    with tab1:
        st.subheader(f"{stock_id} 買賣訊號檢索")
        last_price = float(df['Close'].iloc[-1])
        
        # 計算 20 日均線 (月線)
        df['MA20'] = ta.sma(df['Close'], length=20)
        
        if not df['MA20'].isnull().all():
            ma20_val = float(df['MA20'].iloc[-1])
            col1, col2 = st.columns(2)
            col1.metric("當前現價", f"{last_price:.2f}")
            col2.metric("20日均線 (支撐)", f"{ma20_val:.2f}", f"{last_price - ma20_val:.2f}")

            if last_price > ma20_val:
                st.success("🎯 建議：趨勢向上，股價站穩均線，建議續抱。")
            else:
                st.error("🛑 警告：股價跌破均線，短期轉弱，建議賣出或減碼。")
        
        # 視覺化歷史走勢
        st.line_chart(df[['Close', 'MA20']].tail(100))

    # --- TAB 2: 趨勢預測 ---
    with tab2:
        st.subheader("未來 7 天趨勢預估 (AI 擬合)")
        if st.button("啟動趨勢運算"):
            with st.spinner("計算中..."):
                forecast = run_trend_prediction(df)
                # 生成未來日期 (排除週末)
                dates = []
                current_date = datetime.now()
                while len(dates) < 7:
                    current_date += timedelta(days=1)
                    if current_date.weekday() < 5: # 0-4 為週一至週五
                        dates.append(current_date.strftime("%Y-%m-%d"))
                
                res_df = pd.DataFrame({"日期": dates, "預估價格": [f"{v:.2f}" for v in forecast]})
                st.table(res_df)
                
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(df['Close'].tail(30).values, label="最近 30 天實際價", color="#1f77b4", marker='o')
                ax.plot(np.arange(30, 37), forecast, label="未來 7 天預測趨勢", color="#ff7f0e", linestyle="--", marker='s')
                ax.set_title("股價動能擬合分析")
                ax.legend()
                st.pyplot(fig)

    # --- TAB 3: 模型校正 ---
    with tab3:
        st.subheader("數據校正紀錄")
        st.write(f"資料筆數: {len(df)} 筆")
        st.write(f"最後抓取時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        st.info("系統採用多項式回歸 (Polynomial Regression)，每天啟動時會自動校正權重。")
        if st.button("手動清除快取並重新校正"):
            st.cache_data.clear()
            st.rerun()
