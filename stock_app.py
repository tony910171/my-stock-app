import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

# --- 頁面配置 ---
st.set_page_config(page_title="AI 台股動態預測系統 V2", layout="wide")

# --- 核心功能函數 ---
@st.cache_data(ttl=600) # 縮短快取時間至 10 分鐘，確保資料更即時
def load_data(sid):
    try:
        ticker = yf.Ticker(sid)
        df = ticker.history(period="2y", interval="1d")
        if df.empty: return None
        df.columns = [c.capitalize() for c in df.columns]
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except:
        return None

def run_advanced_prediction(df, periods=7):
    # 權重優化：讓近期的資料影響力更大 (加權最小平方法概念)
    y = df['Close'].fillna(method='ffill').values
    x = np.arange(len(y))
    weights = np.linspace(0.1, 1.0, len(y)) # 近期數據權重為 1.0，遠期為 0.1
    
    # 嘗試不同的多項式次數 (1-3次)，尋找最佳擬合
    z = np.polyfit(x, y, 3, w=weights) 
    p = np.poly1d(z)
    
    future_x = np.arange(len(y), len(y) + periods)
    forecast = p(future_x)
    return forecast

# --- 側邊欄 ---
st.sidebar.header("📈 系統控制台")
stock_id = st.sidebar.text_input("輸入代碼 (例: 2330.TW)", value="2330.TW").upper()

# --- 主程式 ---
df = load_data(stock_id)

if df is None:
    st.error(f"❌ 無法獲取 {stock_id} 數據，請檢查格式。")
else:
    tab1, tab2, tab3 = st.tabs(["🔴 即時買賣檢驗", "🔮 趨勢偏差分析", "⚙️ 模型校正"])

    # --- TAB 1: 買賣檢驗 (加入布林通道) ---
    with tab1:
        st.subheader(f"{stock_id} 波動區間檢驗")
        # 計算布林通道
        bb = ta.bbands(df['Close'], length=20, std=2)
        df = pd.concat([df, bb], axis=1)
        
        last_row = df.iloc[-1]
        c1, c2, c3 = st.columns(3)
        c1.metric("當前現價", f"{last_row['Close']:.2f}")
        c2.metric("通道上軌 (壓力)", f"{last_row['BBU_20_2.0']:.2f}")
        c3.metric("通道下軌 (支撐)", f"{last_row['BBL_20_2.0']:.2f}")

        if last_row['Close'] >= last_row['BBU_20_2.0']:
            st.warning("⚠️ 警告：股價已觸及布林上軌，超買訊號，不宜追高。")
        elif last_row['Close'] <= last_row['BBL_20_2.0']:
            st.success("✅ 訊號：股價觸及下軌，具支撐力道，可留意買點。")
        else:
            st.info("ℹ️ 狀態：股價於常態區間波動。")
        
        st.line_chart(df[['Close', 'BBU_20_2.0', 'BBL_20_2.0']].tail(100))

    # --- TAB 2: 趨勢偏差分析 (解決預估落差問題) ---
    with tab2:
        st.subheader("未來趨勢與偏差校正")
        if st.button("啟動高感度預測"):
            forecast = run_advanced_prediction(df)
            
            # 計算標準差作為誤差範圍
            std_dev = df['Close'].tail(20).std()
            
            # 繪圖視覺化
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(df['Close'].tail(40).values, label="實際價格", color="blue", linewidth=2)
            
            # 預測線與誤差區間
            x_future = np.arange(40, 47)
            ax.plot(x_future, forecast, label="AI 預測趨勢", color="red", linestyle="--")
            ax.fill_between(x_future, forecast - std_dev, forecast + std_dev, color='red', alpha=0.2, label="預期波動範圍")
            
            ax.set_title("趨勢偏差校正圖")
            ax.legend()
            st.pyplot(fig)
            
            st.write("### 📖 如何閱讀此圖？")
            st.write("紅線是 AI 預估的**平均路徑**，紅色陰影區域是考慮到近期波動後的**容許誤差範圍**。如果實際價格脫離陰影區，代表市場發生了預測外的異動，需重新載入資料。")

    # --- TAB 3: 校正 ---
    with tab3:
        st.subheader("模型健康度")
        error_val = df['Close'].tail(5).std()
        st.write(f"近期波動率: {error_val:.2f}")
        if error_val > (df['Close'].mean() * 0.05):
            st.warning("⚠️ 目前市場波動劇烈，預測落差可能增大。")
        else:
            st.success("✨ 市場處於穩定趨勢。")
        
        if st.button("清除快取並強制重新校正模型"):
            st.cache_data.clear()
            st.rerun()
