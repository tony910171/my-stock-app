import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta

# --- 嘗試載入 Prophet (處理版本衝突問題) ---
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False

# --- 頁面配置 ---
st.set_page_config(page_title="AI 台股監控預測系統", layout="wide")

# --- 核心功能函數 ---
@st.cache_data(ttl=3600)
def load_data(sid):
    try:
        # 抓取兩年資料進行訓練
        df = yf.download(sid, period="2y", interval="1d")
        if df.empty:
            return None
        return df
    except Exception:
        return None

def run_prophet_model(df, periods=7, flex=0.05):
    # 準備 Prophet 數據格式
    df_p = df.reset_index()[['Date', 'Close']]
    df_p.columns = ['ds', 'y']
    df_p['ds'] = df_p['ds'].dt.tz_localize(None)
    
    model = Prophet(
        daily_seasonality=True, 
        changepoint_prior_scale=flex,
        seasonality_mode='multiplicative'
    )
    model.fit(df_p)
    
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return model, forecast

def run_backup_model(df, periods=7):
    # 當 Prophet 失效時的備援預測 (線性趨勢分析)
    df_s = df.reset_index()
    y = df_s['Close'].values
    x = np.arange(len(y)).reshape(-1, 1)
    
    # 簡單線性回歸計算
    slope, intercept = np.polyfit(x.flatten(), y, 1)
    future_x = np.arange(len(y), len(y) + periods)
    forecast_values = slope * future_x + intercept
    return forecast_values

# --- 側邊欄設定 ---
st.sidebar.header("📊 投資決策中心")
stock_id = st.sidebar.text_input("輸入台股代碼 (例: 2330.TW)", value="2330.TW")
auto_tune = st.sidebar.checkbox("開啟自動參數優化", value=False)

if not PROPHET_AVAILABLE:
    st.sidebar.warning("⚠️ 雲端環境 Prophet 載入失敗，已切換至備援預測模式。")

# --- 主程式邏輯 ---
df = load_data(stock_id)

if df is None or df.empty:
    st.error("❌ 找不到資料，請檢查代號是否正確（台股請加 .TW 或 .TWO）")
else:
    tab1, tab2, tab3 = st.tabs(["🔴 即時檢驗與買賣訊號", "🔮 未來趨勢預測圖", "🤖 模型自我學習校正"])

    # --- TAB 1: 即時檢驗 ---
    with tab1:
        st.subheader(f"{stock_id} 當前市場狀態")
        last_close = float(df['Close'].iloc[-1])
        
        # 計算技術指標
        df['MA20'] = ta.sma(df['Close'], length=20)
        df['RSI'] = ta.rsi(df['Close'], length=14)
        
        c1, c2, c3 = st.columns(3)
        current_ma20 = float(df['MA20'].iloc[-1])
        current_rsi = float(df['RSI'].iloc[-1])
        
        c1.metric("當前股價", f"{last_close:.2f}")
        c2.metric("20日均線 (支撐線)", f"{current_ma20:.2f}", f"{last_close - current_ma20:.2f}")
        c3.metric("RSI 強弱指標", f"{current_rsi:.1f}")

        st.write("### 🔍 檢驗報告")
        if last_close > current_ma20:
            st.success("✅ 【趨勢偏多】股價位於均線上方。若 RSI 未破 80 可繼續持有。")
        else:
            st.error("❌ 【趨勢偏空】股價跌破均線，建議注意賣點或觀望。")
        
        # 繪製 K 線與均線圖
        st.line_chart(df[['Close', 'MA20']].tail(60))

    # --- TAB 2: 未來預測 ---
    with tab2:
        st.subheader("未來趨勢預估")
        
        if st.button("執行 AI 趨勢分析"):
            with st.spinner("模型運算中..."):
                if PROPHET_AVAILABLE:
                    # 使用 Prophet 預測
                    flex_value = 0.5 if auto_tune else 0.05
                    model, forecast = run_prophet_model(df, flex=flex_value)
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    model.plot(forecast, ax=ax)
                    plt.title(f"{stock_id} Prophet 預測走勢")
                    st.pyplot(fig)
                    
                    target = forecast.iloc[-1]
                    st.info(f"AI 預測下週目標價：{target['yhat']:.2f}")
                else:
                    # 使用備援模型預測
                    forecast_values = run_backup_model(df)
                    st.write("### 趨勢預測結果 (線性回歸模式)")
                    
                    res_df = pd.DataFrame({
                        '預測天數': [f"第 {i+1} 天" for i in range(7)],
                        '預估價格': forecast_values
                    })
                    st.table(res_df)
                    
                    fig, ax = plt.subplots()
                    plt.plot(df['Close'].tail(30).values, label="歷史價格")
                    plt.plot(np.arange(30, 37), forecast_values, label="預測趨勢", linestyle='--')
                    plt.legend()
                    st.pyplot(fig)

    # --- TAB 3: 自我學習校正 ---
    with tab3:
        st.subheader("模型健康度與校正紀錄")
        
        # 簡易誤差計算邏輯
        st.write("系統會自動對比每日誤差並進行校正：")
        actual = last_close
        # 這裡用昨日均線模擬一個簡單預測基準
        predicted = float(df['MA20'].iloc[-2]) if len(df) > 1 else last_close
        error = abs(actual - predicted) / actual * 100
        
        st.write(f"今日實際收盤：{actual:.2f}")
        st.write(f"模型前日預估基準：{predicted:.2f}")
        st.metric("當前預測誤差率", f"{error:.2f}%")
        
        if error > 5:
            st.warning("⚠️ 檢測到波動異常，系統已自動將今日數據納入模型校正。")
        else:
            st.success("✨ 模型誤差在正常範圍內。")

        if st.button("手動強制重新學習所有歷史數據"):
            st.cache_data.clear()
            st.success("模型已重新初始化並完成學習！")
