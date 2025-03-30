import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import ta

# 회사명과 티커 매핑 (한국어 및 글로벌 기업)
def get_ticker_from_name(company_name):
    company_dict = {
        # 기술 분야 (Tech)
        "애플": "AAPL", "구글": "GOOG", "마이크로소프트": "MSFT", "아마존": "AMZN",
        "테슬라": "TSLA", "메타": "META", "엔비디아": "NVDA", "어도비": "ADBE",
        "인텔": "INTC", "트위터": "TWTR", "넷플릭스": "NFLX", "스냅": "SNAP",
        "줌": "ZM", "스포티파이": "SPOT", "슬랙": "WORK", "쇼피파이": "SHOP",
        # 자동차 (Automotive)
        "토요타": "7203.T", "포드": "F", "제너럴모터스": "GM", "BMW": "BMW.DE",
        "폭스바겐": "VOW3.DE", "혼다": "7267.T", "닛산": "7201.T", "피아트 크라이슬러": "FCAU",
        "포르쉐": "P911", "현대차": "005380.KS", "기아차": "000270.KS",
        # 헬스케어 (Healthcare)
        "존슨앤드존슨": "JNJ", "화이자": "PFE", "머크": "MRK", "노바티스": "NVS",
        "로슈": "ROG.SW", "아스트라제네카": "AZN", "애브비": "ABBV", "브리스톨 마이어스": "BMY",
        "애보트": "ABT", "일라이 릴리": "LLY", "메드트로닉": "MDT", "GSK": "GSK",
        # 소비재 (Consumer Goods)
        "코카콜라": "KO", "펩시코": "PEP", "나이키": "NKE", "맥도날드": "MCD",
        "유니레버": "ULVR.L", "프록터앤드갬블": "PG", "콜게이트-팜올리브": "CL",
        "레킷벤키저": "RB", "네슬레": "NESN.SW", "로레알": "OR.PA", "킴벌리클락": "KMB",
        # 금융 (Finance)
        "골드만삭스": "GS", "JP모건체이스": "JPM", "뱅크오브아메리카": "BAC",
        "모건스탠리": "MS", "씨티그룹": "C", "웰스파고": "WFC", "HSBC": "HSBC",
        "아메리칸익스프레스": "AXP", "버크셔해서웨이": "BRK.A", "비자": "V",
        "마스터카드": "MA", "페이팔": "PYPL", "찰스슈왑": "SCHW",
        # 에너지 (Energy)
        "엑슨모빌": "XOM", "쉐브론": "CVX", "BP": "BP", "로열더치쉘": "RDS.A",
        "토탈에너지": "TOTF.PA", "코노코필립스": "COP", "슐럼버거": "SLB", "할리버튼": "HAL",
        "에니": "ENI.MI", "에퀴노르": "EQNR.OL",
        # 통신 (Telecom)
        "AT&T": "T", "버라이즌": "VZ", "차이나모바일": "0941.HK", "보더폰": "VOD",
        "T모바일US": "TMUS", "오렌지": "ORA.PA", "텔레포니카": "TEF.MC",
        # 소매 (Retail)
        "월마트": "WMT", "코스트코": "COST", "타겟": "TGT", "홈디포": "HD",
        "로우스": "LOW", "알리바바": "BABA", "JD닷컴": "JD", "자라(인디텍스)": "ITX.MC", "H&M": "HM-B.ST",
        # 기타 글로벌 기업들
        "텐센트": "0700.HK", "소니": "6758.T", "바이두": "BIDU", "노키아": "NOK",
        "퀄컴": "QCOM", "소프트뱅크": "9984.T", "LVMH": "MC.PA", "지멘스": "SIE.DE",
        "폭스바겐": "VOW3.DE", "디아지오": "DGE.L", "BASF": "BAS.DE", "화웨이": "002502.SZ",
        # 한국 기업들 (Korean Companies)
        "삼성전자": "005930.KS", "현대차": "005380.KS", "SK하이닉스": "000660.KS",
        "LG전자": "066570.KS", "삼성바이오로직스": "207940.KS", "POSCO": "005490.KS",
        "카카오": "035720.KS", "네이버": "035420.KS", "셀트리온": "068270.KS", "현대모비스": "012330.KS",
        "롯데화학": "011170.KS", "한화솔루션": "009830.KS", "삼성물산": "028260.KS",
        "KB금융": "105560.KS", "신한금융지주": "055550.KS", "삼성생명": "032830.KS",
        "하나금융지주": "086790.KS", "SK텔레콤": "017670.KS", "LG화학": "051910.KS",
        "아모레퍼시픽": "090430.KS", "CJ ENM": "035760.KS", "S-Oil": "010950.KS",
        "KCC": "002380.KS", "SK이노베이션": "096770.KS", "대우조선해양": "042660.KS",
        "대한항공": "003490.KS", "POSCO인터내셔널": "047050.KS", "현대중공업": "009540.KS"
    }
    return company_dict.get(company_name, None)

# 주식 데이터 다운로드 함수
def get_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# 데이터 준비 함수
def prepare_data(data, time_step=60):
    data = data['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i-time_step:i, 0])
        y.append(scaled_data[i, 0])
    
    X = np.array(X)
    y = np.array(y)
    
    # LSTM 입력 형태로 3D 배열 변환
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler

# LSTM 모델 구성 함수
def create_model(input_shape):
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=input_shape),
        LSTM(100, return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# 모델 훈련 함수
def train_model(X_train, y_train):
    model = create_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    return model

# 예측 함수
def predict_stock_price(model, X_test, scaler):
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    return predictions

# 주식 예측 실행 함수
def run_stock_prediction(company_name, start_date='2015-01-01', end_date='2023-01-01'):
    ticker = get_ticker_from_name(company_name)
    if ticker is None:
        return None, None, None, f"회사 '{company_name}'에 대한 티커를 찾을 수 없습니다."
