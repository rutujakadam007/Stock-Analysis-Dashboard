import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from datetime import date, timedelta
import streamlit as st

st.title('Stock Data Analysis Dashboard')

st.markdown("""
            <style>
            .stMarkdown p {
                font-size: 25px;
                font-family: "Cambria, Georgia, serif";
            }
            .stMarkdown h1 {
                font-size: 35px;
                font-family: "Cambria, Georgia, serif";
            }
            </style>
            """, unsafe_allow_html=True)

# Input for start and end dates
start_date = st.date_input('Select start date')
end_date = st.date_input('Select end date')

if start_date > end_date:
    st.error("End date must be after start date")
else:
    start_date_MA = start_date - timedelta(days=400)
    
    start_date = start_date.strftime("%Y-%m-%d")
    end_date = end_date.strftime("%Y-%m-%d")
    start_date_MA = start_date_MA.strftime("%Y-%m-%d")
    
    tickers = st.text_input('Enter stock ticker (comma-separated)')
    
    if tickers:
        tickers_list = [ticker.strip() for ticker in tickers.split(',')]
        
        try:
            with st.spinner('Fetching data....'):
                MA_data = yf.download(tickers_list, start=start_date_MA, end=end_date, progress=False)
        except Exception as e:
            st.error(f"Error in fetching data: {e}")
            st.stop()
            
        data = MA_data.loc[start_date:end_date]  # Original data

        if len(tickers_list) > 1:
            data.columns.names = ['Price', 'Ticker']
            data = data.stack(level='Ticker').reset_index()
            melt_data = data.melt(id_vars=['Date', 'Ticker'], var_name='Price', value_name='value')
            pivot_data = melt_data.pivot_table(index=['Date', 'Ticker'], columns='Price', values='value', aggfunc='first')
            stock_data = pivot_data.reset_index()  # Data to work on

            Open_stat = stock_data.groupby('Ticker')['Open']
            st.write('Statistics of opening price')
            st.dataframe(Open_stat.describe())

            Aclose_stat = stock_data.groupby('Ticker')['Adj Close']
            st.write('Statistics of adjusted closing price')
            st.dataframe(Aclose_stat.describe())

            st.markdown("<br>", unsafe_allow_html=True)
            st.write('Adjusted closing price with time')
            fig = px.line(
                stock_data,
                x='Date',
                y='Adj Close',
                color='Ticker'
            )
            fig.update_layout(
                xaxis_title='Date',
                yaxis_title='Adjusted closing price',
                plot_bgcolor='black',
                paper_bgcolor='black',
                font=dict(color='white', size=14),
                width=1000,
                height=400
            )
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='gray', color='white')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='gray', color='white')

            st.plotly_chart(fig, use_container_width=False)

            std_data_adjClose = pd.DataFrame(Aclose_stat.describe())
            st.markdown("<br>", unsafe_allow_html=True)
            st.write('Volatility of Adjusted Closing Price')
            plt.figure(figsize=(10, 4))
            plt.style.use('dark_background')
            sns.barplot(data=std_data_adjClose, x=std_data_adjClose.index, y='std', width=0.4)
            plt.xlabel('Ticker', fontsize=12)
            plt.ylabel('Standard deviation', fontsize=12)
            plt.grid(True, axis='y', color='white')
            st.pyplot(plt)

            # Correlation 
            st.markdown("<br>", unsafe_allow_html=True)
            st.write('Correlation between adjusted close price')
            aclose_pivot = stock_data.pivot(index='Date', columns='Ticker', values='Adj Close')
            corr_matrix = aclose_pivot.corr()

            sns.heatmap(data=corr_matrix, annot=True, cmap='Greens')
            plt.xlabel('')
            plt.ylabel('')
            plt.grid(False)
            st.pyplot(plt)

            # Moving averages
            MA_data.columns.names = ['Price', 'Ticker']
            MA_data = MA_data.stack(level='Ticker').reset_index()
            # Extracting data for Adj Close price for all tickers
            adj_MA_data = MA_data.melt(id_vars=['Date', 'Ticker'], var_name='Price', value_name='value')
            adj_MA_data = adj_MA_data[adj_MA_data['Price']=='Adj Close']

            # Getting ticker values
            ticker_values = adj_MA_data['Ticker'].unique()

            st.markdown("<br>", unsafe_allow_html=True)
            st.write('Adj close price and 50 and 200 days moving average')

            for tick in ticker_values:
                ticker_data = adj_MA_data[adj_MA_data['Ticker']==tick].copy()
                ticker_data = ticker_data.sort_values("Date")
                ticker_data['50_MA'] = ticker_data['value'].rolling(window=50).mean()
                ticker_data['200_MA'] = ticker_data['value'].rolling(window=200).mean()
                ticker_data = ticker_data[ticker_data['Date'] >= pd.to_datetime(start_date)]

                fig = go.Figure()

                fig.add_trace(go.Scatter(x=ticker_data['Date'], y=ticker_data['value'], mode='lines', name='Adj Close', line=dict(color='blue')))
                fig.add_trace(go.Scatter(x=ticker_data['Date'], y=ticker_data['50_MA'], mode='lines', name='50 days MA', line=dict(color='hotpink')))
                fig.add_trace(go.Scatter(x=ticker_data['Date'], y=ticker_data['200_MA'], mode='lines', name='200 days MA', line=dict(color='green')))

                fig.update_layout(
                    title = f'{tick} stock data',
                    xaxis_title = 'Date',
                    yaxis_title = 'Price',
                    legend=dict(title='Legend'),
                    plot_bgcolor='black',
                    paper_bgcolor='black',
                    font=dict(family='Arial', size=12, color='white'),
                    width = 1000,
                    height = 400,
                    xaxis=dict(
                        showgrid=True, gridwidth=0.5, gridcolor='grey', color='black'
                    ),
                    yaxis=dict(
                        showgrid=True, gridwidth=0.5, gridcolor='grey', color='black'
                    )
                )
                
                st.plotly_chart(fig)

        else:
            st.markdown("""
                <style>
                .dataframe tbody tr th, .dataframe tbody tr td {
                    padding: 0.5em 0.5em;  /* Reduce padding to make the cells more compact */
                    font-size: 14px;
                }
                .dataframe thead tr th {
                    padding: 0.5em 0.5em;  /* Reduce padding in the header */
                    font-size: 14px;
                }
                </style>
                """, unsafe_allow_html=True)
            
            data = data.reset_index()
            Open_stat = data['Open']
            Open_stat = (pd.DataFrame(Open_stat.describe())).transpose()
            st.write('Statistics of opening price')
            st.dataframe(Open_stat)

            Aclose_stat = data['Adj Close']
            Aclose_stat = (pd.DataFrame(Aclose_stat.describe())).transpose()
            st.write('Statistics of adjusted closing price')
            st.dataframe(Aclose_stat)

            # Adjusted closing price with time
            st.markdown("<br>", unsafe_allow_html=True)
            st.write('Adjusted closing price with time')
            fig = px.line(
                data,
                x='Date',
                y='Adj Close'
            )
            fig.update_layout(
                xaxis_title='Date',
                yaxis_title='Adjusted closing price',
                plot_bgcolor='black',
                paper_bgcolor='black',
                font=dict(color='white', size=14),
                width=1000,
                height=400
            )
            fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='gray', color='white')
            fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='gray', color='white')

            st.plotly_chart(fig, use_container_width=False)

            # Moving averages
            MA_data = MA_data.reset_index()
            adj_MA_data = MA_data[['Date', 'Adj Close']].copy()

            st.markdown("<br>", unsafe_allow_html=True)
            st.write('Adj close price and 50 and 200 days moving average')

            adj_MA_data['50_MA'] = adj_MA_data['Adj Close'].rolling(window=50).mean()
            adj_MA_data['200_MA'] = adj_MA_data['Adj Close'].rolling(window=200).mean()
            adj_MA_data = adj_MA_data[adj_MA_data['Date'] >= pd.to_datetime(start_date)] 

            fig = go.Figure()

            fig.add_trace(go.Scatter(x=adj_MA_data['Date'], y=adj_MA_data['Adj Close'], mode='lines', name='Adj Close', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=adj_MA_data['Date'], y=adj_MA_data['50_MA'], mode='lines', name='50 days MA', line=dict(color='hotpink')))
            fig.add_trace(go.Scatter(x=adj_MA_data['Date'], y=adj_MA_data['200_MA'], mode='lines', name='200 days MA', line=dict(color='green')))

            fig.update_layout(
                title = f'{tickers_list[0]} stock data',
                xaxis_title = 'Date',
                yaxis_title = 'Price',
                legend=dict(title='Legend'),
                plot_bgcolor='black',
                paper_bgcolor='black',
                font=dict(family='Arial', size=12, color='white'),
                width = 1000,
                height = 400,
                xaxis=dict(
                    showgrid=True, gridwidth=0.5, gridcolor='grey', color='white'
                ),
                yaxis=dict(
                    showgrid=True, gridwidth=0.5, gridcolor='grey', color='white'
                )
            )
            
            st.plotly_chart(fig)
       