#!/usr/bin/env python3
"""
Streamlit Model Comparison Dashboard
A beautiful, interactive dashboard for comparing model training runs
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Model Training Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        color: #333333 !important;
    }
    .metric-card h4 {
        color: #1f77b4 !important;
        margin-bottom: 0.5rem;
        font-weight: bold;
    }
    .metric-card p {
        color: #333333 !important;
        margin: 0.25rem 0;
    }
    .metric-card strong {
        color: #000000 !important;
        font-size: 1.2em;
    }
    .success-card {
        border-left-color: #28a745;
    }
    .warning-card {
        border-left-color: #ffc107;
    }
    .error-card {
        border-left-color: #dc3545;
    }
    .stDataFrame {
        font-size: 0.9rem;
    }
    .refresh-info {
        background-color: #e8f4fd;
        padding: 0.5rem;
        border-radius: 0.3rem;
        border-left: 3px solid #1f77b4;
        margin-bottom: 1rem;
        color: #333333 !important;
    }
    .refresh-info small {
        color: #666666 !important;
    }
</style>
""", unsafe_allow_html=True)

# Auto-refresh functionality
def setup_auto_refresh():
    """Setup auto-refresh every minute"""
    # Add JavaScript for auto-refresh
    st.markdown("""
    <script>
        // Auto-refresh every 60 seconds
        setTimeout(function(){
            window.location.reload();
        }, 60000);
    </script>
    """, unsafe_allow_html=True)

@st.cache_data(ttl=60)  # Cache for 60 seconds
def load_model_log(log_file="model_training_log.csv"):
    """Load and preprocess the model training log"""
    if not os.path.exists(log_file):
        st.error(f"❌ Log file {log_file} not found!")
        return None
    
    df = pd.read_csv(log_file)
    
    # Convert date strings to datetime
    df['Date_Start'] = pd.to_datetime(df['Date_Start'])
    
    # Sort by date (most recent first)
    df = df.sort_values('Date_Start', ascending=False).reset_index(drop=True)
    
    # Add run number
    df['Run_Number'] = range(1, len(df) + 1)
    
    return df

def main():
    # Setup auto-refresh
    setup_auto_refresh()
    
    # Initialize session state for refresh tracking
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()
    
    # Calculate time until next refresh
    time_since_refresh = datetime.now() - st.session_state.last_refresh
    seconds_until_refresh = max(0, 60 - time_since_refresh.seconds)
    
    # Header with refresh controls
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        st.markdown('<h1 class="main-header">🤖 Model Training Dashboard</h1>', unsafe_allow_html=True)
    
    with col2:
        if st.button("🔄 Refresh Now", type="primary"):
            st.session_state.last_refresh = datetime.now()
            st.rerun()
    
    with col3:
        st.markdown(f"""
        <div class="refresh-info">
            <small>🕐 Last updated: {st.session_state.last_refresh.strftime('%H:%M:%S')}</small><br>
            <small>🔄 Auto-refresh: {seconds_until_refresh}s</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Load data
    df = load_model_log()
    if df is None:
        st.stop()
    
    # Check for recent runs (last 5 minutes)
    if not df.empty:
        recent_cutoff = datetime.now() - timedelta(minutes=5)
        recent_runs = df[df['Date_Start'] > recent_cutoff]
        
        if not recent_runs.empty:
            st.success(f"🎉 **New runs detected!** {len(recent_runs)} new training runs in the last 5 minutes.")
            for _, run in recent_runs.iterrows():
                status_icon = "✅" if run['Status'] == 'Success' else "❌"
                st.info(f"{status_icon} **{run['Model_Description']}** - {run['Date_Start'].strftime('%H:%M:%S')} - AUC: {run.get('Test_AUC', 'N/A')}")
    
    # Sidebar
    st.sidebar.title("📊 Dashboard Controls")
    
    # Filter options
    st.sidebar.subheader("🔍 Filters")
    
    # Status filter
    status_filter = st.sidebar.multiselect(
        "Status",
        options=df['Status'].unique(),
        default=df['Status'].unique()
    )
    
    # Model filter
    successful_runs = df[df['Status'] == 'Success']
    if not successful_runs.empty:
        model_filter = st.sidebar.multiselect(
            "Best Model",
            options=successful_runs['Best_Model'].unique(),
            default=successful_runs['Best_Model'].unique()
        )
    else:
        model_filter = []
    
    # Date range filter
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(df['Date_Start'].min().date(), df['Date_Start'].max().date()),
        min_value=df['Date_Start'].min().date(),
        max_value=df['Date_Start'].max().date()
    )
    
    # Apply filters
    filtered_df = df.copy()
    
    if status_filter:
        filtered_df = filtered_df[filtered_df['Status'].isin(status_filter)]
    
    if model_filter and not successful_runs.empty:
        filtered_df = filtered_df[filtered_df['Best_Model'].isin(model_filter)]
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df['Date_Start'].dt.date >= start_date) &
            (filtered_df['Date_Start'].dt.date <= end_date)
        ]
    
    # Main content
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Runs", len(filtered_df))
    
    with col2:
        successful_count = len(filtered_df[filtered_df['Status'] == 'Success'])
        st.metric("Successful Runs", successful_count)
    
    with col3:
        if successful_count > 0:
            avg_auc = filtered_df[filtered_df['Status'] == 'Success']['Test_AUC'].mean()
            st.metric("Avg Test AUC", f"{avg_auc:.4f}")
        else:
            st.metric("Avg Test AUC", "N/A")
    
    with col4:
        if successful_count > 0:
            best_auc = filtered_df[filtered_df['Status'] == 'Success']['Test_AUC'].max()
            st.metric("Best Test AUC", f"{best_auc:.4f}")
        else:
            st.metric("Best Test AUC", "N/A")
    
    # Main table
    st.subheader("📋 All Model Runs")
    
    # Prepare table data
    table_df = filtered_df.copy()
    
    # Format columns for display
    if not table_df.empty:
        # Format date
        table_df['Date'] = table_df['Date_Start'].dt.strftime('%Y-%m-%d %H:%M')
        
        # Format numeric columns
        numeric_columns = ['Test_AUC', 'Test_Accuracy', 'Test_Precision', 'Test_Recall', 'Test_F1', 'Best_Precision', 'Best_Expected_Value']
        for col in numeric_columns:
            if col in table_df.columns:
                table_df[col] = table_df[col].apply(lambda x: f"{x:.4f}" if pd.notna(x) and x != 0 else "N/A")
        
        # Format duration
        if 'Duration_Seconds' in table_df.columns:
            table_df['Duration'] = table_df['Duration_Seconds'].apply(lambda x: f"{x:.1f}s" if pd.notna(x) else "N/A")
        
        # Select and reorder columns for display
        display_columns = [
            'Date', 'Model_Description', 'Status',
            'Best_Model', 'Test_AUC', 'Test_Recall', 'Test_F1', 'Best_Precision', 'Best_Signals',
            'Best_Expected_Value', 'Num_Features', 'Num_Samples', 'Duration', 'Training_File'
        ]
        
        # Filter to only existing columns
        display_columns = [col for col in display_columns if col in table_df.columns]
        
        # Display the table
        st.dataframe(
            table_df[display_columns],
            use_container_width=True,
            hide_index=True
        )
    
    # Performance Analysis
    if successful_count > 0:
        st.subheader("📈 Performance Analysis")
        
        successful_df = filtered_df[filtered_df['Status'] == 'Success']
        
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance Trends", "🤖 Model Comparison", "🔍 Feature Analysis", "📈 Detailed Metrics"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # AUC over time
                fig_auc = px.line(
                    successful_df.sort_values('Date_Start'),
                    x='Date_Start',
                    y='Test_AUC',
                    title='Test AUC Over Time',
                    markers=True
                )
                fig_auc.update_layout(height=400)
                st.plotly_chart(fig_auc, use_container_width=True)
            
            with col2:
                # Duration over time
                fig_duration = px.line(
                    successful_df.sort_values('Date_Start'),
                    x='Date_Start',
                    y='Duration_Seconds',
                    title='Training Duration Over Time',
                    markers=True
                )
                fig_duration.update_layout(height=400)
                st.plotly_chart(fig_duration, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                # Model performance comparison
                model_stats = successful_df.groupby('Best_Model').agg({
                    'Test_AUC': ['count', 'mean', 'std', 'min', 'max'],
                    'Test_F1': ['mean', 'std'],
                    'Duration_Seconds': ['mean', 'std']
                }).round(4)
                
                # Flatten column names
                model_stats.columns = ['_'.join(col).strip() for col in model_stats.columns]
                model_stats = model_stats.reset_index()
                
                st.subheader("Model Performance Summary")
                st.dataframe(model_stats, use_container_width=True)
            
            with col2:
                # AUC by model
                fig_model_auc = px.box(
                    successful_df,
                    x='Best_Model',
                    y='Test_AUC',
                    title='Test AUC Distribution by Model'
                )
                fig_model_auc.update_layout(height=400)
                st.plotly_chart(fig_model_auc, use_container_width=True)
        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                # Feature count analysis
                fig_features = px.scatter(
                    successful_df,
                    x='Num_Features',
                    y='Test_AUC',
                    color='Best_Model',
                    title='AUC vs Number of Features',
                    hover_data=['Model_Description']
                )
                fig_features.update_layout(height=400)
                st.plotly_chart(fig_features, use_container_width=True)
            
            with col2:
                # Sample count analysis
                fig_samples = px.scatter(
                    successful_df,
                    x='Num_Samples',
                    y='Test_AUC',
                    color='Best_Model',
                    title='AUC vs Number of Samples',
                    hover_data=['Model_Description']
                )
                fig_samples.update_layout(height=400)
                st.plotly_chart(fig_samples, use_container_width=True)
        
        with tab4:
            # Detailed metrics table
            st.subheader("Detailed Performance Metrics")
            
            detailed_columns = [
                'Date', 'Model_Description', 'Best_Model',
                'Test_AUC', 'Test_Accuracy', 'Test_Precision', 'Test_Recall', 'Test_F1',
                'Best_Threshold', 'Best_Precision', 'Best_Signals', 'Best_Expected_Value',
                'Top_Feature_1', 'Top_Feature_2', 'Top_Feature_3', 'Model_Performance_Notes', 'Training_File'
            ]
            
            detailed_columns = [col for col in detailed_columns if col in successful_df.columns]
            
            st.dataframe(
                successful_df[detailed_columns],
                use_container_width=True,
                hide_index=True
            )
    
    # Error Analysis
    error_runs = filtered_df[filtered_df['Status'] == 'Error']
    if not error_runs.empty:
        st.subheader("❌ Error Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Failed Runs:**")
            for _, row in error_runs.iterrows():
                with st.expander(f"Run #{row['Run_Number']} - {row['Date_Start'].strftime('%Y-%m-%d %H:%M')}"):
                    st.write(f"**Description:** {row.get('Model_Description', 'No description')}")
                    st.write(f"**Training File:** {row['Training_File']}")
                    st.write(f"**Error:** {row.get('Error_Message', 'No error message')}")
        
        with col2:
            # Error frequency
            if 'Error_Message' in error_runs.columns:
                error_counts = error_runs['Error_Message'].value_counts()
                fig_errors = px.pie(
                    values=error_counts.values,
                    names=error_counts.index,
                    title='Error Types Distribution'
                )
                st.plotly_chart(fig_errors, use_container_width=True)
    
    # Best Performers
    if successful_count > 0:
        st.subheader("🏆 Best Performers")
        st.markdown("""
        <div style="background-color: #f0f8ff; padding: 10px; border-radius: 5px; margin-bottom: 20px;">
        <small>
        <strong>📊 Metric Definitions:</strong><br>
        • <strong>Best AUC:</strong> Highest Area Under ROC Curve (overall model performance)<br>
        • <strong>Best F1:</strong> Highest F1-Score (balance of precision and recall)<br>
        • <strong>Best Precision:</strong> Highest precision with 50-1000 signals (practical trading volume)<br>
        • <strong>Best Expected Value:</strong> Highest expected return per trade (precision - false positive rate)
        </small>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            best_auc_run = successful_df.loc[successful_df['Test_AUC'].idxmax()]
            st.markdown(f"""
            <div class="metric-card success-card">
                <h4>🥇 Best AUC</h4>
                <p><strong>{best_auc_run['Test_AUC']:.4f}</strong></p>
                <p>Run #{best_auc_run['Run_Number']}</p>
                <p>{best_auc_run['Model_Description']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            best_f1_run = successful_df.loc[successful_df['Test_F1'].idxmax()]
            st.markdown(f"""
            <div class="metric-card success-card">
                <h4>🎯 Best F1</h4>
                <p><strong>{best_f1_run['Test_F1']:.4f}</strong></p>
                <p>Run #{best_f1_run['Run_Number']}</p>
                <p>{best_f1_run['Model_Description']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            best_precision_run = successful_df.loc[successful_df['Best_Precision'].idxmax()]
            signals_info = f" ({best_precision_run.get('Best_Signals', 'N/A')} signals)" if 'Best_Signals' in best_precision_run else ""
            st.markdown(f"""
            <div class="metric-card success-card">
                <h4>🎯 Best Precision</h4>
                <p><strong>{best_precision_run['Best_Precision']:.1%}</strong>{signals_info}</p>
                <p>Run #{best_precision_run['Run_Number']}</p>
                <p>{best_precision_run['Model_Description']}</p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
