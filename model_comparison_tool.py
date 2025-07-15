#!/usr/bin/env python3
"""
Model Comparison Tool
Interactive utility to analyze and compare model training runs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import argparse

class ModelComparisonTool:
    def __init__(self, log_file="model_training_log.csv"):
        self.log_file = log_file
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Load the model training log data"""
        if not os.path.exists(self.log_file):
            print(f"❌ Log file {self.log_file} not found!")
            return False
        
        self.df = pd.read_csv(self.log_file)
        # Convert date strings to datetime
        self.df['Date_Start'] = pd.to_datetime(self.df['Date_Start'])
        # Sort by date (most recent first)
        self.df = self.df.sort_values('Date_Start', ascending=False).reset_index(drop=True)
        return True
    
    def show_recent_runs(self, n=10):
        """Show the most recent model training runs"""
        print(f"📊 RECENT MODEL TRAINING RUNS (Last {min(n, len(self.df))})")
        print("=" * 100)
        
        for idx, row in self.df.head(n).iterrows():
            print(f"\n🔹 RUN #{idx + 1} - {row['Date_Start'].strftime('%Y-%m-%d %H:%M:%S')}")
            print("-" * 80)
            print(f"   📝 Description: {row.get('Model_Description', 'No description')}")
            print(f"   📁 Training: {row['Training_File']}")
            print(f"   🧪 Test: {row['Test_File']}")
            print(f"   ⚡ Fast Mode: {row['Fast_Mode']}")
            print(f"   📊 Features: {row['Num_Features']} | Samples: {row['Num_Samples']:,}")
            print(f"   ⏱️ Duration: {row['Duration_Seconds']:.2f}s")
            print(f"   ✅ Status: {row['Status']}")
            
            if row['Status'] == 'Success':
                print(f"   🤖 Best Model: {row['Best_Model']}")
                print(f"   📊 Test AUC: {row['Test_AUC']:.4f}")
                print(f"   🎯 Best F1: {row['Test_F1']:.3f}")
                print(f"   📈 Best Precision: {row['Best_Precision']:.1%}")
                print(f"   📊 Signals: {row['Best_Signals']}")
                print(f"   💰 Expected Value: {row['Best_Expected_Value']:.3f}")
                print(f"   📝 Performance: {row['Model_Performance_Notes']}")
            else:
                print(f"   ❌ Error: {row['Error_Message']}")
    
    def compare_models(self, run_indices=None):
        """Compare specific model runs side by side"""
        if run_indices is None:
            # Compare top 3 successful runs
            successful_runs = self.df[self.df['Status'] == 'Success'].head(3)
        else:
            successful_runs = self.df.iloc[run_indices]
            successful_runs = successful_runs[successful_runs['Status'] == 'Success']
        
        if successful_runs.empty:
            print("❌ No successful runs to compare!")
            return
        
        print("📊 MODEL COMPARISON")
        print("=" * 120)
        
        # Create comparison table
        comparison_data = []
        for idx, row in successful_runs.iterrows():
            comparison_data.append({
                'Run #': idx + 1,
                'Date': row['Date_Start'].strftime('%m-%d %H:%M'),
                'Description': row.get('Model_Description', 'No description')[:30] + '...' if len(str(row.get('Model_Description', 'No description'))) > 30 else row.get('Model_Description', 'No description'),
                'Training File': row['Training_File'],
                'Test File': row['Test_File'],
                'Features': row['Num_Features'],
                'Samples': f"{row['Num_Samples']:,}",
                'Duration (s)': f"{row['Duration_Seconds']:.1f}",
                'Best Model': row['Best_Model'],
                'Test AUC': f"{row['Test_AUC']:.4f}",
                'Best F1': f"{row['Test_F1']:.3f}",
                'Best Precision': f"{row['Best_Precision']:.1%}",
                'Signals': row['Best_Signals'],
                'Expected Value': f"{row['Best_Expected_Value']:.3f}",
                'Performance': row['Model_Performance_Notes']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
    
    def performance_trends(self):
        """Show performance trends over time"""
        successful_runs = self.df[self.df['Status'] == 'Success'].copy()
        
        if len(successful_runs) < 2:
            print("❌ Need at least 2 successful runs to show trends!")
            return
        
        print("📈 PERFORMANCE TRENDS")
        print("=" * 80)
        
        # Calculate trends
        successful_runs = successful_runs.sort_values('Date_Start')
        
        # AUC trend
        auc_trend = np.polyfit(range(len(successful_runs)), successful_runs['Test_AUC'], 1)
        auc_slope = auc_trend[0]
        
        # Duration trend
        duration_trend = np.polyfit(range(len(successful_runs)), successful_runs['Duration_Seconds'], 1)
        duration_slope = duration_trend[0]
        
        # Feature count trend
        feature_trend = np.polyfit(range(len(successful_runs)), successful_runs['Num_Features'], 1)
        feature_slope = feature_trend[0]
        
        print(f"📊 AUC Trend: {'↗️ Improving' if auc_slope > 0.001 else '↘️ Declining' if auc_slope < -0.001 else '➡️ Stable'} ({auc_slope:.4f} per run)")
        print(f"⏱️ Duration Trend: {'↗️ Increasing' if duration_slope > 1 else '↘️ Decreasing' if duration_slope < -1 else '➡️ Stable'} ({duration_slope:.1f}s per run)")
        print(f"📊 Feature Count Trend: {'↗️ Increasing' if feature_slope > 0.5 else '↘️ Decreasing' if feature_slope < -0.5 else '➡️ Stable'} ({feature_slope:.1f} features per run)")
        
        # Best and worst runs
        best_run = successful_runs.loc[successful_runs['Test_AUC'].idxmax()]
        worst_run = successful_runs.loc[successful_runs['Test_AUC'].idxmin()]
        
        print(f"\n🏆 Best Run: {best_run['Date_Start'].strftime('%Y-%m-%d %H:%M')} (AUC: {best_run['Test_AUC']:.4f})")
        print(f"📁 Training: {best_run['Training_File']} | Test: {best_run['Test_File']}")
        print(f"🤖 Model: {best_run['Best_Model']} | Features: {best_run['Num_Features']}")
        
        print(f"\n📉 Worst Run: {worst_run['Date_Start'].strftime('%Y-%m-%d %H:%M')} (AUC: {worst_run['Test_AUC']:.4f})")
        print(f"📁 Training: {worst_run['Training_File']} | Test: {worst_run['Test_File']}")
        print(f"🤖 Model: {worst_run['Best_Model']} | Features: {worst_run['Num_Features']}")
    
    def feature_analysis(self):
        """Analyze feature usage patterns"""
        successful_runs = self.df[self.df['Status'] == 'Success']
        
        if successful_runs.empty:
            print("❌ No successful runs for feature analysis!")
            return
        
        print("🔍 FEATURE ANALYSIS")
        print("=" * 80)
        
        # Feature count distribution
        feature_counts = successful_runs['Num_Features'].value_counts().sort_index()
        print("📊 Feature Count Distribution:")
        for count, freq in feature_counts.items():
            print(f"   {count} features: {freq} runs")
        
        # Performance by feature count
        print(f"\n📈 Performance by Feature Count:")
        for count in feature_counts.index:
            runs_with_count = successful_runs[successful_runs['Num_Features'] == count]
            avg_auc = runs_with_count['Test_AUC'].mean()
            print(f"   {count} features: Avg AUC = {avg_auc:.4f} ({len(runs_with_count)} runs)")
        
        # Most common top features
        top_features = []
        for _, row in successful_runs.iterrows():
            for i in range(1, 4):
                feature = row[f'Top_Feature_{i}']
                if pd.notna(feature) and feature != 'None':
                    top_features.append(feature)
        
        if top_features:
            feature_freq = pd.Series(top_features).value_counts()
            print(f"\n🏆 Most Common Top Features:")
            for feature, freq in feature_freq.head(5).items():
                print(f"   {feature}: {freq} times")
    
    def model_performance_breakdown(self):
        """Break down performance by model type"""
        successful_runs = self.df[self.df['Status'] == 'Success']
        
        if successful_runs.empty:
            print("❌ No successful runs for model analysis!")
            return
        
        print("🤖 MODEL PERFORMANCE BREAKDOWN")
        print("=" * 80)
        
        model_stats = successful_runs.groupby('Best_Model').agg({
            'Test_AUC': ['count', 'mean', 'std', 'min', 'max'],
            'Test_F1': ['mean', 'std'],
            'Duration_Seconds': ['mean', 'std'],
            'Num_Features': ['mean', 'std']
        }).round(4)
        
        for model in model_stats.index:
            print(f"\n🔹 {model}:")
            stats = model_stats.loc[model]
            print(f"   📊 Runs: {stats[('Test_AUC', 'count')]}")
            print(f"   📈 AUC: {stats[('Test_AUC', 'mean')]:.4f} ± {stats[('Test_AUC', 'std')]:.4f}")
            print(f"   📊 AUC Range: {stats[('Test_AUC', 'min')]:.4f} - {stats[('Test_AUC', 'max')]:.4f}")
            print(f"   🎯 F1: {stats[('Test_F1', 'mean')]:.3f} ± {stats[('Test_F1', 'std')]:.3f}")
            print(f"   ⏱️ Duration: {stats[('Duration_Seconds', 'mean')]:.1f}s ± {stats[('Duration_Seconds', 'std')]:.1f}s")
            print(f"   📊 Features: {stats[('Num_Features', 'mean')]:.0f} ± {stats[('Num_Features', 'std')]:.0f}")
    
    def generate_plots(self, save_plots=True):
        """Generate performance visualization plots"""
        successful_runs = self.df[self.df['Status'] == 'Success']
        
        if len(successful_runs) < 2:
            print("❌ Need at least 2 successful runs to generate plots!")
            return
        
        # Set up the plotting style
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Training Performance Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: AUC over time
        successful_runs_sorted = successful_runs.sort_values('Date_Start')
        axes[0, 0].plot(successful_runs_sorted['Date_Start'], successful_runs_sorted['Test_AUC'], 
                       marker='o', linewidth=2, markersize=6)
        axes[0, 0].set_title('Test AUC Over Time', fontweight='bold')
        axes[0, 0].set_ylabel('Test AUC')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Duration vs AUC
        scatter = axes[0, 1].scatter(successful_runs['Duration_Seconds'], successful_runs['Test_AUC'], 
                                   c=successful_runs['Num_Features'], cmap='viridis', s=100, alpha=0.7)
        axes[0, 1].set_title('Duration vs AUC (colored by feature count)', fontweight='bold')
        axes[0, 1].set_xlabel('Duration (seconds)')
        axes[0, 1].set_ylabel('Test AUC')
        axes[0, 1].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0, 1], label='Number of Features')
        
        # Plot 3: Feature count vs AUC
        axes[1, 0].scatter(successful_runs['Num_Features'], successful_runs['Test_AUC'], 
                          c=successful_runs['Duration_Seconds'], cmap='plasma', s=100, alpha=0.7)
        axes[1, 0].set_title('Feature Count vs AUC (colored by duration)', fontweight='bold')
        axes[1, 0].set_xlabel('Number of Features')
        axes[1, 0].set_ylabel('Test AUC')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Model performance comparison
        model_auc = successful_runs.groupby('Best_Model')['Test_AUC'].agg(['mean', 'std', 'count'])
        models = model_auc.index
        means = model_auc['mean']
        stds = model_auc['std']
        counts = model_auc['count']
        
        bars = axes[1, 1].bar(models, means, yerr=stds, capsize=5, alpha=0.7)
        axes[1, 1].set_title('Model Performance Comparison', fontweight='bold')
        axes[1, 1].set_ylabel('Average Test AUC')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'n={count}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('model_performance_analysis.png', dpi=300, bbox_inches='tight')
            print("📊 Plots saved as 'model_performance_analysis.png'")
        
        plt.show()
    
    def interactive_menu(self):
        """Interactive menu for the tool"""
        while True:
            print("\n" + "="*80)
            print("🔧 MODEL COMPARISON TOOL")
            print("="*80)
            print("1. Show recent runs")
            print("2. Compare models")
            print("3. Performance trends")
            print("4. Feature analysis")
            print("5. Model performance breakdown")
            print("6. Generate plots")
            print("7. Reload data")
            print("8. Exit")
            print("-"*80)
            
            choice = input("Select an option (1-8): ").strip()
            
            if choice == '1':
                n = input("Number of recent runs to show (default 10): ").strip()
                n = int(n) if n.isdigit() else 10
                self.show_recent_runs(n)
            
            elif choice == '2':
                self.compare_models()
            
            elif choice == '3':
                self.performance_trends()
            
            elif choice == '4':
                self.feature_analysis()
            
            elif choice == '5':
                self.model_performance_breakdown()
            
            elif choice == '6':
                self.generate_plots()
            
            elif choice == '7':
                self.load_data()
                print("✅ Data reloaded!")
            
            elif choice == '8':
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid option. Please try again.")

def main():
    parser = argparse.ArgumentParser(description='Model Comparison Tool')
    parser.add_argument('--log-file', default='model_training_log.csv', 
                       help='Path to the model training log CSV file')
    parser.add_argument('--recent', type=int, default=5,
                       help='Show recent runs (default: 5)')
    parser.add_argument('--compare', action='store_true',
                       help='Show model comparison')
    parser.add_argument('--trends', action='store_true',
                       help='Show performance trends')
    parser.add_argument('--features', action='store_true',
                       help='Show feature analysis')
    parser.add_argument('--models', action='store_true',
                       help='Show model performance breakdown')
    parser.add_argument('--plots', action='store_true',
                       help='Generate performance plots')
    parser.add_argument('--interactive', action='store_true',
                       help='Run in interactive mode')
    
    args = parser.parse_args()
    
    tool = ModelComparisonTool(args.log_file)
    
    if not tool.df is not None:
        return
    
    if args.interactive:
        tool.interactive_menu()
    else:
        # Run all requested analyses
        if args.recent > 0:
            tool.show_recent_runs(args.recent)
        
        if args.compare:
            tool.compare_models()
        
        if args.trends:
            tool.performance_trends()
        
        if args.features:
            tool.feature_analysis()
        
        if args.models:
            tool.model_performance_breakdown()
        
        if args.plots:
            tool.generate_plots()
        
        # If no specific analysis requested, show recent runs
        if not any([args.recent, args.compare, args.trends, args.features, args.models, args.plots]):
            tool.show_recent_runs(5)

if __name__ == "__main__":
    main() 