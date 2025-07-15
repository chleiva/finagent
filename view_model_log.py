#!/usr/bin/env python3
"""
View Model Training Log
Displays the model training log in a readable format
"""

import pandas as pd
import os

def view_model_log(log_file="model_training_log.csv"):
    """Display the model training log in a readable format"""
    
    if not os.path.exists(log_file):
        print(f"❌ Log file {log_file} not found!")
        return
    
    # Load the log
    df = pd.read_csv(log_file)
    
    print(f"📊 MODEL TRAINING LOG ({len(df)} entries)")
    print("=" * 80)
    
    for idx, row in df.iterrows():
        print(f"\n🔹 RUN #{idx + 1} - {row['Date_Start']}")
        print("-" * 50)
        print(f"   📁 Training File: {row['Training_File']}")
        print(f"   🧪 Test File: {row['Test_File']}")
        print(f"   ⚡ Fast Mode: {row['Fast_Mode']}")
        print(f"   📊 Features: {row['Num_Features']}")
        print(f"   📈 Samples: {row['Num_Samples']:,}")
        print(f"   ⏱️ Duration: {row['Duration_Seconds']:.2f}s")
        print(f"   ✅ Status: {row['Status']}")
        
        if row['Status'] == 'Success':
            print(f"   🤖 Best Model: {row['Best_Model']}")
            print(f"   📊 Test AUC: {row['Test_AUC']:.4f}")
            print(f"   🎯 Best F1: {row['Test_F1']:.3f}")
            print(f"   📈 Best Precision: {row['Best_Precision']:.1%}")
            print(f"   📊 Best Signals: {row['Best_Signals']}")
            print(f"   💰 Best Expected Value: {row['Best_Expected_Value']:.3f}")
            print(f"   🏆 Top Features: {row['Top_Feature_1']}, {row['Top_Feature_2']}, {row['Top_Feature_3']}")
            print(f"   📝 Performance: {row['Model_Performance_Notes']}")
        else:
            print(f"   ❌ Error: {row['Error_Message']}")
    
    # Summary statistics
    print(f"\n📈 SUMMARY STATISTICS")
    print("=" * 50)
    successful_runs = df[df['Status'] == 'Success']
    if not successful_runs.empty:
        print(f"   ✅ Successful runs: {len(successful_runs)}/{len(df)}")
        print(f"   📊 Average AUC: {successful_runs['Test_AUC'].mean():.4f}")
        print(f"   📈 Best AUC: {successful_runs['Test_AUC'].max():.4f}")
        print(f"   ⏱️ Average duration: {successful_runs['Duration_Seconds'].mean():.2f}s")
        print(f"   📊 Average features: {successful_runs['Num_Features'].mean():.0f}")
        print(f"   📈 Average samples: {successful_runs['Num_Samples'].mean():,.0f}")
        
        # Best performing model
        best_run = successful_runs.loc[successful_runs['Test_AUC'].idxmax()]
        print(f"\n🏆 BEST PERFORMING RUN:")
        print(f"   📅 Date: {best_run['Date_Start']}")
        print(f"   🤖 Model: {best_run['Best_Model']}")
        print(f"   📊 AUC: {best_run['Test_AUC']:.4f}")
        print(f"   📁 Training: {best_run['Training_File']}")
        print(f"   🧪 Test: {best_run['Test_File']}")
    else:
        print("   ❌ No successful runs found")

if __name__ == "__main__":
    view_model_log() 