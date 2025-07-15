#!/usr/bin/env python3
"""
Model Artifact Manager
Utility script to find, list, and manage saved model artifacts
"""

import os
import pandas as pd
import argparse
from pathlib import Path

def find_model_by_description(description_keywords):
    """
    Find model artifacts by description keywords
    """
    index_file = "model_artifacts/model_index.csv"
    
    if not os.path.exists(index_file):
        print("❌ No model index found. No models have been saved yet.")
        return []
    
    index_df = pd.read_csv(index_file)
    
    # Search by keywords
    matching_models = []
    for _, row in index_df.iterrows():
        if any(keyword.lower() in row['description'].lower() for keyword in description_keywords):
            matching_models.append(row)
    
    return matching_models

def list_all_models():
    """
    List all saved models with their details
    """
    index_file = "model_artifacts/model_index.csv"
    
    if not os.path.exists(index_file):
        print("❌ No model index found. No models have been saved yet.")
        return
    
    index_df = pd.read_csv(index_file)
    
    if index_df.empty:
        print("❌ No models found in index.")
        return
    
    print(f"\n📋 Found {len(index_df)} saved models:")
    print("=" * 80)
    
    for i, (_, row) in enumerate(index_df.iterrows(), 1):
        print(f"\n{i}. 🆔 {row['model_id']}")
        print(f"   📝 Description: {row['description']}")
        print(f"   📅 Date: {row['date_created']}")
        print(f"   🤖 Best Model: {row['best_model']}")
        print(f"   📊 Test AUC: {row['test_auc']:.4f}")
        print(f"   🎯 Test F1: {row['test_f1']:.4f}")
        print(f"   📈 Best Precision: {row['best_precision']:.1%}")
        print(f"   📁 Path: {row['artifact_path']}")
        print("-" * 80)

def show_model_details(model_id):
    """
    Show detailed information about a specific model
    """
    index_file = "model_artifacts/model_index.csv"
    
    if not os.path.exists(index_file):
        print("❌ No model index found.")
        return
    
    index_df = pd.read_csv(index_file)
    model_row = index_df[index_df['model_id'] == model_id]
    
    if model_row.empty:
        print(f"❌ Model with ID '{model_id}' not found.")
        return
    
    model = model_row.iloc[0]
    model_path = model['artifact_path']
    
    print(f"\n🔍 Model Details: {model_id}")
    print("=" * 60)
    print(f"📝 Description: {model['description']}")
    print(f"📅 Date Created: {model['date_created']}")
    print(f"🤖 Best Model: {model['best_model']}")
    print(f"📊 Test AUC: {model['test_auc']:.4f}")
    print(f"🎯 Test F1: {model['test_f1']:.4f}")
    print(f"📈 Best Precision: {model['best_precision']:.1%}")
    print(f"📁 Artifact Path: {model_path}")
    
    # Show artifact structure
    if os.path.exists(model_path):
        print(f"\n📁 Artifact Structure:")
        print(f"   📂 {model_path}/")
        for subdir in ['models', 'evaluation', 'data', 'config', 'plots']:
            subdir_path = os.path.join(model_path, subdir)
            if os.path.exists(subdir_path):
                files = os.listdir(subdir_path)
                print(f"   ├── {subdir}/ ({len(files)} files)")
                for file in files[:3]:  # Show first 3 files
                    print(f"   │   ├── {file}")
                if len(files) > 3:
                    print(f"   │   └── ... and {len(files) - 3} more")
    else:
        print(f"❌ Artifact directory not found: {model_path}")

def clean_old_models(days_old=30):
    """
    Clean old model artifacts (older than specified days)
    """
    import shutil
    from datetime import datetime, timedelta
    
    index_file = "model_artifacts/model_index.csv"
    
    if not os.path.exists(index_file):
        print("❌ No model index found.")
        return
    
    index_df = pd.read_csv(index_file)
    cutoff_date = datetime.now() - timedelta(days=days_old)
    
    old_models = []
    for _, row in index_df.iterrows():
        try:
            model_date = datetime.strptime(row['date_created'], '%Y-%m-%d %H:%M:%S')
            if model_date < cutoff_date:
                old_models.append(row)
        except:
            continue
    
    if not old_models:
        print(f"✅ No models older than {days_old} days found.")
        return
    
    print(f"🗑️ Found {len(old_models)} models older than {days_old} days:")
    for model in old_models:
        print(f"   - {model['model_id']} ({model['description']})")
    
    confirm = input(f"\n❓ Delete these {len(old_models)} models? (y/N): ")
    if confirm.lower() == 'y':
        deleted_count = 0
        for model in old_models:
            try:
                if os.path.exists(model['artifact_path']):
                    shutil.rmtree(model['artifact_path'])
                deleted_count += 1
                print(f"   ✅ Deleted: {model['model_id']}")
            except Exception as e:
                print(f"   ❌ Failed to delete {model['model_id']}: {e}")
        
        # Update index
        index_df = index_df[~index_df['model_id'].isin([m['model_id'] for m in old_models])]
        index_df.to_csv(index_file, index=False)
        
        print(f"\n✅ Successfully deleted {deleted_count} old models.")

def main():
    parser = argparse.ArgumentParser(description='Model Artifact Manager')
    parser.add_argument('action', choices=['list', 'find', 'show', 'clean'], 
                       help='Action to perform')
    parser.add_argument('--keywords', nargs='+', 
                       help='Keywords to search for (for find action)')
    parser.add_argument('--model-id', 
                       help='Model ID to show details for (for show action)')
    parser.add_argument('--days', type=int, default=30,
                       help='Days old for cleaning (for clean action)')
    
    args = parser.parse_args()
    
    if args.action == 'list':
        list_all_models()
    
    elif args.action == 'find':
        if not args.keywords:
            print("❌ Please provide keywords to search for.")
            return
        matching_models = find_model_by_description(args.keywords)
        if matching_models:
            print(f"\n🔍 Found {len(matching_models)} matching models:")
            for model in matching_models:
                print(f"   🆔 {model['model_id']}")
                print(f"   📝 {model['description']}")
                print(f"   📊 AUC: {model['test_auc']:.4f}")
                print(f"   📁 {model['artifact_path']}")
                print()
        else:
            print("❌ No models found matching your keywords.")
    
    elif args.action == 'show':
        if not args.model_id:
            print("❌ Please provide a model ID to show details for.")
            return
        show_model_details(args.model_id)
    
    elif args.action == 'clean':
        clean_old_models(args.days)

if __name__ == "__main__":
    main()
