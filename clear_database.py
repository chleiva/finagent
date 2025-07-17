#!/usr/bin/env python3
"""
Clear Database Data
Remove all data from the realtime market data database
"""

import sqlite3
import os

DB_PATH = 'database/realtime_market_data.db'

def clear_all_data():
    """Clear all data from database tables"""
    if not os.path.exists(DB_PATH):
        print(f"❌ Database not found: {DB_PATH}")
        return False
    
    try:
        print(f"🗑️ Clearing all data from {DB_PATH}")
        print("=" * 60)
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
        tables = [row[0] for row in cursor.fetchall()]
        
        print(f"📋 Found {len(tables)} tables: {', '.join(tables)}")
        
        # Clear each table and show count
        total_deleted = 0
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count_before = cursor.fetchone()[0]
            
            cursor.execute(f"DELETE FROM {table}")
            deleted = cursor.rowcount
            total_deleted += deleted
            
            print(f"   🗑️ {table}: deleted {deleted:,} records")
        
        # Reset auto-increment sequences
        cursor.execute("DELETE FROM sqlite_sequence")
        
        conn.commit()
        conn.close()
        
        print("=" * 60)
        print(f"✅ Database cleared successfully!")
        print(f"📊 Total records deleted: {total_deleted:,}")
        print(f"🔄 Auto-increment sequences reset")
        print(f"💾 Database is now clean and ready for fresh data")
        
        return True
        
    except Exception as e:
        print(f"❌ Error clearing database: {e}")
        return False

def main():
    print("🗑️ Database Clear Tool")
    print("=" * 40)
    print("⚠️ WARNING: This will DELETE ALL data!")
    print("=" * 40)
    
    # Confirmation
    response = input("Are you sure you want to clear all database data? (yes/no): ")
    if response.lower() != 'yes':
        print("❌ Operation cancelled")
        return
    
    # Clear data
    if clear_all_data():
        print("\n✅ Database cleared successfully!")
        print("🔄 You can now restart the data collector for fresh data")
    else:
        print("\n❌ Failed to clear database")

if __name__ == "__main__":
    main() 