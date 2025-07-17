import sqlite3

class DatabaseValidator:
    def __init__(self, db_path):
        self.db_path = db_path

    def connect(self):
        """Connect to the SQLite database"""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            print("✅ Connected to the database")
        except sqlite3.Error as e:
            print(f"❌ Error connecting to database: {e}")

    def check_table(self, table_name):
        """Check if the table contains data"""
        try:
            self.cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = self.cursor.fetchone()[0]
            if count > 0:
                print(f"✅ {table_name} contains {count} records")
            else:
                print(f"⚠️ {table_name} is empty")
        except sqlite3.Error as e:
            print(f"❌ Error checking {table_name}: {e}")

    def validate(self):
        """Run validation checks on the database"""
        self.connect()
        self.check_table('intraday_minute_data')
        self.check_table('daily_summary_data')

if __name__ == "__main__":
    validator = DatabaseValidator('database/realtime_market_data.db')
    validator.validate() 