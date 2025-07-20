import websocket, json, sqlite3, time, requests
from datetime import datetime

DB_PATH = 'database/realtime_market_data.db'
SYMBOLS = ["NVDA", "MSFT", "AAPL", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "NFLX", "COST"]
BASE_URL = "https://localhost:15000/v1/api"
WS_URL = "wss://localhost:15000/v1/api/ws"

session = requests.Session()
session.verify = False

# Get contract IDs
contracts = {sym: session.get(f"{BASE_URL}/trsrv/stocks", params={'symbols': sym}).json()[sym][0]['contracts'][0]['conid'] for sym in SYMBOLS}

# SQLite connection
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

def store_realtime(data):
    cursor.execute("""
        INSERT OR REPLACE INTO realtime_summary (symbol, last_price, bid_price, ask_price, bid_size, ask_size, volume,
        high_price, low_price, close_price, last_update_server_epoch, last_update_received_epoch)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, data)
    conn.commit()

def fetch_historical():
    for sym, conid in contracts.items():
        res = session.get(f"{BASE_URL}/iserver/marketdata/history", params={
            'conid': conid, 'period': '390min', 'bar': '1min', 'outsideRth': True}).json()
        for bar in res.get('data', []):
            bar_time = datetime.fromtimestamp(bar['t']/1000)
            cursor.execute("""
                INSERT OR IGNORE INTO intraday_minute_data (symbol, bar_time, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (sym, bar_time, bar['o'], bar['h'], bar['l'], bar['c'], bar['v']))
        conn.commit()

# WebSocket data handling
def on_message(ws, message):
    data = json.loads(message)
    if data.get('topic') == 'smd':
        conid = data.get('conid')
        sym = next((s for s, c in contracts.items() if c == conid), None)
        if sym:
            now = time.time()
            store_realtime((sym,
                            data.get('31'), data.get('84'), data.get('86'),
                            data.get('85'), data.get('88'), data.get('87'),
                            data.get('70'), data.get('71'), data.get('7059'),
                            data.get('_updated')/1000, now))

def run():
    ws = websocket.WebSocketApp(WS_URL, on_message=on_message)
    ws.run_forever(sslopt={"cert_reqs": 0})

# Start threads
import threading
threading.Thread(target=run, daemon=True).start()

while True:
    fetch_historical()
    time.sleep(30)
