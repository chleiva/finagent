
# Market Configuration System
MARKET_CONFIGS = {
    'US': {
        'name': 'United States',
        'symbols':
         [
            "NVDA",   # Nvidia
            "MSFT",   # Microsoft
            "AAPL",   # Apple
            "AMZN",   # Amazon
            "GOOGL",  # Alphabet Inc. Class A
            "META",   # Meta Platforms
            "AVGO",   # Broadcom
            "TSLA",   # Tesla
            "NFLX",   # Netflix
            "COST"    # Costco
        ],
        'api_endpoint': '/trsrv/stocks',
        'api_method': 'GET',
        'api_params': lambda symbols: {'symbols': ','.join(symbols)},
        'response_parser': lambda data, symbol: data.get(symbol, [{}])[0].get('contracts', [{}])[0].get('conid') if data.get(symbol) else None,
        'websocket_format': 'smd+{conid}',
        'market_hours': {
            'start': '09:30',
            'end': '16:00',
            'timezone': 'US/Eastern'
        }
    },
    'UK': {
        'name': 'United Kingdom (LSE)',
        'symbols': [
            "AZN.L", "SHEL.L", "HSBA.L", "ULVR.L", "BP.L", "DGE.L", "GSK.L", "BATS.L",
            "RIO.L", "REL.L", "BARC.L", "LLOY.L", "LSEG.L", "PRU.L", "NG.L", "GLEN.L",
            "RKT.L", "CPG.L", "TSCO.L", "VOD.L"
        ],
        'api_endpoint': '/iserver/secdef/search',
        'api_method': 'POST',
        'api_params': lambda symbols: [{'symbol': symbol, 'name': True} for symbol in symbols],
        'response_parser': lambda data, symbol: next((contract.get('conid') for contract in data if isinstance(contract, dict) and contract.get('symbol') == symbol), None) if isinstance(data, list) else None,
        'websocket_format': 'smd+{conid}',
        'market_hours': {
            'start': '08:00',
            'end': '16:30',
            'timezone': 'Europe/London'
        }
    },
    'EU': {
        'name': 'European Union',
        'symbols': ['ASML.AS', 'SAP.DE', 'NESN.SW', 'NOVO-B.CO', 'ROCHE.SW'],
        'api_endpoint': '/iserver/secdef/search',
        'api_method': 'POST',
        'api_params': lambda symbols: [{'symbol': symbol, 'name': True} for symbol in symbols],
        'response_parser': lambda data, symbol: next((contract.get('conid') for contract in data if isinstance(contract, dict) and contract.get('symbol') == symbol), None) if isinstance(data, list) else None,
        'websocket_format': 'smd+{conid}',
        'market_hours': {
            'start': '09:00',
            'end': '17:30',
            'timezone': 'Europe/Paris'
        }
    },
    'CRYPTO': {
        'name': 'Cryptocurrency',
        'symbols': [
            "BTC", "ETH", "SOL", "ADA", "XRP"  # IBKR format: no -USD suffix
        ],
        'api_endpoint': '/iserver/secdef/search',
        'api_method': 'POST',
        'api_params': lambda symbols: [{'symbol': symbol, 'name': True, 'secType': 'CRYPTO'} for symbol in symbols],
        'response_parser': lambda data, symbol: next((contract.get('conid') for contract in data if isinstance(contract, dict) and contract.get('symbol') == symbol), None) if isinstance(data, list) else None,
        'websocket_format': 'smd+{conid}',
        'market_hours': {
            'start': '00:00',  # 24/7 trading
            'end': '23:59',
            'timezone': 'UTC'
        }
    }
}

# Default market (can be changed via command line or environment variable)
DEFAULT_MARKET = 'US'

# Symbol mapping for Yahoo Finance historical data
# Maps IBKR symbols to Yahoo Finance symbols for historical data fetching
YAHOO_FINANCE_SYMBOL_MAP = {
    # Crypto symbols: IBKR format -> Yahoo Finance format
    'BTC': 'BTC-USD',
    'ETH': 'ETH-USD', 
    'SOL': 'SOL-USD',
    'ADA': 'ADA-USD',
    'XRP': 'XRP-USD',
    'BNB': 'BNB-USD',
    'DOGE': 'DOGE-USD',
    'DOT': 'DOT-USD',
    'AVAX': 'AVAX-USD',
    'SHIB': 'SHIB-USD',
    'MATIC': 'MATIC-USD',
    'LTC': 'LTC-USD',
    'LINK': 'LINK-USD',
    'UNI': 'UNI-USD',
    'ATOM': 'ATOM-USD',
    'XLM': 'XLM-USD',
    'ALGO': 'ALGO-USD',
    'VET': 'VET-USD',
    # Add more mappings as needed
}

def get_yahoo_finance_symbol(symbol):
    """Convert IBKR symbol to Yahoo Finance symbol using the mapping."""
    return YAHOO_FINANCE_SYMBOL_MAP.get(symbol, symbol)