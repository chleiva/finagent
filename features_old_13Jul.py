import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List, Union
import warnings
import logging
from functools import lru_cache

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    pass

class FeatureCalculator:
    """
    Calculate trading features from real-time, intraday, and daily data sources.
    Robust implementation with comprehensive error handling and validation.
    """
    
    # Required columns for each data source
    RT_REQUIRED_COLS = ['bidPrice', 'bidSize', 'askPrice', 'askSize', 'lastPrice', 'timestamp']
    INTRA_REQUIRED_COLS = ['open_1min', 'high_1min', 'low_1min', 'close_1min', 'volume_1min']
    DAILY_REQUIRED_COLS = ['open', 'high', 'low', 'close', 'volume', 'date']
    
    def __init__(self, real_time_df: pd.DataFrame, intra_day_df: pd.DataFrame, daily_df: pd.DataFrame):
        """
        Initialize with three data sources and validate them.
        
        Args:
            real_time_df: Current real-time market data
            intra_day_df: Minute-level data for current day
            daily_df: Daily OHLCV data
            
        Raises:
            DataValidationError: If data validation fails
        """
        self.rt = real_time_df.copy() if real_time_df is not None else pd.DataFrame()
        self.intra = intra_day_df.copy() if intra_day_df is not None else pd.DataFrame()
        self.daily = daily_df.copy() if daily_df is not None else pd.DataFrame()
        
        # Validate and prepare data
        self._validate_data()
        self._prepare_data()
        
        # Cache for expensive calculations
        self._cache = {}
        
    def _validate_data(self):
        """Validate that required data and columns are present."""
        if self.rt.empty:
            raise DataValidationError("Real-time data cannot be empty")
            
        # Check required columns
        self._check_required_columns(self.rt, self.RT_REQUIRED_COLS, "real-time")
        
        if not self.intra.empty:
            self._check_required_columns(self.intra, self.INTRA_REQUIRED_COLS, "intraday")
            
        if not self.daily.empty:
            self._check_required_columns(self.daily, self.DAILY_REQUIRED_COLS, "daily")
    
    def _check_required_columns(self, df: pd.DataFrame, required_cols: List[str], data_type: str):
        """Check if DataFrame has required columns."""
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise DataValidationError(f"Missing {data_type} columns: {missing_cols}")
    
    def _prepare_data(self):
        """Prepare and clean data for feature calculation."""
        try:
            # Prepare real-time data
            if 'timestamp' in self.rt.columns:
                self.rt['timestamp'] = pd.to_datetime(self.rt['timestamp'])
                
            # Validate real-time data values
            numeric_cols = ['bidPrice', 'bidSize', 'askPrice', 'askSize', 'lastPrice']
            for col in numeric_cols:
                if col in self.rt.columns:
                    self.rt[col] = pd.to_numeric(self.rt[col], errors='coerce')
                    
            # Prepare intraday data
            if not self.intra.empty:
                if 'ts_event_clean' in self.intra.columns:
                    self.intra['ts_event_clean'] = pd.to_datetime(self.intra['ts_event_clean'])
                    self.intra = self.intra.sort_values('ts_event_clean')
                
                # Ensure numeric columns
                numeric_cols = ['open_1min', 'high_1min', 'low_1min', 'close_1min', 'volume_1min']
                for col in numeric_cols:
                    if col in self.intra.columns:
                        self.intra[col] = pd.to_numeric(self.intra[col], errors='coerce')
                        
                # Remove invalid data
                self.intra = self.intra.dropna(subset=numeric_cols)
                
            # Prepare daily data
            if not self.daily.empty:
                if 'date' in self.daily.columns:
                    self.daily['date'] = pd.to_datetime(self.daily['date'])
                    self.daily = self.daily.sort_values('date')
                    
                # Ensure numeric columns
                numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_cols:
                    if col in self.daily.columns:
                        self.daily[col] = pd.to_numeric(self.daily[col], errors='coerce')
                        
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise DataValidationError(f"Data preparation failed: {e}")
    
    @staticmethod
    def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
        """Safe division with default value for edge cases."""
        try:
            if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
                return default
            result = numerator / denominator
            return default if pd.isna(result) or np.isinf(result) else float(result)
        except:
            return default
    
    @staticmethod
    def safe_percentage(current: float, previous: float, default: float = 0.0) -> float:
        """Calculate percentage change safely."""
        return FeatureCalculator.safe_divide(current - previous, previous, default) * 100
    
    def get_current_price(self) -> float:
        """Get current price with validation."""
        try:
            price = float(self.rt['lastPrice'].iloc[0])
            return price if price > 0 else 0.0
        except:
            return 0.0
    
    def calculate_all_features(self) -> Dict[str, float]:
        """Calculate all features and return as dictionary."""
        features = {}
        
        try:
            # Microstructure features
            features.update(self._calculate_microstructure_features())
            
            # Volume features
            features.update(self._calculate_volume_features())
            
            # Technical indicators
            features.update(self._calculate_technical_indicators())
            
            # Return features
            features.update(self._calculate_return_features())
            
            # Volatility features
            features.update(self._calculate_volatility_features())
            
            # Price pattern features
            features.update(self._calculate_pattern_features())
            
            # Time features
            features.update(self._calculate_time_features())

            # Risk Management features
            features.update(self._calculate_risk_management_features())
            
        except Exception as e:
            logger.error(f"Error calculating features: {e}")
            # Return default values for all features
            return self._get_default_features()
        
        return features
    
    def _get_default_features(self) -> Dict[str, float]:
        """Return default feature values with normalized feature names."""
        return {
            'Bid_Ask_Spread_Pct': 0.0,              # Changed from Bid_Ask_Spread
            'Order_Book_Imbalance': 0.0,
            'Price_Position_In_Spread': 0.5,
            'Mid_Price_Momentum': 0.0,
            'Spread_Percentile_20': 0.5,
            'Volume_Weighted_Order_Book_Imbalance': 0.0,
            'Trade_Count_Per_Minute_Log': 0.0,      # Changed from Trade_Count_Per_Minute
            'Volume_Rate_of_Change': 0.0,
            'Price_Impact_Normalized': 0.0,          # Changed from Price_Impact
            'Volume_vs_SMA10': 1.0,
            'Normalized_Price_VWAP': 1.0,           # Removed absolute VWAP
            'MACD_12_26_9': 0.0,
            'RSI_14': 50.0,
            'Bollinger_Band_Width': 0.0,           # Removed absolute bands
            'Price_vs_SMA50': 0.0,
            'Return_5min': 0.0,
            'Return_Since_Open': 0.0,
            'ATR_Pct': 0.0,                        # Changed from ATR_14
            'Volatility_Regime_Shift_Norm': 0.0,   # Changed name
            'Volatility_Percentile_100': 50.0,
            'ATR_Stop_Risk_Pct': 0.0,
            'Position_In_Day_Range': 0.5,
            'Support_Resistance_Proximity': 5.0,
            'Breakout_Confirmation': 0.0,
            'Price_Acceleration': 0.0,
            'Max_Drawdown_30': 0.0,
            'Time_Of_Day_Cyclical_Sin': 0.0,
            'Time_Of_Day_Cyclical_Cos': 1.0,
            'Current_Price': 0.0,
            '1R_Stop_Loss': 0.0,
            'Risk_Amount_Pct': 0.0,
            'ATR_Distance': 0.0,
            '1R_Target': 0.0,
            '1_5R_Target': 0.0,
            '2R_Target': 0.0,
            '3R_Target': 0.0,
            '4R_Target': 0.0,
            'Position_Size_1pct_Risk': 0.0
        }

    # Microstructure Features
    def _calculate_microstructure_features(self) -> Dict[str, float]:
        """Calculate microstructure features with normalized names."""
        features = {}
        
        features['Bid_Ask_Spread_Pct'] = self.bid_ask_spread()  # Now returns percentage
        features['Order_Book_Imbalance'] = self.order_book_imbalance()
        features['Price_Position_In_Spread'] = self.price_position_in_spread()
        features['Mid_Price_Momentum'] = self.mid_price_momentum()
        features['Spread_Percentile_20'] = self.spread_percentile_20()
        features['Volume_Weighted_Order_Book_Imbalance'] = self.volume_weighted_order_book_imbalance()
        features['Trade_Count_Per_Minute_Log'] = self.trade_count_per_minute()  # Now log-scaled
        features['Volume_Rate_of_Change'] = self.volume_rate_of_change()
        features['Price_Impact_Normalized'] = self.price_impact()  # Now per dollar volume
        
        return features
    
    def bid_ask_spread(self) -> float:
        """Calculate current bid-ask spread as percentage of mid-price."""
        try:
            ask_price = float(self.rt['askPrice'].iloc[0])
            bid_price = float(self.rt['bidPrice'].iloc[0])
            
            if ask_price <= 0 or bid_price <= 0 or ask_price <= bid_price:
                return 0.0
            
            # Convert to percentage of mid-price for normalization
            mid_price = (ask_price + bid_price) / 2
            spread = ask_price - bid_price
            
            return (spread / mid_price) * 100  # Return as percentage
        except:
            return 0.0
    
    def order_book_imbalance(self) -> float:
        """Calculate order book imbalance at NBBO."""
        try:
            bid_vol = float(self.rt['bidSize'].iloc[0])
            ask_vol = float(self.rt['askSize'].iloc[0])
            
            if bid_vol < 0 or ask_vol < 0:
                return 0.0
                
            return self.safe_divide(bid_vol - ask_vol, bid_vol + ask_vol, 0.0)
        except:
            return 0.0
    
    def price_position_in_spread(self) -> float:
        """Calculate where last trade occurred within the spread."""
        try:
            last_price = float(self.rt['lastPrice'].iloc[0])
            bid_price = float(self.rt['bidPrice'].iloc[0])
            ask_price = float(self.rt['askPrice'].iloc[0])
            
            # Validate prices
            if ask_price <= bid_price or last_price <= 0:
                return 0.5
            
            # FIXED: Handle outside-spread trades correctly
            if last_price < bid_price:
                return 0.0  # Trading below bid
            elif last_price > ask_price:
                return 1.0  # Trading above ask (aggressive buying)
            else:
                # Normal case: within spread
                position = self.safe_divide(last_price - bid_price, ask_price - bid_price, 0.5)
                return max(0.0, min(1.0, position))
        except:
            return 0.5
    
    def mid_price_momentum(self) -> float:
        """Calculate rate of change in bid-ask midpoint."""
        if self.intra.empty or len(self.intra) < 5:  # Need at least 5 periods for stable momentum
            return 0.0
            
        try:
            # Use available bid/ask data from intraday or calculate from OHLC
            if 'bid_px_00' in self.intra.columns and 'ask_px_00' in self.intra.columns:
                recent_data = self.intra[['bid_px_00', 'ask_px_00']].tail(5).dropna()
                if len(recent_data) < 5:
                    return 0.0
                recent_data['mid_price'] = (recent_data['bid_px_00'] + recent_data['ask_px_00']) / 2
            else:
                # Fallback to using high+low as proxy for mid price
                recent_data = self.intra[['high_1min', 'low_1min']].tail(5).dropna()
                if len(recent_data) < 5:
                    return 0.0
                recent_data['mid_price'] = (recent_data['high_1min'] + recent_data['low_1min']) / 2
            
            # Calculate momentum over 5-period window for stability
            start_price = recent_data['mid_price'].iloc[0]
            end_price = recent_data['mid_price'].iloc[-1]
            
            momentum = self.safe_percentage(end_price, start_price)
            
            # For 5-minute window, cap momentum at reasonable range for liquid stocks
            return float(np.clip(momentum, -2.0, 2.0))  # ±2% max in 5 minutes
        except:
            return 0.0
    
    def spread_percentile_20(self) -> float:
        """Calculate current spread percentile vs 20-period rolling distribution (robust version)."""
        if self.intra.empty or len(self.intra) < 20:
            return 0.5
            
        try:
            # Calculate spreads from intraday data
            if 'bid_px_00' in self.intra.columns and 'ask_px_00' in self.intra.columns:
                recent_data = self.intra[['bid_px_00', 'ask_px_00']].tail(20).dropna()
                if len(recent_data) < 5:
                    return 0.5
                
                # Calculate spreads as percentages for consistency
                mid_prices = (recent_data['bid_px_00'] + recent_data['ask_px_00']) / 2
                spreads = recent_data['ask_px_00'] - recent_data['bid_px_00']
                recent_data['spread_pct'] = (spreads / mid_prices * 100).fillna(0)
            else:
                # Fallback to high-low as proxy for spread percentage
                recent_data = self.intra[['high_1min', 'low_1min', 'close_1min']].tail(20).dropna()
                if len(recent_data) < 5:
                    return 0.5
                
                spreads = recent_data['high_1min'] - recent_data['low_1min']
                recent_data['spread_pct'] = (spreads / recent_data['close_1min'] * 100).fillna(0)
            
            current_spread = self.bid_ask_spread()  # This now returns percentage
            
            # Handle edge cases
            if current_spread <= 0:
                return 0.5
            
            # Filter out zero spreads for percentile calculation
            non_zero_spreads = recent_data['spread_pct'][recent_data['spread_pct'] > 0]
            
            if len(non_zero_spreads) == 0:
                return 0.5  # All spreads are zero
            
            # Calculate percentile against non-zero spreads
            percentile = (non_zero_spreads <= current_spread).sum() / len(non_zero_spreads)
            
            return float(np.clip(percentile, 0.0, 1.0))
            
        except:
            return 0.5
    
    def volume_weighted_order_book_imbalance(self) -> float:
        """Calculate volume-weighted order book imbalance."""
        try:
            bid_price = float(self.rt['bidPrice'].iloc[0])
            ask_price = float(self.rt['askPrice'].iloc[0])
            bid_size = float(self.rt['bidSize'].iloc[0])
            ask_size = float(self.rt['askSize'].iloc[0])
            
            if bid_price <= 0 or ask_price <= 0 or bid_size < 0 or ask_size < 0:
                return 0.0
                
            bid_value = bid_price * bid_size
            ask_value = ask_price * ask_size
            
            return self.safe_divide(bid_value - ask_value, bid_value + ask_value, 0.0)
        except:
            return 0.0
    
    def trade_count_per_minute(self) -> float:
        """Calculate normalized trade count per minute."""
        if self.intra.empty:
            return 0.0
            
        try:
            # If we have trade count data, use it
            if 'trade_count' in self.intra.columns:
                count = float(self.intra['trade_count'].iloc[-1])
                return np.clip(count, 1.0, 100.0)  # Normalize to reasonable range
            
            # Otherwise estimate from volume using adaptive trade size
            current_volume = float(self.intra['volume_1min'].iloc[-1])
            
            if current_volume <= 0:
                return 0.0
            
            # ADAPTIVE TRADE SIZE: Based on volume level and stock price
            current_price = self.get_current_price()
            
            # Higher priced stocks typically have smaller share sizes
            if current_price > 200:  # High price stock like AAPL
                base_trade_size = 200
            elif current_price > 50:   # Medium price stock
                base_trade_size = 300
            else:  # Lower price stock
                base_trade_size = 500
            
            # Adjust for volume level
            if current_volume < 500:
                avg_trade_size = base_trade_size * 0.5  # Smaller trades in low volume
            elif current_volume > 5000:
                avg_trade_size = base_trade_size * 1.5  # Larger trades in high volume
            else:
                avg_trade_size = base_trade_size
            
            estimated_trades = current_volume / avg_trade_size
            
            # NORMALIZE: Return log-scaled trade count for better ML distribution
            normalized_count = np.log1p(estimated_trades)  # log(1 + trades)
            return float(np.clip(normalized_count, 0.0, 5.0))  # Cap log scale at 5
            
        except:
            return 0.0
    
    def volume_rate_of_change(self) -> float:
        """Calculate volume rate of change with robust bounds."""
        if self.intra.empty or len(self.intra) < 2:
            return 0.0
            
        try:
            volumes = self.intra['volume_1min'].tail(2)
            if len(volumes) < 2:
                return 0.0
            
            current_vol = float(volumes.iloc[-1])
            previous_vol = float(volumes.iloc[-2])
            
            if previous_vol <= 0:
                return 0.0 if current_vol <= 0 else 500.0  # Cap at 500% instead of 1000%
            
            # Calculate the actual percentage change
            change = (current_vol - previous_vol) / previous_vol * 100
            
            # ROBUST CLIPPING: More conservative bounds
            return float(np.clip(change, -95.0, 500.0))  # Reduced upper bound
        except:
            return 0.0
    
    def price_impact(self) -> float:
        """Calculate normalized price impact (basis points per $1000 volume)."""
        if self.intra.empty or len(self.intra) < 3:
            return 0.0
            
        try:
            recent = self.intra[['close_1min', 'volume_1min']].tail(5).dropna()
            if len(recent) < 3:
                return 0.0
            
            # Calculate price changes and volumes
            price_changes = recent['close_1min'].diff().abs()
            volumes = recent['volume_1min']
            current_price = self.get_current_price()
            
            if current_price <= 0:
                return 0.0
            
            # Filter valid data with volume threshold
            min_volume = 100
            valid_mask = (volumes >= min_volume) & (~price_changes.isna()) & (price_changes > 0)
            
            if valid_mask.sum() < 2:
                return 0.0
            
            valid_price_changes = price_changes[valid_mask]
            valid_volumes = volumes[valid_mask]
            
            # IMPROVED NORMALIZATION: Impact per $1000 of dollar volume (not share volume)
            dollar_volumes = valid_volumes * current_price  # Convert to dollar volume
            impacts = (valid_price_changes / dollar_volumes * 1000) / current_price * 10000
            
            if len(impacts) == 0:
                return 0.0
            
            # Use median and apply more conservative bounds
            impact = float(np.median(impacts))
            return np.clip(impact, 0.0, 5.0)  # More realistic cap for liquid stocks
            
        except:
            return 0.0
    
    # Volume Features
    def _calculate_volume_features(self) -> Dict[str, float]:
        """Calculate volume features (excluding absolute VWAP)."""
        features = {}
        
        features['Volume_vs_SMA10'] = self.volume_vs_sma10()
        # Remove absolute VWAP, keep only normalized version
        features['Normalized_Price_VWAP'] = self.normalized_price_vwap()
        
        return features
    
    def volume_vs_sma10(self) -> float:
        """Calculate ratio of current volume to its 10-period moving average."""
        if self.intra.empty or len(self.intra) < 2:
            return 1.0
            
        try:
            # FIXED: Use proper window for SMA calculation
            n_periods = min(10, len(self.intra))
            volumes = self.intra['volume_1min'].tail(n_periods).dropna()
            
            if len(volumes) == 0:
                return 1.0
                
            current_volume = float(volumes.iloc[-1])
            
            # FIXED: Calculate SMA excluding current volume for proper comparison
            if len(volumes) > 1:
                # Use previous periods for SMA (excluding current)
                historical_volumes = volumes.iloc[:-1]
                avg_volume = historical_volumes.mean()
            else:
                avg_volume = current_volume  # Only one data point
            
            if avg_volume <= 0:
                return 1.0
                
            ratio = current_volume / avg_volume
            return float(np.clip(ratio, 0.0, 50.0))
        except:
            return 1.0
    
    @lru_cache(maxsize=1)
    def vwap(self) -> float:
        """Calculate Volume Weighted Average Price for the day."""
        if self.intra.empty:
            return self.get_current_price()
            
        try:
            # Calculate typical price for each bar
            high = self.intra['high_1min']
            low = self.intra['low_1min'] 
            close = self.intra['close_1min']
            volume = self.intra['volume_1min']
            
            # Remove invalid data
            valid_mask = (volume > 0) & (~high.isna()) & (~low.isna()) & (~close.isna())
            if valid_mask.sum() == 0:
                return self.get_current_price()
            
            valid_data = self.intra[valid_mask].copy()
            
            # Calculate typical price
            typical_price = (valid_data['high_1min'] + valid_data['low_1min'] + valid_data['close_1min']) / 3
            
            # Calculate VWAP
            cumulative_tpv = (typical_price * valid_data['volume_1min']).sum()
            cumulative_volume = valid_data['volume_1min'].sum()
            
            if cumulative_volume <= 0:
                return float(valid_data['close_1min'].iloc[-1])
                
            vwap_price = cumulative_tpv / cumulative_volume
            return float(vwap_price)
        except:
            return self.get_current_price()
    
    def normalized_price_vwap(self) -> float:
        """Calculate current price normalized by intraday VWAP."""
        try:
            vwap_price = self.vwap()
            current_price = self.get_current_price()
            
            if vwap_price <= 0 or current_price <= 0:
                return 1.0
                
            # Ensure mathematical consistency: current_price / vwap
            ratio = current_price / vwap_price
            return float(np.clip(ratio, 0.5, 2.0))  # Reasonable bounds for intraday
        except:
            return 1.0
    
    # Technical Indicators
    def _calculate_technical_indicators(self) -> Dict[str, float]:
        """Calculate technical indicators (excluding absolute price features)."""
        features = {}
        
        features['MACD_12_26_9'] = self.macd_12_26_9()
        features['RSI_14'] = self.rsi_14()
        
        # REMOVE absolute Bollinger Bands, keep only width
        _, _, bb_width = self.bollinger_bands_20_2()
        features['Bollinger_Band_Width'] = bb_width
        
        features['Price_vs_SMA50'] = self.price_vs_sma50()
        
        return features
    
    def macd_12_26_9(self) -> float:
        """Calculate normalized MACD histogram."""
        if self.intra.empty or len(self.intra) < 35:
            return 0.0
            
        try:
            closes = self.intra['close_1min'].dropna()
            if len(closes) < 35:
                return 0.0
            
            # Calculate EMAs
            ema12 = closes.ewm(span=12, adjust=False).mean()
            ema26 = closes.ewm(span=26, adjust=False).mean()
            
            # MACD line
            macd_line = ema12 - ema26
            
            # Signal line
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            
            # MACD histogram
            macd_histogram = macd_line - signal_line
            
            # NORMALIZE: Convert to percentage of current price
            current_price = self.get_current_price()
            if current_price <= 0:
                return 0.0
            
            normalized_macd = (macd_histogram.iloc[-1] / current_price) * 100
            
            # Apply reasonable bounds
            return float(np.clip(normalized_macd, -2.0, 2.0))
        except:
            return 0.0
    
    def rsi_14(self) -> float:
        """Calculate 14-period RSI using Wilder's smoothing."""
        if self.intra.empty or len(self.intra) < 15:
            return 50.0
            
        try:
            closes = self.intra['close_1min'].dropna()
            if len(closes) < 15:
                return 50.0
            
            # Calculate price changes
            deltas = closes.diff()
            
            # Separate gains and losses
            gains = deltas.where(deltas > 0, 0)
            losses = -deltas.where(deltas < 0, 0)
            
            # Use Wilder's smoothing (equivalent to EMA with alpha = 1/14)
            avg_gains = gains.ewm(alpha=1/14, adjust=False).mean()
            avg_losses = losses.ewm(alpha=1/14, adjust=False).mean()
            
            # Calculate RSI
            current_avg_gain = avg_gains.iloc[-1]
            current_avg_loss = avg_losses.iloc[-1]
            
            if current_avg_loss == 0:
                return 100.0
                
            rs = current_avg_gain / current_avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(np.clip(rsi, 0.0, 100.0))
        except:
            return 50.0
    
    def bollinger_bands_20_2(self) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands (20,2) and band width."""
        current_price = self.get_current_price()
        
        if self.intra.empty or len(self.intra) < 20:
            return current_price, current_price, 0.0
            
        try:
            closes = self.intra['close_1min'].tail(20).dropna()
            if len(closes) < 20:
                return current_price, current_price, 0.0
            
            sma20 = closes.mean()
            std20 = closes.std()
            
            if std20 == 0 or sma20 <= 0:
                return current_price, current_price, 0.0
            
            upper_band = sma20 + 2 * std20
            lower_band = sma20 - 2 * std20
            
            # Calculate band width as percentage of middle band (SMA)
            band_width = (upper_band - lower_band) / sma20
            
            # Ensure mathematical consistency
            if upper_band <= lower_band:
                return current_price, current_price, 0.0
            
            return float(upper_band), float(lower_band), float(band_width)
        except:
            return current_price, current_price, 0.0
    
    def price_vs_sma50(self) -> float:
        """Calculate price relative to 50-period SMA."""
        current_price = self.get_current_price()
        
        try:
            # Try intraday data first
            if not self.intra.empty and len(self.intra) >= 50:
                sma50 = self.intra['close_1min'].tail(50).mean()
            # Fall back to daily data
            elif not self.daily.empty and len(self.daily) >= 50:
                sma50 = self.daily['close'].tail(50).mean()
            elif not self.daily.empty:
                sma50 = self.daily['close'].mean()
            else:
                return 0.0
            
            if sma50 <= 0:
                return 0.0
                
            return self.safe_percentage(current_price, sma50)
        except:
            return 0.0
    
    # Return Features
    def _calculate_return_features(self) -> Dict[str, float]:
        """Calculate all return-based features."""
        features = {}
        
        features['Return_5min'] = self.return_5min()
        features['Return_Since_Open'] = self.return_since_open()
        
        return features
    
    def return_5min(self) -> float:
        """Calculate 5-minute return."""
        if self.intra.empty or len(self.intra) < 6:  # Need at least 6 bars for 5-minute lookback
            return 0.0
            
        try:
            current_price = self.get_current_price()
            
            # Use close price from 5 bars ago (5 minutes ago)
            prices = self.intra['close_1min'].tail(6)  # Get last 6 bars
            if len(prices) < 6:
                return 0.0
                
            price_5min_ago = float(prices.iloc[0])  # First of the 6 bars (5 bars back from current)
            
            if price_5min_ago <= 0:
                return 0.0
                
            return_5min = self.safe_percentage(current_price, price_5min_ago)
            
            # For 5-minute returns on liquid stocks, cap at reasonable range
            return float(np.clip(return_5min, -5.0, 5.0))  # ±5% max in 5 minutes
        except:
            return 0.0
    
    def return_since_open(self) -> float:
        """Calculate return since market open."""
        if self.intra.empty:
            return 0.0
            
        try:
            open_price = float(self.intra['open_1min'].iloc[0])
            current_price = self.get_current_price()
            
            return self.safe_percentage(current_price, open_price)
        except:
            return 0.0
    
    # Volatility Features
    def _calculate_volatility_features(self) -> Dict[str, float]:
        """Calculate volatility features with normalized ATR."""
        features = {}
        
        features['ATR_Pct'] = self.atr_14()  # Now returns percentage
        features['Volatility_Regime_Shift_Norm'] = self.volatility_regime_shift()  # Now bounded [-1,1]
        features['Volatility_Percentile_100'] = self.volatility_percentile_100()
        features['ATR_Stop_Risk_Pct'] = self.atr_stop_risk_pct()
        
        return features
    
    @lru_cache(maxsize=1)
    def atr_14(self) -> float:
        """Calculate 14-period ATR as percentage of current price."""
        if self.intra.empty or len(self.intra) < 15:
            return 0.0
            
        try:
            data = self.intra[['high_1min', 'low_1min', 'close_1min']].tail(30).dropna()
            if len(data) < 15:
                return 0.0
            
            # Calculate true range components
            data = data.copy()
            data['hl'] = data['high_1min'] - data['low_1min']
            data['hc'] = abs(data['high_1min'] - data['close_1min'].shift(1))
            data['lc'] = abs(data['low_1min'] - data['close_1min'].shift(1))
            
            # True range is the maximum of the three components
            data['true_range'] = data[['hl', 'hc', 'lc']].max(axis=1)
            
            # Handle first period where shifted values are NaN
            data['true_range'].iloc[0] = data['hl'].iloc[0]
            
            # Calculate ATR using Wilder's smoothing
            atr_series = data['true_range'].ewm(alpha=1/14, adjust=False).mean()
            atr_value = float(atr_series.iloc[-1])
            
            # NORMALIZE: Convert to percentage of current price
            current_price = self.get_current_price()
            if current_price <= 0:
                return 0.0
                
            return (atr_value / current_price) * 100  # Return as percentage
        except:
            return 0.0
    

    def calculate_comprehensive_r_levels(self, atr_multiplier: float = 1.0) -> Dict[str, float]:
        """
        Calculate comprehensive R-levels with multiple targets and risk metrics.
        
        Args:
            atr_multiplier (float): ATR multiplier for stop distance (default 1.0)
        
        Returns:
            Dictionary with multiple R-level targets and risk metrics
        """
        try:
            current_price = self.get_current_price()
            atr_pct = self.atr_14()
            
            if current_price <= 0 or atr_pct <= 0:
                return {'Error': 'Invalid price or ATR data'}
            
            # Calculate base risk distance
            atr_absolute = (atr_pct / 100) * current_price
            risk_distance = atr_multiplier * atr_absolute
            
            # Stop loss level
            stop_loss = current_price - risk_distance
            
            # Multiple take profit targets
            targets = {}
            target_mapping = {1.0: '1R_Target', 1.5: '1_5R_Target', 2.0: '2R_Target', 3.0: '3R_Target', 4.0: '4R_Target'}
            for r_multiple in [1.0, 1.5, 2.0, 3.0, 4.0]:
                target_price = current_price + (r_multiple * risk_distance)
                key_name = target_mapping[r_multiple]
                targets[key_name] = round(target_price, 2)
            
            # Risk metrics
            risk_amount_pct = (risk_distance / current_price) * 100
            
            # Combine all results
            result = {
                'Current_Price': round(current_price, 2),
                '1R_Stop_Loss': round(stop_loss, 2),
                'Risk_Amount_Pct': round(risk_amount_pct, 2),
                'ATR_Distance': round(atr_absolute, 3),
                'ATR_Multiplier_Used': atr_multiplier,
                **targets
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to calculate comprehensive R levels: {e}")
            return {'Error': f'Calculation failed: {e}'}




    def _calculate_risk_management_features(self) -> Dict[str, float]:
        """Calculate risk management features including R-levels."""
        # Get comprehensive R-levels
        r_levels = self.calculate_comprehensive_r_levels(atr_multiplier=1.0)
        
        # You can optionally add position sizing calculations
        risk_pct = r_levels.get('Risk_Amount_Pct', 0)
        if risk_pct > 0:
            r_levels['Position_Size_1pct_Risk'] = round(1.0 / risk_pct * 100, 2)  # For 1% account risk
        else:
            r_levels['Position_Size_1pct_Risk'] = 0.0
        
        return r_levels



    def volatility_regime_shift(self) -> float:
        """Detect volatility regime shifts with normalized output."""
        if self.intra.empty or len(self.intra) < 30:
            return 0.0
            
        try:
            # Calculate returns
            returns = self.intra['close_1min'].pct_change().dropna()
            if len(returns) < 30:
                return 0.0
            
            # Recent vs historical volatility
            recent_returns = returns.tail(10)
            historical_returns = returns.iloc[-30:-10]
            
            if len(recent_returns) < 5 or len(historical_returns) < 10:
                return 0.0
            
            current_vol = recent_returns.std()
            historical_vol = historical_returns.std()
            
            if historical_vol == 0:
                return 0.0
                
            # Calculate regime shift signal
            ratio = current_vol / historical_vol
            
            # IMPROVED NORMALIZATION: Use tanh transformation for bounded output
            if ratio > 1.0:
                # High volatility regime: transform to [0, 1] range
                excess = ratio - 1.0
                normalized = np.tanh(excess / 2.0)  # Smooth saturation
                return float(normalized)
            else:
                # Low volatility regime: transform to [-1, 0] range  
                deficit = 1.0 - ratio
                normalized = -np.tanh(deficit / 2.0)
                return float(normalized)
        except:
            return 0.0
    
    def volatility_percentile_100(self) -> float:
        """Calculate ATR percentile over 100 periods."""
        if self.intra.empty or len(self.intra) < 100:
            return 50.0
            
        try:
            # Get sufficient data for rolling ATR calculation
            data = self.intra[['high_1min', 'low_1min', 'close_1min']].tail(130).dropna()
            if len(data) < 100:
                return 50.0
            
            # Calculate true range
            data = data.copy()
            data['hl'] = data['high_1min'] - data['low_1min']
            data['hc'] = abs(data['high_1min'] - data['close_1min'].shift(1))
            data['lc'] = abs(data['low_1min'] - data['close_1min'].shift(1))
            data['true_range'] = data[['hl', 'hc', 'lc']].max(axis=1)
            
            # Handle first period
            data['true_range'].iloc[0] = data['hl'].iloc[0]
            
            # Calculate rolling ATR
            data['atr'] = data['true_range'].ewm(alpha=1/14, adjust=False).mean()
            
            # Get last 100 ATR values
            atr_values = data['atr'].dropna().tail(100)
            if len(atr_values) < 100:
                return 50.0
            
            current_atr = atr_values.iloc[-1]
            percentile = (atr_values <= current_atr).sum() / len(atr_values) * 100
            
            return float(np.clip(percentile, 0.0, 100.0))
        except:
            return 50.0
    
    def atr_stop_risk_pct(self) -> float:
        """Calculate volatility-normalized stop distance."""
        try:
            atr = self.atr_14()
            current_price = self.get_current_price()
            
            if current_price <= 0 or atr <= 0:
                return 0.0
                
            # Typical stop at 2 ATR
            stop_distance = 2 * atr
            risk_pct = self.safe_divide(stop_distance, current_price, 0.0) * 100
            
            return float(np.clip(risk_pct, 0.0, 50.0))  # Cap at reasonable maximum
        except:
            return 0.0
    
    # Price Pattern Features
    def _calculate_pattern_features(self) -> Dict[str, float]:
        """Calculate all price pattern features."""
        features = {}
        
        features['Position_In_Day_Range'] = self.position_in_day_range()
        features['Support_Resistance_Proximity'] = self.support_resistance_proximity()
        features['Breakout_Confirmation'] = self.breakout_confirmation()
        features['Price_Acceleration'] = self.price_acceleration()
        features['Max_Drawdown_30'] = self.max_drawdown_30()
        
        return features
    
    def position_in_day_range(self) -> float:
        """Calculate position of price within today's range."""
        if self.intra.empty:
            return 0.5
            
        try:
            day_high = self.intra['high_1min'].max()
            day_low = self.intra['low_1min'].min()
            current_price = self.get_current_price()
            
            if day_high <= day_low or current_price <= 0:
                return 0.5
                
            position = self.safe_divide(current_price - day_low, day_high - day_low, 0.5)
            return float(np.clip(position, 0.0, 1.0))
        except:
            return 0.5
    
    def support_resistance_proximity(self) -> float:
        """Calculate distance to nearest support/resistance level."""
        if self.intra.empty or len(self.intra) < 20:
            return 5.0
            
        try:
            # Use a more robust approach to identify S/R levels
            data = self.intra[['high_1min', 'low_1min', 'close_1min']].tail(50).dropna()
            if len(data) < 20:
                return 5.0
            
            current_price = self.get_current_price()
            if current_price <= 0:
                return 5.0
            
            # Identify significant price levels using multiple timeframes
            resistance_levels = set()
            support_levels = set()
            
            # Find swing highs and lows over different periods
            for window in [10, 20]:
                if len(data) >= window:
                    # Rolling max/min to find swing points
                    highs = data['high_1min'].rolling(window=window, center=True).max()
                    lows = data['low_1min'].rolling(window=window, center=True).min()
                    
                    # Identify actual swing points (where price equals the rolling max/min)
                    swing_highs = data['high_1min'][data['high_1min'] == highs].values
                    swing_lows = data['low_1min'][data['low_1min'] == lows].values
                    
                    resistance_levels.update(swing_highs)
                    support_levels.update(swing_lows)
            
            # Also add recent significant levels (top and bottom 10% of range)
            price_range = data['high_1min'].max() - data['low_1min'].min()
            if price_range > 0:
                high_threshold = data['high_1min'].max() - price_range * 0.1
                low_threshold = data['low_1min'].min() + price_range * 0.1
                
                resistance_levels.update(data['high_1min'][data['high_1min'] >= high_threshold].values)
                support_levels.update(data['low_1min'][data['low_1min'] <= low_threshold].values)
            
            # Remove levels too close to current price (less than 0.1%)
            min_distance = current_price * 0.001  # 0.1%
            resistance_levels = [r for r in resistance_levels if abs(r - current_price) >= min_distance]
            support_levels = [s for s in support_levels if abs(s - current_price) >= min_distance]
            
            # Find nearest levels
            above_price = [r for r in resistance_levels if r > current_price]
            below_price = [s for s in support_levels if s < current_price]
            
            distances = []
            
            if above_price:
                nearest_resistance = min(above_price)
                res_distance = abs(nearest_resistance - current_price) / current_price * 100
                distances.append(res_distance)
            
            if below_price:
                nearest_support = max(below_price)
                sup_distance = abs(current_price - nearest_support) / current_price * 100
                distances.append(sup_distance)
            
            if not distances:
                return 5.0  # Default when no levels found
                
            # Return minimum distance, but ensure it's at least 0.1%
            min_distance_pct = min(distances)
            return float(max(min_distance_pct, 0.1))
        except:
            return 5.0
    
    def breakout_confirmation(self) -> float:
        """Detect volume-confirmed breakout above resistance."""
        if self.intra.empty or len(self.intra) < 20:
            return 0.0
            
        try:
            data = self.intra[['high_1min', 'volume_1min']].tail(20).dropna()
            if len(data) < 20:
                return 0.0
            
            # Get recent high (resistance level) - exclude current bar
            historical_data = data.iloc[:-1]  # Everything except current bar
            recent_high = historical_data['high_1min'].max()
            
            current_price = self.get_current_price()
            current_volume = float(data['volume_1min'].iloc[-1])
            avg_volume = historical_data['volume_1min'].mean()
            
            # FIXED: Proper breakout logic
            price_breakout = current_price > recent_high
            
            # FIXED: Volume confirmation requires INCREASE, not decrease
            volume_confirmation = (current_volume > 1.5 * avg_volume) if avg_volume > 0 else False
            
            if price_breakout and volume_confirmation:
                # Calculate strength of breakout
                price_strength = (current_price - recent_high) / recent_high if recent_high > 0 else 0
                volume_strength = (current_volume / avg_volume - 1.0) if avg_volume > 0 else 0
                
                # Combine signals (cap at 1.0)
                breakout_strength = min(1.0, price_strength * 100 + volume_strength * 0.1)
                return float(max(0.0, breakout_strength))
            
            return 0.0
        except:
            return 0.0
    
    def price_acceleration(self) -> float:
        """Calculate second derivative of price movement."""
        if self.intra.empty or len(self.intra) < 3:
            return 0.0
            
        try:
            prices = self.intra['close_1min'].tail(5).dropna()
            if len(prices) < 3:
                return 0.0
            
            # First derivative (velocity)
            velocity = prices.diff()
            
            # Second derivative (acceleration)
            acceleration = velocity.diff()
            
            # Normalize by price level to make it price-independent
            current_price = self.get_current_price()
            if current_price <= 0:
                return 0.0
                
            normalized_acceleration = acceleration.iloc[-1] / current_price * 100
            
            # Cap at reasonable limits
            return float(np.clip(normalized_acceleration, -10.0, 10.0))
        except:
            return 0.0
    
    def max_drawdown_30(self) -> float:
        """Calculate maximum drawdown over last 30 bars."""
        if self.intra.empty or len(self.intra) < 5:
            return 0.0
            
        try:
            n_bars = min(30, len(self.intra))
            prices = self.intra['close_1min'].tail(n_bars).dropna()
            
            if len(prices) < 2:
                return 0.0
            
            # Calculate running maximum
            running_max = prices.expanding().max()
            
            # Calculate drawdown
            drawdown = (prices - running_max) / running_max * 100
            
            max_dd = float(drawdown.min())
            
            # Ensure drawdown is negative or zero
            return min(0.0, max_dd)
        except:
            return 0.0
    
    # Time Features
    def _calculate_time_features(self) -> Dict[str, float]:
        """Calculate all time-based features."""
        features = {}
        
        sin_time, cos_time = self.time_of_day_cyclical()
        features['Time_Of_Day_Cyclical_Sin'] = sin_time
        features['Time_Of_Day_Cyclical_Cos'] = cos_time
        
        return features
    
    def time_of_day_cyclical(self) -> Tuple[float, float]:
        """Encode time of day as cyclical features."""
        try:
            if 'timestamp' in self.rt.columns:
                current_time = pd.to_datetime(self.rt['timestamp'].iloc[0])
            else:
                # Fallback to current time if timestamp not available
                current_time = pd.Timestamp.now()
            
            # Convert to market time if needed (assuming Eastern Time for US markets)
            minutes_since_midnight = current_time.hour * 60 + current_time.minute
            
            # Convert to radians (full day = 2π)
            angle = 2 * np.pi * minutes_since_midnight / (24 * 60)
            
            return float(np.sin(angle)), float(np.cos(angle))
        except:
            return 0.0, 1.0

    def validate_feature_consistency(self, features: Dict[str, float]) -> Dict[str, str]:
        """
        Validate mathematical consistency between related features.
        Returns dictionary of validation warnings.
        """
        warnings = {}
        
        try:
            # 1. Validate VWAP consistency
            current_price = self.get_current_price()
            if current_price > 0 and features.get('VWAP', 0) > 0:
                expected_norm = current_price / features['VWAP']
                actual_norm = features.get('Normalized_Price_VWAP', 1.0)
                if abs(expected_norm - actual_norm) > 0.001:  # 0.1% tolerance
                    warnings['VWAP_Normalization'] = f"Expected {expected_norm:.4f}, got {actual_norm:.4f}"
            
            # 2. Validate Bollinger Bands consistency
            bb_upper = features.get('Bollinger_Bands_Upper', 0)
            bb_lower = features.get('Bollinger_Bands_Lower', 0)
            bb_width = features.get('Bollinger_Band_Width', 0)
            
            if bb_upper > 0 and bb_lower > 0 and bb_width > 0:
                mid_band = (bb_upper + bb_lower) / 2
                expected_width = (bb_upper - bb_lower) / mid_band
                if abs(expected_width - bb_width) > 0.001:
                    warnings['Bollinger_Bands'] = f"Width inconsistency: expected {expected_width:.4f}, got {bb_width:.4f}"
                
                if bb_upper <= bb_lower:
                    warnings['Bollinger_Bands_Order'] = f"Upper band ({bb_upper:.2f}) <= Lower band ({bb_lower:.2f})"
            
            # 3. Validate Order Book Imbalances consistency
            regular_imb = features.get('Order_Book_Imbalance', 0)
            vw_imb = features.get('Volume_Weighted_Order_Book_Imbalance', 0)
            
            # They should generally have the same sign (both positive or both negative)
            if abs(regular_imb) > 0.1 and abs(vw_imb) > 0.1:
                if (regular_imb > 0 and vw_imb < 0) or (regular_imb < 0 and vw_imb > 0):
                    warnings['Imbalance_Signs'] = f"Regular: {regular_imb:.3f}, VW: {vw_imb:.3f} have opposite signs"
            
            # 4. Validate spread percentile reasonableness
            spread = features.get('Bid_Ask_Spread_Pct', 0)
            spread_pct = features.get('Spread_Percentile_20', 0.5)
            
            if spread <= 0 and spread_pct not in [0.0, 0.5, 1.0]:
                warnings['Spread_Percentile'] = f"Zero spread but percentile is {spread_pct:.2f} (should be 0.5)"
        
            
            # 5. Validate ATR and volatility consistency
            atr = features.get('ATR_14', 0)
            vol_regime = features.get('Volatility_Regime_Shift', 0)
            
            if atr > 0 and abs(vol_regime) > 1.0:  # Strong regime shift
                # ATR should be relatively high for strong regime shifts
                current_price = self.get_current_price()
                if current_price > 0:
                    atr_pct = atr / current_price * 100
                    if atr_pct < 0.05:  # Less than 0.05% ATR with strong regime shift
                        warnings['Volatility_Consistency'] = f"Low ATR ({atr_pct:.3f}%) with strong regime shift ({vol_regime:.2f})"
            
            # 6. Validate price position bounds
            price_pos = features.get('Price_Position_In_Spread', 0.5)
            if not (0 <= price_pos <= 1):
                warnings['Price_Position_Bounds'] = f"Price position {price_pos:.3f} outside [0,1] range"
            
            day_pos = features.get('Position_In_Day_Range', 0.5)
            if not (0 <= day_pos <= 1):
                warnings['Day_Position_Bounds'] = f"Day position {day_pos:.3f} outside [0,1] range"
            
            # 7. Validate RSI bounds
            rsi = features.get('RSI_14', 50)
            if not (0 <= rsi <= 100):
                warnings['RSI_Bounds'] = f"RSI {rsi:.1f} outside [0,100] range"

            # ADD R-LEVELS VALIDATION
            current_price = features.get('Current_Price', 0)
            stop_loss = features.get('1R_Stop_Loss', 0)
            target_2r = features.get('2R_Target', 0)
            
            if current_price > 0:
                # Validate stop loss is below current price
                if stop_loss >= current_price:
                    warnings['R_Levels_Stop'] = f"Stop loss ({stop_loss}) >= current price ({current_price})"
                
                # Validate targets are above current price
                if target_2r <= current_price:
                    warnings['R_Levels_Target'] = f"2R target ({target_2r}) <= current price ({current_price})"
            
        except Exception as e:
            warnings['Validation_Error'] = f"Validation failed: {e}"
        
        return warnings


def calculate_features(real_time_df: pd.DataFrame, 
                      intra_day_df: pd.DataFrame, 
                      daily_df: pd.DataFrame,
                      validate: bool = True) -> Dict[str, float]:
    """
    Main function to calculate all features from the three data sources.
    
    Args:
        real_time_df: Real-time data DataFrame with columns:
                     ['bidPrice', 'bidSize', 'askPrice', 'askSize', 'lastPrice', 'timestamp']
        intra_day_df: Intraday minute data DataFrame with columns:
                     ['open_1min', 'high_1min', 'low_1min', 'close_1min', 'volume_1min']
        daily_df: Daily OHLCV data DataFrame with columns:
                 ['open', 'high', 'low', 'close', 'volume', 'date']
        validate: Whether to run consistency validation
        
    Returns:
        Dictionary with all calculated features
        
    Raises:
        DataValidationError: If data validation fails
    """
    try:
        calculator = FeatureCalculator(real_time_df, intra_day_df, daily_df)
        features = calculator.calculate_all_features()
        
        # Run validation if requested
        if validate:
            warnings = calculator.validate_feature_consistency(features)
            if warnings:
                logger.warning("Feature consistency warnings:")
                for feature, warning in warnings.items():
                    logger.warning(f"  {feature}: {warning}")
        
        return features
    except Exception as e:
        logger.error(f"Feature calculation failed: {e}")
        # Return default values instead of failing
        return _get_default_features_static()


# UPDATED: _get_default_features_static() function

def _get_default_features_static() -> Dict[str, float]:
    """Static method to get default features when calculator fails to initialize."""
    return {
        # Microstructure features (updated names)
        'Bid_Ask_Spread_Pct': 0.0,              # Changed from 'Bid_Ask_Spread'
        'Order_Book_Imbalance': 0.0,
        'Price_Position_In_Spread': 0.5,
        'Mid_Price_Momentum': 0.0,
        'Spread_Percentile_20': 0.5,
        'Volume_Weighted_Order_Book_Imbalance': 0.0,
        'Trade_Count_Per_Minute_Log': 0.0,      # Changed from 'Trade_Count_Per_Minute'
        'Volume_Rate_of_Change': 0.0,
        'Price_Impact_Normalized': 0.0,          # Changed from 'Price_Impact'
        
        # Volume features (removed absolute VWAP)
        'Volume_vs_SMA10': 1.0,
        'Normalized_Price_VWAP': 1.0,           # Kept only normalized version
        
        # Technical indicators (removed absolute Bollinger Bands)
        'MACD_12_26_9': 0.0,
        'RSI_14': 50.0,
        'Bollinger_Band_Width': 0.0,           # Kept only width, removed Upper/Lower
        'Price_vs_SMA50': 0.0,
        
        # Return features (unchanged)
        'Return_5min': 0.0,
        'Return_Since_Open': 0.0,
        
        # Volatility features (updated names)
        'ATR_Pct': 0.0,                        # Changed from 'ATR_14'
        'Volatility_Regime_Shift_Norm': 0.0,   # Changed from 'Volatility_Regime_Shift'
        'Volatility_Percentile_100': 50.0,
        'ATR_Stop_Risk_Pct': 0.0,
        
        # Price pattern features (unchanged)
        'Position_In_Day_Range': 0.5,
        'Support_Resistance_Proximity': 5.0,
        'Breakout_Confirmation': 0.0,
        'Price_Acceleration': 0.0,
        'Max_Drawdown_30': 0.0,
        
        # Time features (unchanged)
        'Time_Of_Day_Cyclical_Sin': 0.0,
        'Time_Of_Day_Cyclical_Cos': 1.0,
        'Current_Price': 0.0,
        '1R_Stop_Loss': 0.0,
        'Risk_Amount_Pct': 0.0,
        'ATR_Distance': 0.0,
        '1R_Target': 0.0,
        '1_5R_Target': 0.0,
        '2R_Target': 0.0,
        '3R_Target': 0.0,
        '4R_Target': 0.0,
        'Position_Size_1pct_Risk': 0.0
    }


# Example usage and testing
if __name__ == "__main__":
    # Example data creation for testing
    import datetime
    
    # Create sample real-time data
    rt_data = pd.DataFrame({
        'bidPrice': [227.77],
        'bidSize': [500],
        'askPrice': [227.90],
        'askSize': [54],
        'lastPrice': [227.85],
        'lastSize': [200],
        'volume': [1500000],
        'timestamp': ['2025-07-12T08:01:00Z'],
        'tradeSide': ['B']
    })
    
    # Create sample intraday data
    dates = pd.date_range('2025-07-12 08:00:00', periods=60, freq='1min')
    intra_data = pd.DataFrame({
        'ts_event_clean': dates,
        'open_1min': np.random.normal(227.9, 0.5, 60),
        'high_1min': np.random.normal(228.2, 0.5, 60),
        'low_1min': np.random.normal(227.6, 0.5, 60),
        'close_1min': np.random.normal(227.9, 0.5, 60),
        'volume_1min': np.random.randint(1000, 10000, 60),
        'bid_px_00': np.random.normal(227.77, 0.5, 60),
        'ask_px_00': np.random.normal(227.90, 0.5, 60)
    })
    
    # Create sample daily data
    daily_dates = pd.date_range('2025-01-01', periods=100, freq='1D')
    daily_data = pd.DataFrame({
        'date': daily_dates,
        'open': np.random.normal(225, 5, 100),
        'high': np.random.normal(230, 5, 100),
        'low': np.random.normal(220, 5, 100),
        'close': np.random.normal(225, 5, 100),
        'volume': np.random.randint(1000000, 10000000, 100)
    })
    
    # Calculate features
    try:
        features = calculate_features(rt_data, intra_data, daily_data, validate=True)
        print("Feature calculation successful!")
        print(f"Calculated {len(features)} features:")
        for feature, value in features.items():
            print(f"{feature}: {value:.6f}")
            
        # Test validation separately
        print("\n" + "="*50)
        print("VALIDATION TEST")
        print("="*50)
        
        calculator = FeatureCalculator(rt_data, intra_data, daily_data)
        warnings = calculator.validate_feature_consistency(features)
        
        if warnings:
            print("⚠️  Consistency warnings found:")
            for feature, warning in warnings.items():
                print(f"   {feature}: {warning}")
        else:
            print("✅ All features passed consistency validation!")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()