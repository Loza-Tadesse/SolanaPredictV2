import requests
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime, timedelta
from mlProject import logger
from mlProject.entity.config_entity import DataIngestionConfig
import zipfile


class CryptoDataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
        self.base_url = "https://api.coingecko.com/api/v3"
        
    def get_cryptocompare_minute_data(self, crypto_symbol: str, total_minutes: int = 43200, batch_size: int = 2000):
        """Fetch minute-level data from CryptoCompare (up to 2000 samples per request)."""
        url = "https://min-api.cryptocompare.com/data/v2/histominute"
        remaining = max(total_minutes, batch_size)
        to_timestamp = int(datetime.utcnow().timestamp())
        frames = []

        while remaining > 0:
            limit = min(batch_size, remaining)
            params = {
                'fsym': crypto_symbol.upper(),
                'tsym': 'USD',
                'limit': limit,
                'toTs': to_timestamp,
                'aggregate': 1
            }

            try:
                logger.info(
                    "Fetching %s minute candles from CryptoCompare for %s (remaining=%s)",
                    limit,
                    crypto_symbol,
                    remaining,
                )
                response = requests.get(url, params=params, timeout=20)
                response.raise_for_status()

                data = response.json()
                if data.get('Response') != 'Success':
                    logger.warning("CryptoCompare minute API returned non-success: %s", data.get('Message'))
                    break

                chunk = pd.DataFrame(data['Data']['Data'])
                if chunk.empty:
                    logger.warning("Received empty minute data chunk from CryptoCompare")
                    break

                chunk['datetime'] = pd.to_datetime(chunk['time'], unit='s')
                chunk = chunk.rename(columns={
                    'close': 'price',
                    'volumefrom': 'volume',
                    'volumeto': 'market_cap'
                })
                chunk = chunk[['datetime', 'price', 'volume', 'market_cap']]
                frames.append(chunk)

                to_timestamp = int(chunk['datetime'].min().timestamp()) - 60
                remaining -= limit
                time.sleep(0.25)
            except Exception as exc:
                logger.warning("CryptoCompare minute API failed: %s", exc)
                break

        if not frames:
            return None

        df = pd.concat(frames, ignore_index=True)
        df = df.drop_duplicates(subset='datetime')
        df = df.sort_values('datetime').set_index('datetime')
        logger.info("CryptoCompare: fetched %d minute data points", len(df))
        return df
    
    def get_binance_data(self, symbol, interval='1h', limit=1000):
        """Fetch data from Binance (Free: 1000 candles per request)"""
        try:
            url = "https://api.binance.com/api/v3/klines"
            params = {'symbol': f'{symbol}USDT', 'interval': interval, 'limit': limit}
            
            logger.info(f"Fetching {limit} candles from Binance for {symbol}")
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df['price'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            df['market_cap'] = df['quote_volume'].astype(float)
            df = df[['datetime', 'price', 'volume', 'market_cap']].set_index('datetime')
            logger.info(f"Binance: Fetched {len(df)} data points")
            return df
        except Exception as e:
            logger.warning(f"Binance API failed: {e}")
            return None
        
    def get_crypto_data(self, crypto_id="bitcoin", vs_currency="usd", days=365):
        """
        Fetch cryptocurrency data from CoinGecko API
        
        Args:
            crypto_id (str): Cryptocurrency ID (bitcoin, ethereum, etc.)
            vs_currency (str): Currency to compare against
            days (int): Number of days of historical data
        """
        try:
            # Historical price data
            price_url = f"{self.base_url}/coins/{crypto_id}/market_chart"
            params = {
                'vs_currency': vs_currency,
                'days': days,
                'interval': 'hourly' if days <= 90 else 'daily'
            }
            
            logger.info(f"Fetching {crypto_id} data for {days} days...")
            response = requests.get(price_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Convert to DataFrame
            prices = data['prices']
            volumes = data['total_volumes']
            market_caps = data['market_caps']
            
            df = pd.DataFrame({
                'timestamp': [item[0] for item in prices],
                'price': [item[1] for item in prices],
                'volume': [item[1] for item in volumes],
                'market_cap': [item[1] for item in market_caps]
            })
            
            # Convert timestamp to datetime
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.drop('timestamp', axis=1)
            df = df.set_index('datetime')
            
            return df
            
        except Exception as e:
            logger.exception(f"Error fetching crypto data: {str(e)}")
            raise e
    
    def add_technical_indicators(self, df):
        """Add technical indicators to the dataset using minute-level windows."""
        try:
            # Rolling window definitions (in minutes)
            SMA_SHORT = 60          # ~1 hour
            SMA_MEDIUM = 360        # ~6 hours
            SMA_LONG = 1440         # ~1 day
            BB_WINDOW = 120         # ~2 hours
            RSI_PERIOD = 30         # 30 minutes
            PRICE_CHANGE_1H = 60
            PRICE_CHANGE_24H = 1440
            PRICE_CHANGE_7D = 10080
            VOLUME_WINDOW = 180     # 3 hours
            VOLATILITY_WINDOW = 120 # 2 hours
            POSITION_WINDOW = 60 * 24 * 14  # 14 days

            # Simple Moving Averages
            df['sma_7'] = df['price'].rolling(window=SMA_SHORT, min_periods=1).mean()
            df['sma_14'] = df['price'].rolling(window=SMA_MEDIUM, min_periods=1).mean()
            df['sma_30'] = df['price'].rolling(window=SMA_LONG, min_periods=1).mean()

            # Exponential Moving Averages
            df['ema_7'] = df['price'].ewm(span=SMA_SHORT, adjust=False).mean()
            df['ema_14'] = df['price'].ewm(span=SMA_MEDIUM, adjust=False).mean()

            # MACD (Moving Average Convergence Divergence)
            ema_fast = df['price'].ewm(span=12, adjust=False).mean()
            ema_slow = df['price'].ewm(span=26, adjust=False).mean()
            df['macd'] = ema_fast - ema_slow
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']

            # RSI (Relative Strength Index)
            delta = df['price'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=RSI_PERIOD, min_periods=1).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=RSI_PERIOD, min_periods=1).mean()
            rs = gain / loss.replace({0: np.nan})
            df['rsi'] = 100 - (100 / (1 + rs))
            df['rsi'] = df['rsi'].fillna(50.0)

            # Bollinger Bands
            df['bb_middle'] = df['price'].rolling(window=BB_WINDOW, min_periods=1).mean()
            bb_std = df['price'].rolling(window=BB_WINDOW, min_periods=1).std().fillna(0.0)
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)

            # Price change percentages
            df['price_change_1h'] = df['price'].pct_change(PRICE_CHANGE_1H).fillna(0.0)
            df['price_change_24h'] = df['price'].pct_change(PRICE_CHANGE_24H).fillna(0.0)
            df['price_change_7d'] = df['price'].pct_change(PRICE_CHANGE_7D).fillna(0.0)

            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=VOLUME_WINDOW, min_periods=1).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            df['volume_ratio'] = df['volume_ratio'].replace([np.inf, -np.inf], np.nan).fillna(1.0)

            # Volatility
            df['volatility'] = df['price'].rolling(window=VOLATILITY_WINDOW, min_periods=1).std().fillna(0.0)

            # Price position relative to recent high/low
            df['high_14d'] = df['price'].rolling(window=POSITION_WINDOW, min_periods=1).max()
            df['low_14d'] = df['price'].rolling(window=POSITION_WINDOW, min_periods=1).min()
            range_span = (df['high_14d'] - df['low_14d']).replace(0, np.nan)
            df['price_position'] = (df['price'] - df['low_14d']) / range_span
            df['price_position'] = df['price_position'].clip(0.0, 1.0).fillna(0.5)

            return df

        except Exception as e:
            logger.exception(f"Error adding technical indicators: {str(e)}")
            raise e
    
    def create_prediction_targets(self, df):
        """Create prediction targets (future prices)"""
        try:
            df['target_price_1min'] = df['price'].shift(-1)
            df['target_price_5min'] = df['price'].shift(-5)
            df['target_price_1h'] = df['price'].shift(-60)
            df['target_price_24h'] = df['price'].shift(-1440)

            # Back-fill price-based targets when insufficient future data exists
            for col in ['target_price_5min', 'target_price_1h', 'target_price_24h']:
                df[col] = df[col].fillna(df['price'])

            df['target_direction_1min'] = np.where(df['target_price_1min'] > df['price'], 1, 0)
            df['target_direction_5min'] = np.where(df['target_price_5min'] > df['price'], 1, 0)
            df['target_direction_1h'] = np.where(df['target_price_1h'] > df['price'], 1, 0)
            df['target_direction_24h'] = np.where(df['target_price_24h'] > df['price'], 1, 0)

            df['target_change_1min'] = (df['target_price_1min'] - df['price']) / df['price'] * 100
            df['target_change_5min'] = (df['target_price_5min'] - df['price']) / df['price'] * 100
            df['target_change_1h'] = (df['target_price_1h'] - df['price']) / df['price'] * 100
            df['target_change_24h'] = (df['target_price_24h'] - df['price']) / df['price'] * 100

            # Replace residual NaNs introduced by limited look-ahead horizon
            change_cols = ['target_change_1min', 'target_change_5min', 'target_change_1h', 'target_change_24h']
            df[change_cols] = df[change_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            
            return df
            
        except Exception as e:
            logger.exception(f"Error creating prediction targets: {str(e)}")
            raise e

    def _gather_crypto_sources(self, crypto: str, symbols: dict):
        """Fetch raw dataframes from available providers for a given asset."""
        frames = []

        cc_df = self.get_cryptocompare_minute_data(
            symbols['cryptocompare'],
            total_minutes=60 * 24 * 60,
        )
        if cc_df is not None and not cc_df.empty:
            frames.append(cc_df)
        time.sleep(0.5)

        binance_df = self.get_binance_data(
            symbols['binance'],
            interval='1m',
            limit=1000,
        )
        if binance_df is not None and not binance_df.empty:
            frames.append(binance_df)
        time.sleep(0.5)

        try:
            cg_df = self.get_crypto_data(crypto_id=crypto, days=90)
            if cg_df is not None and not cg_df.empty:
                frames.append(cg_df)
        except Exception as exc:
            logger.warning(f"CoinGecko failed for {crypto}: {exc}")

        return frames

    def _finalize_dataframe(self, dfs, crypto: str):
        if not dfs:
            return None

        combined_df = pd.concat(dfs)
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
        combined_df = combined_df.sort_index()

        logger.info(f"Total data points after aggregation: {len(combined_df)}")

        combined_df = self.add_technical_indicators(combined_df)
        combined_df = self.create_prediction_targets(combined_df)
        combined_df['crypto_symbol'] = crypto

        # Restrict to the most recent four hours of minute-level data for training
        latest_timestamp = combined_df.index.max()
        if pd.notnull(latest_timestamp):
            four_hour_cutoff = latest_timestamp - timedelta(hours=4)
            combined_df = combined_df[combined_df.index >= four_hour_cutoff]

        # Ensure statistical stability by removing rows without the immediate target
        combined_df = combined_df.replace([np.inf, -np.inf], np.nan)
        combined_df = combined_df.dropna(subset=['price', 'volume', 'market_cap', 'target_price_1min'])
        combined_df = combined_df.fillna(0.0)

        combined_df = combined_df.dropna().reset_index()
        if 'datetime' in combined_df.columns:
            combined_df = combined_df.drop(columns=['datetime'])

        logger.info(f"Final clean data points: {len(combined_df)}")
        return combined_df
    
    def download_file(self, cryptocurrencies=None):
        """Main method to fetch and aggregate crypto data from multiple APIs"""
        try:
            os.makedirs(os.path.dirname(self.config.local_data_file), exist_ok=True)
            
            if cryptocurrencies is None:
                cryptos = ['solana']
            else:
                cryptos = cryptocurrencies
            
            # Crypto symbol mapping for different APIs
            symbol_map = {
                'solana': {'cryptocompare': 'SOL', 'binance': 'SOL', 'coingecko': 'solana'},
                'bitcoin': {'cryptocompare': 'BTC', 'binance': 'BTC', 'coingecko': 'bitcoin'},
                'ethereum': {'cryptocompare': 'ETH', 'binance': 'ETH', 'coingecko': 'ethereum'}
            }
            
            all_data = []
            
            for crypto in cryptos:
                logger.info("=" * 70)
                logger.info(f"Aggregating data for {crypto} from multiple sources...")

                symbols = symbol_map.get(crypto)
                if symbols is None:
                    logger.warning(f"{crypto} not supported in symbol map")
                    continue

                data_frames = self._gather_crypto_sources(crypto, symbols)
                processed_df = self._finalize_dataframe(data_frames, crypto)

                if processed_df is None or processed_df.empty:
                    logger.error(f"No data sources available for {crypto}")
                    continue

                all_data.append(processed_df)
            
            if not all_data:
                raise ValueError("No crypto data collected; all data sources failed")

            # Combine all crypto data
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Save to CSV
            output_path = self.config.local_data_file.replace('.zip', '.csv')
            combined_df.to_csv(output_path, index=False)
            
            logger.info(f"Crypto data saved to {output_path}")
            logger.info(f"Dataset shape: {combined_df.shape}")
            logger.info(f"Columns: {list(combined_df.columns)}")
            
            return output_path
            
        except Exception as e:
            logger.exception(f"Error in download_file: {str(e)}")
            raise e
    
    def extract_zip_file(self):
        """Extract zip file (not needed for crypto data, but keeping for compatibility)"""
        try:
            # For crypto data, we don't need zip extraction
            # But we'll create the unzip directory anyway
            os.makedirs(self.config.unzip_dir, exist_ok=True)
            
            # Check if CSV file exists (no copying needed since paths are the same)
            csv_file = self.config.local_data_file.replace('.zip', '.csv')
            if os.path.exists(csv_file):
                logger.info(f"Crypto data already available at {csv_file}")
            else:
                logger.error(f"Crypto data file not found at {csv_file}")
            
        except Exception as e:
            logger.exception(f"Error in extract_zip_file: {str(e)}")
            raise e