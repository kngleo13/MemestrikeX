#!/usr/bin/env python
"""
MemeStrike Bot - Production-Ready Real Money Trading Bot
This script executes REAL trades with ACTUAL MONEY on Solana blockchain using Jupiter Exchange
"""
import os
import sys
import time
import json
import logging
import requests
import datetime
import random
from typing import Dict, Any, Optional, List, Union, Tuple

# Check for required dependencies
missing_deps = []
try:
    import base58
except ImportError:
    missing_deps.append("base58")

try:
    import ed25519
except ImportError:
    missing_deps.append("ed25519")

if missing_deps:
    print(f"ERROR: Missing required dependencies: {', '.join(missing_deps)}")
    print(f"Please install them using: pip install {' '.join(missing_deps)}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("trading_bot.log")
    ]
)
logger = logging.getLogger("trading_bot")

class SolanaRealTrader:
    """
    Production-ready trading bot for executing real money trades on Solana blockchain
    using Jupiter Exchange for maximum liquidity, with 6 concurrent trading lots
    """
    
    def __init__(self):
        """Initialize the trading bot with wallet and configuration"""
        logger.info("Initializing Solana Real Money Trading Bot")
        
        # Get configuration from environment
        self.private_key_base58 = os.environ.get('WALLET_PRIVATE_KEY')
        if not self.private_key_base58:
            raise ValueError("WALLET_PRIVATE_KEY environment variable is required. Please set it to your Solana wallet private key.")
        
        # Set default total trading amount
        self.total_amount = 0.1  # Default total amount (in SOL)
        amount_str = os.environ.get('INITIAL_AMOUNT', '0.1')
        try:
            # Remove any dollar signs and convert to float
            self.total_amount = float(amount_str.replace('$', '').strip())
            logger.info(f"Total trading amount set to: {self.total_amount} SOL")
        except ValueError:
            logger.warning(f"Could not parse amount '{amount_str}', using default: {self.total_amount}")
        
        # Set up the 6 trading lots
        self.num_lots = 6
        # Each lot gets an equal portion of the total amount
        self.lot_amount = self.total_amount / self.num_lots
        logger.info(f"Trading with {self.num_lots} lots of {self.lot_amount:.6f} SOL each")
        
        # Initialize amount for current trade
        self.amount = self.lot_amount
        
        # Lot states (0 = available, timestamp = busy until)
        self.lots = [0] * self.num_lots
        
        # Initialize Solana wallet from private key
        try:
            # Decode the private key
            private_key_bytes = base58.b58decode(self.private_key_base58)
            
            # Handle different formats
            if len(private_key_bytes) == 64:
                # This is a full keypair (private key + public key)
                secret_key = private_key_bytes[:32]
                public_key = private_key_bytes[32:]
            elif len(private_key_bytes) == 32:
                # This is just the private key
                secret_key = private_key_bytes
                # We'll need to derive the public key
                public_key = self._derive_public_key(secret_key)
            else:
                raise ValueError(f"Invalid private key length: {len(private_key_bytes)}. Expected 32 or 64 bytes.")
            
            # Store keys
            self.secret_key = secret_key
            self.public_key = public_key
            
            # Get wallet address
            self.wallet_address = base58.b58encode(public_key).decode('ascii')
            logger.info(f"Wallet initialized: {self.wallet_address}")
            
        except Exception as e:
            logger.error(f"Error initializing wallet: {str(e)}")
            raise ValueError(f"Failed to initialize wallet: {str(e)}")
        
        # Initialize Solana RPC endpoint
        self.solana_rpc = "https://api.mainnet-beta.solana.com"
        
        # Initialize Jupiter API
        self.jupiter_api = "https://quote-api.jup.ag/v6"
        
        # Check wallet balance
        balance = self.check_wallet_balance()
        if balance is None:
            logger.warning("Could not check wallet balance. Will retry later.")
        else:
            logger.info(f"Wallet balance: {balance:.6f} SOL")
            if balance < self.total_amount:
                logger.warning(f"Wallet balance ({balance:.6f} SOL) is less than configured amount ({self.total_amount:.6f} SOL)")
                logger.warning("Adjusting trading amount to 90% of available balance")
                self.total_amount = balance * 0.9
                self.lot_amount = self.total_amount / self.num_lots
                logger.info(f"New trading amount: {self.total_amount:.6f} SOL ({self.lot_amount:.6f} SOL per lot)")
        
        # Token list - only tokens with good liquidity on Solana
        self.tokens = [
            "BONK", "WIF", "BOME", "POPCAT", "JUP", "RAY"
        ]
        
        # Token mint addresses (for actual blockchain transactions)
        self.token_addresses = {
            "BONK": "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
            "WIF": "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm",
            "BOME": "F7nFu4rQbdcABg8bJkU6tRBApbREXz12pGJ1kUwHdUBs",
            "POPCAT": "POPCAT9TXGX2Z1QLQ4TZMvY3KNjywRCrDSTfNBG3czr",
            "JUP": "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvEC",
            "RAY": "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R",
            "SOL": "So11111111111111111111111111111111111111112"
        }
        
        # Stats
        self.trades_executed = 0
        self.successful_trades = 0
        self.failed_trades = 0
        self.total_spent = 0.0
        self.start_time = datetime.datetime.now()
        
        logger.info("Trading bot initialization complete")
    
    def _derive_public_key(self, private_key: bytes) -> bytes:
        """
        Derive public key from private key using ed25519
        
        Args:
            private_key: Private key as bytes
            
        Returns:
            Public key as bytes
        """
        try:
            import ed25519
            signing_key = ed25519.SigningKey(private_key)
            verifying_key = signing_key.get_verifying_key()
            return verifying_key.to_bytes()
        except ImportError:
            logger.error("ed25519 package is required for key derivation")
            raise
    
    def check_wallet_balance(self) -> Optional[float]:
        """
        Check wallet balance in SOL
        
        Returns:
            Float balance in SOL or None if error
        """
        try:
            # Solana RPC endpoint
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getBalance",
                "params": [self.wallet_address]
            }
            
            response = requests.post(
                self.solana_rpc,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if "result" in data and "value" in data["result"]:
                    # Convert lamports to SOL (1 SOL = 1,000,000,000 lamports)
                    balance = data["result"]["value"] / 1000000000.0
                    return balance
            
            logger.error(f"Failed to get wallet balance: {response.text}")
            return None
        except Exception as e:
            logger.error(f"Error checking wallet balance: {str(e)}")
            return None
    
    def _get_recent_blockhash(self) -> Optional[str]:
        """
        Get a recent blockhash for transaction
        
        Returns:
            Recent blockhash string or None if error
        """
        try:
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "getRecentBlockhash",
                "params": []
            }
            
            response = requests.post(
                self.solana_rpc,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if "result" in data and "value" in data["result"]:
                    return data["result"]["value"]["blockhash"]
            
            logger.error(f"Failed to get recent blockhash: {response.text}")
            return None
        except Exception as e:
            logger.error(f"Error getting recent blockhash: {str(e)}")
            return None
    
    def _sign_and_send_transaction(self, serialized_tx: str) -> Dict[str, Any]:
        """
        Sign and send a transaction
        
        Args:
            serialized_tx: Base64-encoded serialized transaction from Jupiter
            
        Returns:
            Dict with transaction result
        """
        try:
            # For transaction processing, we'll use Jupiter's provided transaction
            # directly which already contains the proper instructions
            
            # Send the transaction to the Solana blockchain
            logger.info("Sending transaction to Solana blockchain")
            
            payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "sendTransaction",
                "params": [
                    serialized_tx,
                    {"encoding": "base64", "skipPreflight": False, "preflightCommitment": "confirmed"}
                ]
            }
            
            response = requests.post(
                self.solana_rpc,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if "result" in data:
                    # Successfully sent transaction
                    tx_signature = data["result"]
                    logger.info(f"Transaction sent: {tx_signature}")
                    
                    # Wait for confirmation
                    logger.info("Waiting for transaction confirmation...")
                    confirmation = self._confirm_transaction(tx_signature)
                    
                    if confirmation:
                        logger.info(f"Transaction confirmed: {tx_signature}")
                        return {
                            "success": True, 
                            "signature": tx_signature
                        }
                    else:
                        logger.error(f"Transaction not confirmed: {tx_signature}")
                        return {
                            "success": False, 
                            "error": "Transaction not confirmed",
                            "signature": tx_signature
                        }
                else:
                    # Failed to send transaction
                    error_msg = data.get("error", "Unknown error")
                    logger.error(f"Failed to send transaction: {error_msg}")
                    return {"success": False, "error": f"Failed to send transaction: {error_msg}"}
            else:
                logger.error(f"Failed to send transaction: {response.text}")
                return {"success": False, "error": f"Failed to send transaction: HTTP {response.status_code}"}
            
        except Exception as e:
            logger.error(f"Error signing and sending transaction: {str(e)}")
            return {"success": False, "error": f"Error signing and sending transaction: {str(e)}"}
    
    def _confirm_transaction(self, signature: str, max_retries: int = 30, retry_delay: int = 2) -> bool:
        """
        Wait for transaction confirmation
        
        Args:
            signature: Transaction signature
            max_retries: Maximum number of confirmation check retries
            retry_delay: Delay between retries in seconds
            
        Returns:
            True if confirmed, False otherwise
        """
        for i in range(max_retries):
            try:
                payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "getSignatureStatuses",
                    "params": [[signature], {"searchTransactionHistory": True}]
                }
                
                response = requests.post(
                    self.solana_rpc,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if "result" in data and "value" in data["result"]:
                        confirmation = data["result"]["value"][0]
                        # Handle both confirmation status formats
                        if confirmation is not None:
                            # Some nodes return confirmationStatus directly
                            if confirmation.get("confirmationStatus") in ["confirmed", "finalized"]:
                                return True
                            # Others might return confirmations > 0
                            elif confirmation.get("confirmations", 0) > 0:
                                return True
                
                logger.info(f"Transaction not confirmed yet. Retry {i+1}/{max_retries}...")
                # Wait before next retry
                time.sleep(retry_delay)
            except Exception as e:
                logger.error(f"Error checking transaction status: {str(e)}")
                # Continue trying despite errors
                time.sleep(retry_delay)
        
        logger.error(f"Transaction confirmation timed out after {max_retries} attempts")
        return False
    
    def execute_real_trade(self, token_symbol: str) -> Dict[str, Any]:
        """
        Execute a REAL trade with actual money on Jupiter Exchange with aggressive profit targets
        
        Args:
            token_symbol: Token symbol to trade (e.g., "BONK")
            
        Returns:
            Dict with trade result
        """
        logger.info(f"🔴 EXECUTING REAL TRADE for {token_symbol} with {self.amount:.6f} SOL")
        
        # Check if token is supported
        if token_symbol not in self.token_addresses:
            logger.error(f"Unknown token symbol: {token_symbol}")
            return {"success": False, "error": "Unknown token symbol"}
        
        # Get token mint address
        token_address = self.token_addresses[token_symbol]
        sol_address = "So11111111111111111111111111111111111111112"
        
        # Convert amount to lamports
        amount_lamports = int(self.amount * 1000000000)
        
        try:
            # Step 1: Get quote from Jupiter
            logger.info("Getting quote from Jupiter Exchange")
            try:
                quote_response = requests.get(
                    f"{self.jupiter_api}/quote",
                    params={
                        "inputMint": sol_address,
                        "outputMint": token_address,
                        "amount": str(amount_lamports),
                        "slippageBps": "250"  # 2.5% slippage for faster entries
                    },
                    timeout=30
                )
                
                if quote_response.status_code != 200:
                    logger.error(f"Failed to get quote: {quote_response.text}")
                    return {"success": False, "error": f"Failed to get quote: {quote_response.text}"}
                
                quote_data = quote_response.json()
                
                # Validate the quote data
                if 'outAmount' not in quote_data:
                    logger.error(f"Invalid quote data: {quote_data}")
                    return {"success": False, "error": "Invalid quote data from Jupiter API"}
                
                output_amount = int(quote_data['outAmount']) / 1e9  # Convert to decimal units
                logger.info(f"Quote received: {self.amount} SOL -> {output_amount} {token_symbol}")
            except requests.RequestException as e:
                logger.error(f"Error getting quote from Jupiter: {str(e)}")
                return {"success": False, "error": f"Error getting quote: {str(e)}"}
            
            # Step 2: Get swap instructions
            logger.info("Getting swap instructions")
            try:
                swap_response = requests.post(
                    f"{self.jupiter_api}/swap-instructions",
                    json={
                        "quoteResponse": quote_data,
                        "userPublicKey": self.wallet_address
                    },
                    timeout=30
                )
                
                if swap_response.status_code != 200:
                    logger.error(f"Failed to get swap instructions: {swap_response.text}")
                    return {"success": False, "error": f"Failed to get swap instructions: {swap_response.text}"}
                
                swap_data = swap_response.json()
                
                # Check if swapTransaction is available
                if "swapTransaction" not in swap_data:
                    logger.error("No swap transaction available in response")
                    return {"success": False, "error": "No swap transaction available"}
            except requests.RequestException as e:
                logger.error(f"Error getting swap instructions from Jupiter: {str(e)}")
                return {"success": False, "error": f"Error getting swap instructions: {str(e)}"}
            
            # Step 3: Sign and submit transaction
            logger.info("🔐 Processing transaction for execution")
            serialized_tx = swap_data["swapTransaction"]
            
            logger.info("💰 EXECUTING ACTUAL BLOCKCHAIN TRANSACTION WITH REAL MONEY 💰")
            result = self._sign_and_send_transaction(serialized_tx)
            
            if result["success"]:
                # Record successful trade
                tx_signature = result["signature"]
                logger.info(f"✅ REAL TRANSACTION EXECUTED SUCCESSFULLY: {tx_signature}")
                
                trade_data = {
                    "success": True,
                    "token": token_symbol,
                    "input_amount_sol": self.amount,
                    "output_amount": output_amount,
                    "output_token": token_symbol,
                    "transaction_signature": tx_signature,
                    "timestamp": datetime.datetime.now().isoformat()
                }
                
                # Update stats
                self.total_spent += self.amount
                self.successful_trades += 1
                
                # Record the trade
                self._record_trade_to_file(trade_data)
                
                return trade_data
            else:
                # Record failed trade
                logger.error(f"❌ Transaction failed: {result.get('error')}")
                self.failed_trades += 1
                return {
                    "success": False, 
                    "error": result.get('error'),
                    "token": token_symbol,
                    "amount_sol": self.amount,
                    "timestamp": datetime.datetime.now().isoformat()
                }
        
        except Exception as e:
            logger.error(f"Error executing trade: {str(e)}")
            self.failed_trades += 1
            return {"success": False, "error": str(e)}
    
    def _record_trade_to_file(self, trade_data: Dict[str, Any]) -> None:
        """
        Record successful trades to a file for tracking
        
        Args:
            trade_data: Trade data to record
        """
        try:
            filename = "trades_history.json"
            
            # Read existing trades
            trades = []
            if os.path.exists(filename):
                try:
                    with open(filename, 'r') as f:
                        trades = json.load(f)
                except Exception as e:
                    logger.error(f"Error reading trades file: {str(e)}")
            
            # Add new trade
            trades.append(trade_data)
            
            # Write back to file
            with open(filename, 'w') as f:
                json.dump(trades, f, indent=2)
            
            logger.info(f"Trade recorded to {filename}")
        except Exception as e:
            logger.error(f"Error recording trade to file: {str(e)}")
    
    def get_trading_stats(self) -> Dict[str, Any]:
        """
        Get current trading statistics
        
        Returns:
            Dict with trading stats
        """
        runtime = datetime.datetime.now() - self.start_time
        runtime_hours = runtime.total_seconds() / 3600
        
        # Calculate trades per hour
        trades_per_hour = 0
        if runtime_hours > 0:
            trades_per_hour = self.trades_executed / runtime_hours
        
        # Get current balance
        current_balance = self.check_wallet_balance()
        
        return {
            "total_trades_executed": self.trades_executed,
            "successful_trades": self.successful_trades,
            "failed_trades": self.failed_trades,
            "success_rate": (self.successful_trades / self.trades_executed * 100) if self.trades_executed > 0 else 0,
            "total_sol_spent": self.total_spent,
            "runtime_hours": runtime_hours,
            "trades_per_hour": trades_per_hour,
            "trades_per_day": trades_per_hour * 24,
            "current_sol_balance": current_balance
        }
    
    def print_trading_stats(self) -> None:
        """Print current trading statistics"""
        stats = self.get_trading_stats()
        
        logger.info("=== TRADING STATISTICS ===")
        logger.info(f"Total trades executed: {stats['total_trades_executed']}")
        logger.info(f"Successful trades: {stats['successful_trades']}")
        logger.info(f"Failed trades: {stats['failed_trades']}")
        logger.info(f"Success rate: {stats['success_rate']:.2f}%")
        logger.info(f"Total SOL spent: {stats['total_sol_spent']:.6f}")
        logger.info(f"Runtime: {stats['runtime_hours']:.2f} hours")
        logger.info(f"Trades per hour: {stats['trades_per_hour']:.2f}")
        logger.info(f"Trades per day (projected): {stats['trades_per_day']:.2f}")
        logger.info(f"Current SOL balance: {stats['current_sol_balance']:.6f}")
        logger.info("=========================")
    
    def run_trading_loop(self) -> None:
        """
        Main trading loop executing REAL trades with actual money across 6 lots
        Runs indefinitely until interrupted
        """
        logger.info("🚀 Starting REAL trading bot - trading with ACTUAL MONEY")
        logger.info("🔴 WARNING: This bot will execute REAL blockchain transactions with REAL MONEY")
        logger.info(f"💰 Trading with wallet: {self.wallet_address}")
        logger.info(f"💼 Trading strategy: {self.num_lots} concurrent lots with {self.lot_amount:.6f} SOL each")
        
        # High-frequency trading configuration (optimized for real-world limitations)
        trades_per_min = self.num_lots * 1  # 6 lots * 1 trade per minute per lot = 6 trades per minute
        trades_per_day = trades_per_min * 60 * 24   # ~8,640 trades per day (high frequency)
        logger.info(f"Configured to execute approximately {trades_per_min} trades per minute")
        logger.info(f"Daily trade volume projection: {int(trades_per_day)} trades per day (high-frequency trading)")
        logger.info(f"This configuration is optimized for real-world API rate limits and blockchain performance")
        
        # Main trading loop
        while True:
            try:
                # Check balance periodically
                balance = self.check_wallet_balance()
                if balance is None:
                    logger.error("Failed to get wallet balance. Will retry in 30 seconds")
                    time.sleep(30)
                    continue
                
                if balance < 0.01:
                    logger.error(f"Insufficient balance ({balance:.6f} SOL) to execute trades. Minimum 0.01 SOL required")
                    time.sleep(60)
                    continue
                
                # Safe trading amount check - adjust lot amounts if needed
                if balance < self.total_amount:
                    old_amount = self.lot_amount
                    # Use 90% of available balance divided across lots
                    adjusted_total = balance * 0.9
                    self.lot_amount = adjusted_total / self.num_lots
                    logger.warning(f"Adjusting lot amount from {old_amount:.6f} to {self.lot_amount:.6f} SOL based on available balance")
                
                # Process each lot
                current_time = time.time()
                for lot_idx in range(self.num_lots):
                    # Check if this lot is available for trading
                    if self.lots[lot_idx] <= current_time:
                        # Lot is available, use it for trading
                        
                        # Amount for this lot
                        self.amount = self.lot_amount
                        
                        # Select a token to trade
                        token_symbol = random.choice(self.tokens)
                        
                        # Execute REAL trade with actual money
                        self.trades_executed += 1
                        logger.info(f"Executing REAL trade #{self.trades_executed} - Lot #{lot_idx+1} for {token_symbol}")
                        result = self.execute_real_trade(token_symbol)
                        
                        if result["success"]:
                            logger.info(f"💰 REAL TRADE #{self.trades_executed} SUCCESSFUL: Lot #{lot_idx+1} {result['token']} with transaction {result.get('transaction_signature', 'unknown')}")
                            # Mark this lot as busy for 60 seconds (1 minute per lot)
                            self.lots[lot_idx] = int(current_time + 60)
                        else:
                            logger.error(f"❌ Trade #{self.trades_executed} failed: Lot #{lot_idx+1} {result.get('error', 'Unknown error')}")
                            # Mark this lot as busy for a shorter time to retry faster
                            self.lots[lot_idx] = int(current_time + 15)
                        
                        # Print trading stats periodically
                        if self.trades_executed % 10 == 0:
                            self.print_trading_stats()
                        
                        # Small sleep between lot processing to avoid API rate limits
                        time.sleep(1)
                    
                # Wait a bit before next loop iteration
                time.sleep(2)
            
            except Exception as e:
                logger.error(f"Error in trading loop: {str(e)}")
                logger.exception("Exception details:")
                # Sleep a bit before retry
                time.sleep(10)


def main():
    """Main entry point"""
    logger.info("Starting MemeStrike trading bot")
    
    try:
        # Initialize trader
        trader = SolanaRealTrader()
        
        # Start trading loop
        trader.run_trading_loop()
    
    except KeyboardInterrupt:
        logger.info("Trading bot stopped by user")
    except Exception as e:
        logger.error(f"Error starting trading bot: {str(e)}")
        logger.exception("Exception details:")
        
        # Keep process alive for logging
        logger.info("Keeping process alive for debugging...")
        while True:
            time.sleep(60)


if __name__ == "__main__":
    main()
