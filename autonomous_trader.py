#!/usr/bin/env python
"""
MemeStrike Bot - Fully Autonomous Trading Bot
This script executes REAL trades with ACTUAL MONEY on Solana blockchain
Fully autonomous with transaction signing and execution
"""
import os
import sys
import time
import json
import logging
import requests
import datetime
import random
import base64
from typing import Dict, Any, Optional, List, Union, Tuple

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

# Check for required Solana SDK dependencies
try:
    import base58
    from solana.rpc.api import Client
    from solana.transaction import Transaction
    
    # Different versions of solana have different import paths
    # Try multiple import paths for Keypair
    try:
        from solana.keypair import Keypair
    except ImportError:
        try:
            from solders.keypair import Keypair
        except ImportError:
            logger.error("Could not import Keypair from either solana.keypair or solders.keypair")
            raise ImportError("Keypair module not found")
    
    # Try multiple paths for PublicKey
    try:
        from solana.publickey import PublicKey
    except ImportError:
        try:
            from solders.pubkey import Pubkey as PublicKey
        except ImportError:
            logger.error("Could not import PublicKey from either solana.publickey or solders.pubkey")
            raise ImportError("PublicKey module not found")
    
    # Try different paths for TxOpts
    try:
        from solana.rpc.types import TxOpts
    except ImportError:
        try:
            from solders.rpc.config import RpcSendTransactionConfig as TxOpts
        except ImportError:
            logger.error("Could not import TxOpts")
            # Define a simple replacement if we can't import it
            class TxOpts:
                def __init__(self, skip_preflight=False, preflight_commitment="confirmed"):
                    self.skip_preflight = skip_preflight
                    self.preflight_commitment = preflight_commitment
                    
except ImportError as e:
    logger.error(f"Missing required dependency: {str(e)}")
    logger.error("Please ensure solana and base58 packages are installed: pip install solana==0.30.2 base58==2.1.1")
    sys.exit(1)

class FullyAutonomousTrader:
    """
    Fully autonomous trading bot that executes real trades on Solana blockchain
    using Jupiter Exchange for maximum liquidity
    """
    
    def __init__(self):
        """Initialize the trading bot with wallet and configuration"""
        logger.info("Initializing Fully Autonomous MemeStrike Trading Bot")
        
        # Get private key from environment
        self.private_key_base58 = os.environ.get('WALLET_PRIVATE_KEY')
        if not self.private_key_base58:
            logger.error("WALLET_PRIVATE_KEY environment variable is required for autonomous trading")
            logger.error("Please set this environment variable with your private key in Base58 format")
            sys.exit(1)
        
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
        
        # Initialize Solana RPC endpoints with fallbacks
        self.solana_rpc_endpoints = [
            "https://api.mainnet-beta.solana.com",
            "https://solana-api.projectserum.com",
            "https://rpc.ankr.com/solana",
            "https://solana.getblock.io/mainnet/"
        ]
        self.current_rpc_index = 0
        self.solana_rpc_url = self.solana_rpc_endpoints[self.current_rpc_index]
        
        # Initialize Jupiter API endpoints with version fallbacks
        self.jupiter_api_versions = {
            "v6": "https://quote-api.jup.ag/v6",
            "v4": "https://quote-api.jup.ag/v4"
        }
        self.current_jupiter_version = "v6"
        self.jupiter_api = self.jupiter_api_versions[self.current_jupiter_version]
        
        # Initialize Solana client and wallet
        try:
            # Create Solana client
            self.solana_client = Client(self.solana_rpc_url)
            
            # Clean up private key and initialize wallet
            self._init_wallet()
            
            # Get wallet balance
            balance = self.check_wallet_balance()
            if balance is not None:
                logger.info(f"Wallet balance: {balance:.6f} SOL")
                
                # Check if we have enough balance for trading
                if balance < self.total_amount:
                    logger.warning(f"Wallet balance ({balance:.6f} SOL) is less than configured amount ({self.total_amount:.6f} SOL)")
                    logger.warning("Adjusting trading amount to 90% of available balance")
                    self.total_amount = balance * 0.9
                    self.lot_amount = self.total_amount / self.num_lots
                    logger.info(f"New trading amount: {self.total_amount:.6f} SOL ({self.lot_amount:.6f} SOL per lot)")
            else:
                logger.warning("Could not check wallet balance. Will retry later.")
                
        except Exception as e:
            logger.error(f"Error initializing Solana client and wallet: {str(e)}")
            raise ValueError(f"Failed to initialize Solana client: {str(e)}")
        
        # Token list - only tokens verified working
        self.tokens = [
            "BONK", "WIF"  # Only using tokens that are working reliably
        ]
        
        # Token mint addresses (for actual blockchain transactions)
        self.token_addresses = {
            "BONK": "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
            "WIF": "EKpQGSJtjMFqKZ9KQanSqYXRcF8fBopzLHYxdM65zcjm",
            "SOL": "So11111111111111111111111111111111111111112"
        }
        
        # Stats
        self.trades_executed = 0
        self.successful_trades = 0
        self.failed_trades = 0
        self.total_spent = 0.0
        self.start_time = datetime.datetime.now()
        
        # Create trades history file if it doesn't exist
        if not os.path.exists("trades_history.json"):
            with open("trades_history.json", 'w') as f:
                json.dump([], f)
            logger.info("Created empty trades history file")
            
        logger.info("Trading bot initialization complete")
    
    def _init_wallet(self):
        """Initialize Solana wallet from private key"""
        try:
            # Check if we have a private key
            if not self.private_key_base58:
                logger.error("No wallet private key provided. Cannot initialize wallet.")
                logger.error("Please set WALLET_PRIVATE_KEY environment variable with your Base58 private key")
                sys.exit(1)
                
            # First, try to validate and clean up the private key in case it was copied with extra characters
            self.private_key_base58 = self.private_key_base58.strip()
            
            # Check if key is correct base58 format and clean if needed
            if not all(c in "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz" for c in self.private_key_base58):
                logger.warning("Private key contains invalid characters for base58 encoding")
                logger.warning("Attempting to clean up key...")
                # Try to clean up by removing common non-base58 characters
                self.private_key_base58 = ''.join(c for c in self.private_key_base58 
                                           if c in "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz")
                logger.info("Cleaned up private key for processing")
            
            # Decode the private key to bytes
            try:
                private_key_bytes = base58.b58decode(self.private_key_base58)
                logger.info(f"Successfully decoded private key (length: {len(private_key_bytes)} bytes)")
            except Exception as e:
                logger.error(f"Failed to decode private key: {str(e)}")
                logger.error("Please check that your private key is in valid Base58 format")
                sys.exit(1)
            
            # Create Solana keypair from private key bytes
            try:
                if len(private_key_bytes) == 64:
                    # This is a full keypair format (public key + private key)
                    self.keypair = Keypair.from_secret_key(private_key_bytes[:32])
                    logger.info("Created Solana keypair from full keypair format (64 bytes)")
                elif len(private_key_bytes) == 32:
                    # This is just a private key
                    self.keypair = Keypair.from_secret_key(private_key_bytes)
                    logger.info("Created Solana keypair from private key format (32 bytes)")
                else:
                    logger.error(f"Invalid private key length: {len(private_key_bytes)} bytes. Expected 32 or 64 bytes.")
                    sys.exit(1)
                
                # Get wallet public key
                self.wallet_address = str(self.keypair.public_key)
                logger.info(f"Wallet initialized: {self.wallet_address}")
            except Exception as kp_error:
                logger.error(f"Failed to create keypair: {str(kp_error)}")
                logger.error("This may be due to incompatible Solana library versions")
                logger.error("Trying alternative methods...")
                
                # Emergency fallback: Just use a dummy wallet address for now
                # Real trades won't work, but the bot will still generate links as fallback
                from uuid import uuid4
                dummy_id = str(uuid4())[:32]
                self.wallet_address = f"DummyWallet{dummy_id}"
                logger.warning(f"Using dummy wallet address: {self.wallet_address}")
                logger.warning("IMPORTANT: Autonomous trading will not work, but fallback links will be generated")
                
                # Create a flag to indicate we're in fallback mode
                self.autonomous_mode = False
                
                return
            
            # If we got here, we have a valid wallet
            self.autonomous_mode = True
            
        except Exception as e:
            logger.error(f"Error initializing wallet: {str(e)}")
            # Don't terminate on wallet init failure
            logger.error("Will continue in fallback mode (link generation only)")
            self.autonomous_mode = False
    
    def _fallback_rpc(self):
        """
        Rotate to a different RPC endpoint if the current one fails
        """
        self.current_rpc_index = (self.current_rpc_index + 1) % len(self.solana_rpc_endpoints)
        self.solana_rpc_url = self.solana_rpc_endpoints[self.current_rpc_index]
        self.solana_client = Client(self.solana_rpc_url)
        logger.info(f"Switched to alternative RPC endpoint: {self.solana_rpc_url}")
    
    def _fallback_jupiter_api(self):
        """
        Switch to an alternative Jupiter API version if the current one fails
        """
        if self.current_jupiter_version == "v6":
            self.current_jupiter_version = "v4"
        else:
            self.current_jupiter_version = "v6"
            
        self.jupiter_api = self.jupiter_api_versions[self.current_jupiter_version]
        logger.info(f"Switched to Jupiter API {self.current_jupiter_version}: {self.jupiter_api}")
    
    def check_wallet_balance(self) -> Optional[float]:
        """
        Check wallet balance in SOL
        
        Returns:
            Float balance in SOL or None if error
        """
        errors = 0
        max_retries = 3
        
        while errors < max_retries:
            try:
                # Use Solana client to get balance
                response = self.solana_client.get_balance(self.wallet_address)
                
                if "result" in response and "value" in response["result"]:
                    # Convert lamports to SOL (1 SOL = 1,000,000,000 lamports)
                    balance = int(response["result"]["value"]) / 1000000000.0
                    return balance
                
                logger.error(f"Failed to get wallet balance: {response}")
                errors += 1
                
                # Try a different RPC endpoint
                self._fallback_rpc()
                
            except Exception as e:
                logger.error(f"Error checking wallet balance: {str(e)}")
                errors += 1
                
                # Try a different RPC endpoint
                self._fallback_rpc()
        
        return None
    
    def execute_trade(self, token_symbol: str) -> Dict[str, Any]:
        """
        Execute a REAL trade with actual money using Jupiter API and Solana SDK
        
        Args:
            token_symbol: Token symbol to trade (e.g., "BONK")
            
        Returns:
            Dict with trade result
        """
        logger.info(f"🔴 EXECUTING REAL TRADE for {token_symbol} with {self.amount:.6f} SOL")
        
        # Check if we're in fallback mode
        if not hasattr(self, 'autonomous_mode') or not self.autonomous_mode:
            logger.warning("Running in fallback mode (autonomous trading disabled)")
            # Generate fallback link
            direct_url = f"https://jup.ag/swap/SOL-{token_symbol}?amount={self.amount}&slippage=2.5"
            logger.info(f"Generated fallback link: {direct_url}")
            return {
                "success": False,
                "error": "Autonomous trading disabled due to import issues",
                "fallback_link": direct_url
            }
            
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
                    # Try alternative Jupiter API version
                    self._fallback_jupiter_api()
                    return {"success": False, "error": f"Quote failed: {quote_response.status_code}"}
                
                quote_data = quote_response.json()
                
            except Exception as e:
                logger.error(f"Error getting quote: {str(e)}")
                # Try alternative Jupiter API version
                self._fallback_jupiter_api()
                return {"success": False, "error": f"Quote error: {str(e)}"}
                
            # Step 2: Get swap transaction from Jupiter
            logger.info("Getting swap transaction from Jupiter Exchange")
            try:
                swap_response = requests.post(
                    f"{self.jupiter_api}/swap",
                    json={
                        "quoteResponse": quote_data,
                        "userPublicKey": self.wallet_address,
                        "wrapUnwrapSOL": True
                    },
                    timeout=30
                )
                
                if swap_response.status_code != 200:
                    logger.error(f"Failed to get swap transaction: {swap_response.text}")
                    # Generate fallback link if swap fails
                    fallback_url = f"https://jup.ag/swap/SOL-{token_symbol}?amount={self.amount}&slippage=2.5"
                    logger.info(f"Generated fallback link: {fallback_url}")
                    return {
                        "success": False, 
                        "error": f"Swap transaction failed: {swap_response.status_code}",
                        "fallback_link": fallback_url
                    }
                    
                swap_data = swap_response.json()
                
                # Check if the swap transaction data is valid
                if "swapTransaction" not in swap_data:
                    logger.error("No swap transaction available in response")
                    logger.error(f"Response data: {swap_data}")
                    # Try alternative Jupiter API version
                    self._fallback_jupiter_api()
                    return {"success": False, "error": "No swap transaction in response"}
                
                # Get the base64-encoded transaction
                swap_transaction_b64 = swap_data["swapTransaction"]
                
            except Exception as e:
                logger.error(f"Error getting swap transaction: {str(e)}")
                # Try alternative Jupiter API version
                self._fallback_jupiter_api()
                return {"success": False, "error": f"Swap error: {str(e)}"}
            
            # Step 3: Sign and send the transaction
            logger.info("Signing and executing transaction")
            try:
                # Decode base64 transaction
                decoded_transaction = base64.b64decode(swap_transaction_b64)
                
                # Create a Transaction object from the decoded data
                transaction = Transaction.deserialize(decoded_transaction)
                
                # Sign the transaction with our keypair
                transaction.sign([self.keypair])
                
                # Send the signed transaction
                tx_opts = TxOpts(skip_preflight=True)
                tx_sig = self.solana_client.send_transaction(transaction, self.keypair, opts=tx_opts)
                
                if "result" in tx_sig:
                    signature = tx_sig["result"]
                    logger.info(f"Transaction sent with signature: {signature}")
                    
                    # Wait for confirmation
                    logger.info("Waiting for transaction confirmation...")
                    confirmed = self._confirm_transaction(signature)
                    
                    if confirmed:
                        logger.info(f"🎉 Transaction confirmed: {signature}")
                        self.trades_executed += 1
                        self.successful_trades += 1
                        self.total_spent += self.amount
                        
                        # Record trade data
                        trade_data = {
                            "token": token_symbol,
                            "amount_sol": self.amount,
                            "tx_signature": signature,
                            "time": datetime.datetime.now().isoformat(),
                            "status": "success"
                        }
                        self._record_trade_to_file(trade_data)
                        
                        return {
                            "success": True,
                            "token": token_symbol,
                            "amount": self.amount,
                            "signature": signature,
                            "timestamp": datetime.datetime.now().isoformat()
                        }
                    else:
                        logger.error(f"Transaction failed to confirm: {signature}")
                        self.trades_executed += 1
                        self.failed_trades += 1
                        
                        # Record trade data
                        trade_data = {
                            "token": token_symbol,
                            "amount_sol": self.amount,
                            "tx_signature": signature,
                            "time": datetime.datetime.now().isoformat(),
                            "status": "failed",
                            "error": "Transaction not confirmed"
                        }
                        self._record_trade_to_file(trade_data)
                        
                        return {
                            "success": False,
                            "error": "Transaction not confirmed",
                            "signature": signature
                        }
                else:
                    logger.error(f"Failed to send transaction: {tx_sig}")
                    self.trades_executed += 1
                    self.failed_trades += 1
                    
                    # Generate fallback link if transaction submission fails
                    fallback_url = f"https://jup.ag/swap/SOL-{token_symbol}?amount={self.amount}&slippage=2.5"
                    logger.info(f"Generated fallback link: {fallback_url}")
                    
                    return {
                        "success": False,
                        "error": "Failed to send transaction",
                        "fallback_link": fallback_url
                    }
                    
            except Exception as e:
                logger.error(f"Error signing and sending transaction: {str(e)}")
                self.trades_executed += 1
                self.failed_trades += 1
                
                # Generate fallback link if transaction signing fails
                fallback_url = f"https://jup.ag/swap/SOL-{token_symbol}?amount={self.amount}&slippage=2.5"
                logger.info(f"Generated fallback link: {fallback_url}")
                
                return {
                    "success": False,
                    "error": f"Transaction error: {str(e)}",
                    "fallback_link": fallback_url
                }
                
        except Exception as e:
            logger.error(f"Unexpected error executing trade: {str(e)}")
            
            # Generate fallback link for any unexpected errors
            fallback_url = f"https://jup.ag/swap/SOL-{token_symbol}?amount={self.amount}&slippage=2.5"
            logger.info(f"Generated fallback link: {fallback_url}")
            
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "fallback_link": fallback_url
            }
    
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
                resp = self.solana_client.get_signature_statuses([signature], search_transaction_history=True)
                
                if "result" in resp and resp["result"] and resp["result"]["value"] and resp["result"]["value"][0]:
                    confirmation = resp["result"]["value"][0]
                    
                    if confirmation.get("confirmationStatus") in ["confirmed", "finalized"]:
                        return True
                        
                    if "err" in confirmation and confirmation["err"] is not None:
                        logger.error(f"Transaction error: {confirmation['err']}")
                        return False
                        
                logger.info(f"Waiting for confirmation... ({i+1}/{max_retries})")
                time.sleep(retry_delay)
                
            except Exception as e:
                logger.error(f"Error checking transaction status: {str(e)}")
                # Try a different RPC endpoint
                self._fallback_rpc()
                time.sleep(retry_delay)
        
        logger.error(f"Transaction confirmation timed out after {max_retries} attempts")
        return False
    
    def _record_trade_to_file(self, trade_data: Dict[str, Any]) -> None:
        """
        Record trade data to a file for tracking
        
        Args:
            trade_data: Trade data to record
        """
        try:
            # Load existing trades
            trades = []
            if os.path.exists("trades_history.json"):
                with open("trades_history.json", 'r') as f:
                    try:
                        trades = json.load(f)
                    except json.JSONDecodeError:
                        logger.error("Error loading trades history file. Creating new one.")
                        trades = []
            
            # Add new trade
            trades.append(trade_data)
            
            # Save updated trades
            with open("trades_history.json", 'w') as f:
                json.dump(trades, f, indent=2)
                
            # Also save to daily file for backup
            today = datetime.datetime.now().strftime("%Y%m%d")
            daily_file = f"trade_links_{today}.txt"
            
            with open(daily_file, 'a') as f:
                status = "✅" if trade_data.get("status") == "success" else "❌"
                token = trade_data.get("token", "Unknown")
                amount = trade_data.get("amount_sol", 0)
                signature = trade_data.get("tx_signature", "No signature")
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                
                f.write(f"{status} {timestamp} - {token} - {amount:.6f} SOL - {signature}\n")
                
                # Add fallback link if available
                if "fallback_link" in trade_data:
                    f.write(f"    Link: {trade_data['fallback_link']}\n")
                    
        except Exception as e:
            logger.error(f"Error recording trade data: {str(e)}")
    
    def get_trading_stats(self) -> Dict[str, Any]:
        """
        Get current trading statistics
        
        Returns:
            Dict with trading stats
        """
        now = datetime.datetime.now()
        runtime = now - self.start_time
        runtime_hours = runtime.total_seconds() / 3600
        
        # Avoid division by zero
        trades_per_hour = 0
        if runtime_hours > 0:
            trades_per_hour = self.trades_executed / runtime_hours
            
        success_rate = 0
        if self.trades_executed > 0:
            success_rate = (self.successful_trades / self.trades_executed) * 100
            
        return {
            "trades_executed": self.trades_executed,
            "successful_trades": self.successful_trades,
            "failed_trades": self.failed_trades,
            "success_rate": success_rate,
            "total_spent": self.total_spent,
            "runtime_hours": runtime_hours,
            "trades_per_hour": trades_per_hour,
            "trading_lots": self.num_lots,
            "lot_size": self.lot_amount,
            "active_lots": sum(1 for lot_time in self.lots if lot_time > 0),
            "start_time": self.start_time.isoformat(),
            "autonomous_mode": hasattr(self, 'autonomous_mode') and self.autonomous_mode
        }
    
    def print_trading_stats(self) -> None:
        """Print current trading statistics"""
        stats = self.get_trading_stats()
        
        logger.info("=== MemeStrike Trading Bot Stats ===")
        logger.info(f"Runtime: {stats['runtime_hours']:.2f} hours")
        logger.info(f"Trades Executed: {stats['trades_executed']}")
        logger.info(f"Success Rate: {stats['success_rate']:.2f}%")
        logger.info(f"Trading Lots: {stats['trading_lots']} (Lot Size: {stats['lot_size']:.6f} SOL)")
        logger.info(f"Active Lots: {stats['active_lots']}")
        logger.info(f"Autonomous Mode: {'✅' if stats['autonomous_mode'] else '❌'}")
        logger.info("====================================")
    
    def run_trading_bot(self) -> None:
        """
        Main trading loop executing REAL trades with actual money across 6 lots
        Runs indefinitely until interrupted
        """
        logger.info("🚀 Starting FULLY AUTONOMOUS MemeStrike Trading Bot")
        logger.info("🔴 WARNING: This bot will execute REAL blockchain transactions with REAL MONEY")
        logger.info(f"💰 Trading with wallet: {self.wallet_address}")
        logger.info(f"💼 Trading strategy: {self.num_lots} lots with {self.lot_amount:.6f} SOL each")
        
        # Check if we're in fallback mode
        if not hasattr(self, 'autonomous_mode') or not self.autonomous_mode:
            logger.warning("⚠️ RUNNING IN FALLBACK MODE - Direct transaction execution disabled")
            logger.warning("Trade links will be generated instead of executing trades directly")
            logger.warning("To enable fully autonomous mode, please check dependency installation")
            logger.warning("Required: solana==0.30.2 base58==2.1.1 with proper module structure")
        
        # High-frequency trading configuration
        trades_per_min = self.num_lots  # 6 lots = 6 trades per minute
        trades_per_day = trades_per_min * 60 * 24  # ~8,640 trades per day
        logger.info(f"Configured to execute approximately {trades_per_min} trades per minute")
        logger.info(f"Projected daily trade volume: {trades_per_day} trades")
        
        try:
            # Main infinite trading loop
            while True:
                # Get current time
                now = time.time()
                
                # Print stats every 10 minutes
                if self.trades_executed % 60 == 0 and self.trades_executed > 0:
                    self.print_trading_stats()
                
                # Check for available lot
                available_lot = -1
                for i, lot_time in enumerate(self.lots):
                    # If lot is available (0) or its busy time has expired
                    if lot_time == 0 or lot_time < now:
                        available_lot = i
                        break
                
                if available_lot >= 0:
                    # Select a token to trade (using only verified working tokens)
                    token = random.choice(self.tokens)
                    logger.info(f"Lot {available_lot+1}: Trading {token}")
                    
                    # Set amount for this lot
                    self.amount = self.lot_amount
                    
                    # Execute the trade
                    result = self.execute_trade(token)
                    
                    # Mark lot as busy for next 10 seconds
                    self.lots[available_lot] = now + 10
                    
                    # Log result
                    if result.get("success"):
                        logger.info(f"Lot {available_lot+1}: Trade successful! Signature: {result.get('signature')}")
                    else:
                        logger.warning(f"Lot {available_lot+1}: Trade failed: {result.get('error')}")
                        if "fallback_link" in result:
                            logger.info(f"Fallback link: {result.get('fallback_link')}")
                else:
                    # All lots are busy, wait a bit
                    time.sleep(1)
                    
                # Small delay between iterations to prevent hammering the API
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            logger.info("Trading bot stopped by user")
            self.print_trading_stats()
            
        except Exception as e:
            logger.error(f"Unexpected error in trading loop: {str(e)}")
            logger.exception("Exception details:")
            self.print_trading_stats()

def main():
    """Main entry point"""
    trader = FullyAutonomousTrader()
    trader.run_trading_bot()
    
if __name__ == "__main__":
    main()
