# app/config/settings.py

import os
from pathlib import Path
from typing import Optional


class Settings:
    """
    Central configuration for the AI system with comprehensive API key management
    """

    def __init__(self):
        # -----------------------
        # ENVIRONMENT
        # -----------------------
        self.ENV = os.getenv("ENV", "development")
        self.DEBUG = self._get_bool("DEBUG", self.ENV == "development")

        # -----------------------
        # LLM API KEYS (REQUIRED)
        # -----------------------
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.COHERE_API_KEY = os.getenv("COHERE_API_KEY")

        # -----------------------
        # MARKETING API KEYS (OPTIONAL but recommended)
        # -----------------------
        # Email
        self.SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")
        self.MAILCHIMP_API_KEY = os.getenv("MAILCHIMP_API_KEY")
        
        # Social Media
        self.FACEBOOK_ACCESS_TOKEN = os.getenv("FACEBOOK_ACCESS_TOKEN")
        self.FACEBOOK_PIXEL_ID = os.getenv("FACEBOOK_PIXEL_ID")
        self.FACEBOOK_BUSINESS_ACCOUNT_ID = os.getenv("FACEBOOK_BUSINESS_ACCOUNT_ID")
        
        self.INSTAGRAM_ACCESS_TOKEN = os.getenv("INSTAGRAM_ACCESS_TOKEN")
        self.INSTAGRAM_BUSINESS_ACCOUNT_ID = os.getenv("INSTAGRAM_BUSINESS_ACCOUNT_ID")
        
        self.TIKTOK_ACCESS_TOKEN = os.getenv("TIKTOK_ACCESS_TOKEN")
        self.TIKTOK_BUSINESS_ID = os.getenv("TIKTOK_BUSINESS_ID")
        
        self.WHATSAPP_BUSINESS_ACCOUNT_ID = os.getenv("WHATSAPP_BUSINESS_ACCOUNT_ID")
        self.WHATSAPP_ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN")
        self.WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")

        # -----------------------
        # PAYMENT API KEYS (OPTIONAL)
        # -----------------------
        self.STRIPE_API_KEY = os.getenv("STRIPE_API_KEY")
        self.STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")
        
        self.PAYPAL_CLIENT_ID = os.getenv("PAYPAL_CLIENT_ID")
        self.PAYPAL_CLIENT_SECRET = os.getenv("PAYPAL_CLIENT_SECRET")

        # -----------------------
        # PUBLISHING PLATFORM KEYS (OPTIONAL)
        # -----------------------
        self.GUMROAD_API_KEY = os.getenv("GUMROAD_API_KEY")
        self.SELAR_API_KEY = os.getenv("SELAR_API_KEY")

        # -----------------------
        # LLM CONFIG
        # -----------------------
        self.LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # switch here
        self.MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
        self.TEMPERATURE = float(os.getenv("TEMPERATURE", 0.7))
        self.MAX_TOKENS = int(os.getenv("MAX_TOKENS", 300))

        # -----------------------
        # MARKET & LOCALIZATION
        # -----------------------
        self.MARKET_REGION = os.getenv("MARKET_REGION", "NG")  # Primary market
        
        # -----------------------
        # FEATURE FLAGS (Agent Controls)
        # -----------------------
        # Core localization for market
        self.ENABLE_LOCALIZATION = self._get_bool("ENABLE_LOCALIZATION", True)
        
        # Optional agents (toggleable)
        self.ENABLE_RECOMMENDATION = self._get_bool("ENABLE_RECOMMENDATION", False)
        self.ENABLE_ORIGINALITY_CHECK = self._get_bool("ENABLE_ORIGINALITY_CHECK", False)
        self.ENABLE_ADV_MARKETING = self._get_bool("ENABLE_ADV_MARKETING", False)
        self.ENABLE_PAYMENT_AUTOMATION = self._get_bool("ENABLE_PAYMENT_AUTOMATION", False)
        
        # System features
        self.ENABLE_MEMORY = self._get_bool("ENABLE_MEMORY", True)
        self.ENABLE_FEEDBACK_LOOP = self._get_bool("ENABLE_FEEDBACK_LOOP", True)
        self.ENABLE_A_B_TESTING = self._get_bool("ENABLE_A_B_TESTING", False)
        self.ENABLE_PAYMENT_PROCESSING = self._get_bool("ENABLE_PAYMENT_PROCESSING", False)

        # -----------------------
        # PATHS
        # -----------------------
        base_dir = Path(__file__).resolve().parent.parent
        self.DATA_DIR = base_dir / "core" / "memory"
        self.PRODUCT_DB = self.DATA_DIR / "products.db"
        self.CAMPAIGN_DB = self.DATA_DIR / "campaigns.db"
        self.OUTPUTS_DIR = base_dir.parent / "outputs"
        self.LOGS_DIR = base_dir.parent / "logs"

        # -----------------------
        # SERVER
        # -----------------------
        # API host/port used by launch scripts
        self.API_PORT = int(os.getenv("API_PORT", 8000))
        self.API_HOST = os.getenv("API_HOST", "0.0.0.0")

        # Ensure directories exist
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        self.LOGS_DIR.mkdir(parents=True, exist_ok=True)

        # -----------------------
        # VALIDATION
        # -----------------------
        self._validate()

    # -----------------------
    # HELPERS
    # -----------------------
    def _get_bool(self, key: str, default: bool) -> bool:
        """Convert environment variable to boolean"""
        return os.getenv(key, str(default)).lower() in ("true", "1", "yes")

    def _validate(self) -> None:
        """Validate required API keys"""
        # LLM keys are required
        if self.LLM_PROVIDER == "openai" and not self.OPENAI_API_KEY:
            raise ValueError("❌ OPENAI_API_KEY is missing in .env")

        if self.LLM_PROVIDER == "cohere" and not self.COHERE_API_KEY:
            raise ValueError("❌ COHERE_API_KEY is missing in .env")

    def get_missing_api_keys(self) -> dict:
        """
        Identify which optional API keys are missing.
        Useful for debugging and informing users what features won't work.
        """
        missing = {}
        
        # Email services
        if not self.SENDGRID_API_KEY and not self.MAILCHIMP_API_KEY:
            missing["email"] = "No email service configured (SendGrid or Mailchimp)"
        
        # Social media
        if not self.FACEBOOK_ACCESS_TOKEN:
            missing["facebook"] = "Facebook API not configured"
        if not self.INSTAGRAM_ACCESS_TOKEN:
            missing["instagram"] = "Instagram API not configured"
        if not self.TIKTOK_ACCESS_TOKEN:
            missing["tiktok"] = "TikTok API not configured"
        if not self.WHATSAPP_ACCESS_TOKEN:
            missing["whatsapp"] = "WhatsApp API not configured"
        
        # Payment processing
        if not self.STRIPE_API_KEY and not self.PAYPAL_CLIENT_ID:
            missing["payment"] = "No payment processor configured (Stripe or PayPal)"
        
        return missing

    def print_config_status(self) -> None:
        """Print current configuration status for debugging"""
        print("\n" + "="*60)
        print("🔧 CONFIGURATION STATUS")
        print("="*60)
        print(f"Environment: {self.ENV}")
        print(f"LLM Provider: {self.LLM_PROVIDER}")
        print(f"Model: {self.MODEL_NAME}")
        print(f"Debug Mode: {self.DEBUG}")
        print("\n✅ REQUIRED:")
        print(f"  OpenAI Key: {'✓' if self.OPENAI_API_KEY else '✗ MISSING'}")
        if self.LLM_PROVIDER == "cohere":
            print(f"  Cohere Key: {'✓' if self.COHERE_API_KEY else '✗ MISSING'}")
        
        print("\n📧 EMAIL SERVICES:")
        print(f"  SendGrid: {'✓' if self.SENDGRID_API_KEY else '✗'}")
        print(f"  Mailchimp: {'✓' if self.MAILCHIMP_API_KEY else '✗'}")
        
        print("\n📱 SOCIAL MEDIA:")
        print(f"  Facebook: {'✓' if self.FACEBOOK_ACCESS_TOKEN else '✗'}")
        print(f"  Instagram: {'✓' if self.INSTAGRAM_ACCESS_TOKEN else '✗'}")
        print(f"  TikTok: {'✓' if self.TIKTOK_ACCESS_TOKEN else '✗'}")
        print(f"  WhatsApp: {'✓' if self.WHATSAPP_ACCESS_TOKEN else '✗'}")
        
        print("\n💰 PAYMENTS:")
        print(f"  Stripe: {'✓' if self.STRIPE_API_KEY else '✗'}")
        print(f"  PayPal: {'✓' if self.PAYPAL_CLIENT_ID else '✗'}")
        print(f"  API Host: {self.API_HOST}")
        print(f"  API Port: {self.API_PORT}")
        
        print("\n🎛️  FEATURES:")
        print(f"  Memory: {'✓' if self.ENABLE_MEMORY else '✗'}")
        print(f"  A/B Testing: {'✓' if self.ENABLE_A_B_TESTING else '✗'}")
        print(f"  Payment Processing: {'✓' if self.ENABLE_PAYMENT_PROCESSING else '✗'}")
        print("="*60 + "\n")


# Singleton
settings = Settings()