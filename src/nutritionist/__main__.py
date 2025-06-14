import uvicorn
import os
from .config.settings import get_settings

def main():
    """Main entry point for the application"""
    settings = get_settings()
    port = int(os.environ.get("PORT", settings.PORT))
    uvicorn.run(
        "src.nutritionist.api.app:app",
        host="0.0.0.0",  # Bind to all interfaces
        port=port,
        workers=1,  # Add worker configuration
        reload=False  # Disable reload in production
    )

if __name__ == "__main__":
    main() 