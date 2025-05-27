import uvicorn
from .config.settings import get_settings

def main():
    """Main entry point for the application"""
    settings = get_settings()
    uvicorn.run(
        "src.nutritionist.api.app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    )

if __name__ == "__main__":
    main() 