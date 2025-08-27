#!/usr/bin/env python3
"""
OpenBB Test Script for Quantitative Finance
Test basic functionality of OpenBB platform
"""

try:
    from openbb import obb
    import pandas as pd
    import numpy as np
    
    print("✅ OpenBB Platform imported successfully!")
    
    # Test 1: Get stock data
    print("\n📈 Testing stock data retrieval...")
    # Get Apple stock data for the last 30 days
    data = obb.equity.price.historical("AAPL", period="1m")
    df = data.to_df()  # Convert to pandas DataFrame
    print(f"✅ Retrieved {len(df)} days of AAPL data")
    print(f"📊 Latest close price: ${df['close'].iloc[-1]:.2f}")
    
    # Test 2: Economic data
    print("\n📊 Testing economic data...")
    # Get GDP data (if available)
    try:
        gdp_data = obb.economy.gdp()
        print("✅ Economic data retrieved successfully")
    except Exception as e:
        print(f"⚠️  Economic data test skipped: {e}")
    
    # Test 3: Market news
    print("\n📰 Testing news data...")
    try:
        news = obb.news.company("AAPL", limit=5)
        news_df = news.to_df()
        print(f"✅ Retrieved {len(news_df)} news articles")
    except Exception as e:
        print(f"⚠️  News test skipped: {e}")
    
    print("\n🎉 OpenBB Platform is ready for quantitative analysis!")
    print("\n💡 Next steps:")
    print("   • Activate environment: conda activate quant-env")
    print("   • Start Jupyter: jupyter lab")
    print("   • Import: from openbb import obb")
    
except ImportError as e:
    print(f"❌ Error importing OpenBB: {e}")
    print("Please make sure you're in the quant-env environment")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
