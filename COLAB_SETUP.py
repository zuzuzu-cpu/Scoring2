# ========================================================================
# 2026 FIFA WORLD CUP MONTE CARLO - GOOGLE COLAB QUICK START
# ========================================================================
# Run this cell first in Google Colab to set up everything automatically

print("=" * 85)
print("  🏆 2026 FIFA WORLD CUP MONTE CARLO - COLAB SETUP")
print("=" * 85)
print()

# Step 1: Install all dependencies
print("📦 Installing required packages...")
import subprocess
import sys

packages = ['numpy', 'pandas', 'matplotlib', 'seaborn', 'scipy', 'requests']

for package in packages:
    try:
        __import__(package)
        print(f"  ✓ {package} already installed")
    except ImportError:
        print(f"  ⏳ Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
        print(f"  ✓ {package} installed successfully")

print("\n✅ All dependencies installed!\n")

# Step 2: Download and execute the main simulation script
print("=" * 85)
print("  DOWNLOADING SIMULATION ENGINE")
print("=" * 85)
print()

import requests

print("📥 Fetching 2026_FIFA_World_Cup_Monte_Carlo_Simulation.py...")

try:
    url = 'https://raw.githubusercontent.com/zuzuzu-cpu/Scoring2/main/2026_FIFA_World_Cup_Monte_Carlo_Simulation.py'
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    
    print("✅ Script downloaded successfully!\n")
    
    # Execute the main script
    print("=" * 85)
    print("  LAUNCHING SIMULATION ENGINE")
    print("=" * 85)
    print()
    
    exec(response.text)
    
except Exception as e:
    print(f"\n❌ Error downloading script: {e}")
    print("\nFallback: Running simulation from local file...")
    print("\nPlease upload '2026_FIFA_World_Cup_Monte_Carlo_Simulation.py' or")
    print("run the script directly from your repository.")
