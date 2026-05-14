"""
Manual CIFAR-10 Dataset Downloader
This script helps download the CIFAR-10 dataset with SSL workarounds
"""

import os
import urllib.request
import ssl
import sys

def download_cifar10():
    """Download CIFAR-10 dataset with SSL fix"""
    
    # URL for CIFAR-10 dataset
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    
    # Determine the correct path for Windows
    home = os.path.expanduser('~')
    keras_dir = os.path.join(home, '.keras', 'datasets')
    
    # Create directory if it doesn't exist
    os.makedirs(keras_dir, exist_ok=True)
    
    # File path
    filepath = os.path.join(keras_dir, 'cifar-10-python.tar.gz')
    
    print("="*70)
    print("CIFAR-10 DATASET DOWNLOADER")
    print("="*70)
    print(f"\n📁 Download destination: {filepath}")
    print(f"🌐 Source URL: {url}")
    print(f"📦 File size: ~170 MB")
    
    # Check if file already exists
    if os.path.exists(filepath):
        file_size = os.path.getsize(filepath) / (1024 * 1024)  # Convert to MB
        print(f"\n✓ File already exists! ({file_size:.1f} MB)")
        
        response = input("\nDo you want to re-download? (y/n): ").strip().lower()
        if response != 'y':
            print("✓ Using existing file.")
            return True
        else:
            print("🗑️  Removing old file...")
            os.remove(filepath)
    
    print("\n" + "-"*70)
    print("DOWNLOADING...")
    print("-"*70)
    
    # Try multiple methods
    methods = [
        ("Method 1: Using SSL workaround", download_with_ssl_workaround),
        ("Method 2: Using urllib without SSL verification", download_with_urllib),
        ("Method 3: Manual instructions", show_manual_instructions)
    ]
    
    for method_name, method_func in methods:
        print(f"\n🔄 Trying {method_name}...")
        try:
            if method_func(url, filepath):
                print(f"\n✅ SUCCESS!")
                print(f"✓ Dataset downloaded to: {filepath}")
                print(f"✓ You can now run: python image_classifier_fixed.py")
                return True
        except Exception as e:
            print(f"❌ Failed: {e}")
            continue
    
    print("\n" + "="*70)
    print("⚠️  AUTOMATIC DOWNLOAD FAILED")
    print("="*70)
    print("Please download manually using the instructions above.")
    return False


def download_with_ssl_workaround(url, filepath):
    """Download using SSL workaround"""
    # Create unverified SSL context
    ssl._create_default_https_context = ssl._create_unverified_context
    
    # Download with progress
    def reporthook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\r  Progress: {percent}% ")
        sys.stdout.flush()
    
    urllib.request.urlretrieve(url, filepath, reporthook)
    print()  # New line after progress
    return True


def download_with_urllib(url, filepath):
    """Download using urllib with custom context"""
    context = ssl._create_unverified_context()
    
    with urllib.request.urlopen(url, context=context) as response:
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        downloaded = 0
        
        with open(filepath, 'wb') as f:
            while True:
                chunk = response.read(block_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                percent = int(downloaded * 100 / total_size)
                sys.stdout.write(f"\r  Progress: {percent}% ")
                sys.stdout.flush()
    
    print()  # New line after progress
    return True


def show_manual_instructions(url, filepath):
    """Show manual download instructions"""
    print("\n" + "="*70)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("="*70)
    print("\n1. Open your web browser")
    print(f"2. Go to: {url}")
    print("3. Download the file (cifar-10-python.tar.gz)")
    print(f"4. Move it to: {filepath}")
    print("\n5. Then run: python image_classifier_fixed.py")
    print("="*70)
    
    input("\nPress Enter after you've downloaded the file manually...")
    
    # Check if user downloaded it
    if os.path.exists(filepath):
        print("✓ File found! You're ready to go!")
        return True
    else:
        print("❌ File not found. Please try again.")
        return False


if __name__ == "__main__":
    try:
        success = download_cifar10()
        if success:
            print("\n" + "="*70)
            print("✨ NEXT STEPS")
            print("="*70)
            print("1. Run the image classifier:")
            print("   python image_classifier_fixed.py")
            print("\n2. Wait for model training (10-20 minutes)")
            print("\n3. Start classifying your images!")
            print("="*70)
        else:
            print("\n❌ Download failed. Please try the manual method.")
    except KeyboardInterrupt:
        print("\n\n⚠️  Download cancelled by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()