#!/usr/bin/env python3
"""
Simple Endpoint Validator for ANE Studio API

This script validates that all Ollama-compatible endpoints are working.

Usage:
    python tests/test_endpoints.py
    python tests/test_endpoints.py --network
"""

import argparse
import json
import socket
import sys

try:
    import requests
except ImportError:
    print("Error: requests library not installed")
    print("Install with: pip3 install requests")
    sys.exit(1)


def get_local_ip():
    """Get local network IP"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return None


def test_endpoint(base_url, method, endpoint, description, payload=None, headers=None):
    """Test a single endpoint"""
    url = f"{base_url}{endpoint}"
    default_headers = {"Accept": "application/json"}
    if headers:
        default_headers.update(headers)

    try:
        if method == "GET":
            response = requests.get(url, headers=default_headers, timeout=5)
        elif method == "POST":
            response = requests.post(
                url, json=payload, headers=default_headers, timeout=5
            )
        elif method == "OPTIONS":
            response = requests.options(url, headers=default_headers, timeout=5)

        success = 200 <= response.status_code < 300
        status = "✓" if success else "✗"
        print(f"  {status} {description}: HTTP {response.status_code}")

        if success and "application/json" in response.headers.get("content-type", ""):
            try:
                data = response.json()
                return success, data
            except:
                return success, None
        return success, None

    except requests.exceptions.ConnectionError:
        print(f"  ✗ {description}: Connection refused")
        return False, None
    except requests.exceptions.Timeout:
        print(f"  ✗ {description}: Timeout")
        return False, None
    except Exception as e:
        print(f"  ✗ {description}: {str(e)[:50]}")
        return False, None


def run_tests(base_url):
    """Run all endpoint tests"""
    print(f"\nTesting server at: {base_url}")
    print("=" * 60)

    tests_passed = 0
    tests_failed = 0

    # Test 1: Root with API client header
    print("\n1. Root Endpoint (API Client)")
    success, data = test_endpoint(
        base_url,
        "GET",
        "/",
        "Root with API header",
        headers={"User-Agent": "OllamaClient/1.0"},
    )
    if success and data and isinstance(data, dict):
        print(f"     Server: {data.get('name', 'Unknown')}")
        print(f"     Version: {data.get('version', 'Unknown')}")
        print(f"     Ollama Compatible: {data.get('ollama_compatible', False)}")
        tests_passed += 1
    else:
        # Try without headers (browser mode)
        success2, _ = test_endpoint(base_url, "GET", "/", "Root HTML fallback")
        if success2:
            print("     (Returns HTML for browsers - this is OK)")
            tests_passed += 1
        else:
            tests_failed += 1

    # Test 2: Version
    print("\n2. Version Endpoint")
    success, data = test_endpoint(base_url, "GET", "/api/version", "/api/version")
    if success and data and "version" in data:
        print(f"     Version: {data['version']}")
        tests_passed += 1
    else:
        tests_failed += 1

    # Test 3: Tags (Models)
    print("\n3. Models Endpoint")
    success, data = test_endpoint(base_url, "GET", "/api/tags", "/api/tags")
    if success and data and "models" in data:
        models = data["models"]
        print(f"     Available models: {len(models)}")
        if models:
            for m in models[:3]:  # Show first 3
                print(f"       - {m.get('name', 'unknown')}")
            tests_passed += 1
        else:
            print("     (No models installed - endpoint works but no data)")
            tests_passed += 1
    else:
        tests_failed += 1

    # Test 4: Running Models
    print("\n4. Running Models Endpoint")
    success, data = test_endpoint(base_url, "GET", "/api/ps", "/api/ps")
    if success and data and "models" in data:
        running = data["models"]
        print(f"     Running models: {len(running)}")
        tests_passed += 1
    else:
        tests_failed += 1

    # Test 5: Model Details
    print("\n5. Model Details Endpoint")
    # First get a model name
    success_tags, data_tags = test_endpoint(
        base_url, "GET", "/api/tags", "Get model for show test"
    )
    if success_tags and data_tags and data_tags.get("models"):
        model_name = data_tags["models"][0]["name"]
        success, data = test_endpoint(
            base_url,
            "POST",
            "/api/show",
            f"/api/show (model: {model_name})",
            payload={"name": model_name},
        )
        if success and data:
            if "error" not in data:
                print(f"     License: {data.get('license', 'unknown')}")
                tests_passed += 1
            else:
                print(f"     Error: {data['error']}")
                tests_failed += 1
        else:
            tests_failed += 1
    else:
        print("     (Skipping - no models available)")
        tests_passed += 1

    # Test 6: CORS
    print("\n6. CORS Headers")
    success, _ = test_endpoint(
        base_url,
        "OPTIONS",
        "/api/chat",
        "CORS preflight",
        headers={"Origin": "http://zed-editor.local"},
    )
    # We don't check success here because OPTIONS might return various status codes
    # Just check if the response exists
    try:
        response = requests.options(
            f"{base_url}/api/chat", headers={"Origin": "http://test.com"}, timeout=3
        )
        if "access-control-allow-origin" in response.headers:
            cors_val = response.headers["access-control-allow-origin"]
            print(f"     CORS Origin: {cors_val}")
            tests_passed += 1
        else:
            print("     ✗ CORS headers missing")
            tests_failed += 1
    except Exception as e:
        print(f"     ✗ CORS check failed: {e}")
        tests_failed += 1

    # Test 7: Chat (if models available)
    print("\n7. Chat Endpoint")
    if success_tags and data_tags and data_tags.get("models"):
        model_name = data_tags["models"][0]["name"]
        try:
            response = requests.post(
                f"{base_url}/api/chat",
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": True,
                },
                headers={"Accept": "application/x-ndjson"},
                stream=True,
                timeout=10,
            )

            if response.status_code == 200:
                # Read first chunk
                chunk = None
                for line in response.iter_lines():
                    if line:
                        try:
                            chunk = json.loads(line)
                            break
                        except:
                            continue

                if chunk and "message" in chunk:
                    print(f"     ✓ Streaming works")
                    print(f"     Model: {chunk.get('model', 'unknown')}")
                    tests_passed += 1
                else:
                    print(f"     ✗ Unexpected response format")
                    tests_failed += 1
            else:
                print(f"     ✗ HTTP {response.status_code}")
                tests_failed += 1
        except Exception as e:
            print(f"     ✗ Chat test failed: {e}")
            tests_failed += 1
    else:
        print("     (Skipping - no models available)")
        tests_passed += 1

    # Summary
    print("\n" + "=" * 60)
    print(f"Results: {tests_passed} passed, {tests_failed} failed")
    print("=" * 60)

    return tests_failed == 0


def main():
    parser = argparse.ArgumentParser(description="Test ANE Studio API endpoints")
    parser.add_argument("--url", help="Server URL (default: auto-detect)")
    parser.add_argument("--network", action="store_true", help="Test using network IP")
    args = parser.parse_args()

    if args.url:
        base_url = args.url
    elif args.network:
        local_ip = get_local_ip()
        if local_ip:
            base_url = f"http://{local_ip}:11436"
        else:
            print("Error: Could not determine network IP")
            sys.exit(1)
    else:
        base_url = "http://127.0.0.1:11436"

    print("=" * 60)
    print("ANE Studio API Endpoint Validator")
    print("=" * 60)
    print(f"\nTarget: {base_url}")

    # Check if server is running
    try:
        response = requests.get(f"{base_url}/api/version", timeout=3)
        print("Server status: ✓ Online")
    except requests.exceptions.ConnectionError:
        print("Server status: ✗ Not running")
        print(f"\nStart the server first with:")
        print(f"  ./boot_api.sh")
        print(f"or")
        print(f"  python ane_studio/server.py")
        sys.exit(1)
    except Exception as e:
        print(f"Server status: ✗ Error ({e})")
        sys.exit(1)

    success = run_tests(base_url)

    if success:
        print("\n✓ All tests passed!")
        print("\nThe API is ready for use with:")
        print("  - Zed Editor")
        print("  - Other Ollama-compatible clients")
        print(f"\nUse this URL: {base_url}")
        return 0
    else:
        print("\n✗ Some tests failed")
        print("\nCheck the server logs for errors")
        return 1


if __name__ == "__main__":
    sys.exit(main())
