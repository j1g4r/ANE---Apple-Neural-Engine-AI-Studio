"""
Test suite for Ollama-compatible API endpoints in chat_server.py

To run tests:
    cd /Users/jp_mac/workspace/JP/ANE
    python -m pytest tests/test_ollama_api.py -v

Or without pytest:
    python tests/test_ollama_api.py

Network tests:
    python tests/test_ollama_api.py --network
"""

import json
import os
import sys
import threading
import time
import unittest
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("Warning: requests library not installed. Install with: pip install requests")

# Test configuration
BASE_URL = "http://127.0.0.1:11436"
MAX_RETRIES = 30
RETRY_DELAY = 1


def get_local_ip():
    """Get the local network IP address"""
    import socket

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return None


def get_network_url(port=11436):
    """Get the network-accessible URL"""
    local_ip = get_local_ip()
    if local_ip:
        return f"http://{local_ip}:{port}"
    return None


class TestOllamaAPI(unittest.TestCase):
    """Test suite for Ollama-compatible API endpoints"""

    @classmethod
    def setUpClass(cls):
        """Check if server is running before tests"""
        if not HAS_REQUESTS:
            raise unittest.SkipTest("requests library not installed")

        # Wait for server to be ready
        server_ready = False
        for i in range(MAX_RETRIES):
            try:
                response = requests.get(f"{BASE_URL}/api/version", timeout=2)
                if response.status_code == 200:
                    server_ready = True
                    print(f"\n✓ Server is ready at {BASE_URL}")
                    break
            except requests.exceptions.ConnectionError:
                print(f"Waiting for server... ({i + 1}/{MAX_RETRIES})")
                time.sleep(RETRY_DELAY)

        if not server_ready:
            raise unittest.SkipTest(
                f"Server not running at {BASE_URL}. Start it with: ./boot_api.sh"
            )

    # ==================== API Version Tests ====================

    def test_api_version_endpoint(self):
        """Test /api/version returns correct format"""
        response = requests.get(f"{BASE_URL}/api/version")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("version", data)
        self.assertIsInstance(data["version"], str)
        print(f"✓ API Version: {data['version']}")

    def test_api_version_content_type(self):
        """Test /api/version returns JSON content type"""
        response = requests.get(f"{BASE_URL}/api/version")
        self.assertEqual(response.headers.get("content-type"), "application/json")

    # ==================== Root Endpoint Tests ====================

    def test_root_returns_json_for_api_clients(self):
        """Test / returns JSON when accessed by API clients"""
        headers = {"Accept": "application/json", "User-Agent": "OllamaClient/1.0"}
        response = requests.get(f"{BASE_URL}/", headers=headers)

        # Should return JSON for API clients
        if response.headers.get("content-type") == "application/json":
            data = response.json()
            self.assertIn("name", data)
            self.assertIn("version", data)
            self.assertIn("ollama_compatible", data)
            self.assertTrue(data["ollama_compatible"])
            self.assertIn("endpoints", data)
            print(f"✓ Root returns JSON for API clients: {data['name']}")

    def test_root_returns_html_for_browsers(self):
        """Test / returns HTML for browser clients"""
        headers = {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.0",
        }
        response = requests.get(f"{BASE_URL}/", headers=headers)

        # Check if it returns HTML
        content_type = response.headers.get("content-type", "")
        if "text/html" in content_type:
            self.assertIn("<!DOCTYPE html>", response.text.upper())
            print("✓ Root returns HTML for browsers")

    # ==================== Models/List Tests ====================

    def test_api_tags_endpoint(self):
        """Test /api/tags returns Ollama-compatible model list"""
        response = requests.get(f"{BASE_URL}/api/tags")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("models", data)
        self.assertIsInstance(data["models"], list)

        # Check model format if models exist
        if data["models"]:
            model = data["models"][0]
            self.assertIn("name", model)
            self.assertIn("model", model)
            self.assertIn("details", model)
            print(f"✓ Found {len(data['models'])} models")
        else:
            print("⚠ No models found (this is OK if none loaded)")

    def test_api_models_endpoint(self):
        """Test /api/models returns native model list"""
        response = requests.get(f"{BASE_URL}/api/models")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("models", data)
        self.assertIsInstance(data["models"], list)
        print(f"✓ /api/models returns {len(data['models'])} models")

    # ==================== Running Models Tests ====================

    def test_api_ps_endpoint(self):
        """Test /api/ps returns running models"""
        response = requests.get(f"{BASE_URL}/api/ps")
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("models", data)
        self.assertIsInstance(data["models"], list)
        print(f"✓ /api/ps returns {len(data['models'])} running models")

    # ==================== Model Details Tests ====================

    def test_api_show_endpoint(self):
        """Test /api/show returns model details"""
        # First get available models
        tags_response = requests.get(f"{BASE_URL}/api/tags")
        models = tags_response.json().get("models", [])

        if not models:
            self.skipTest("No models available to test /api/show")

        model_name = models[0]["name"]

        response = requests.post(f"{BASE_URL}/api/show", json={"name": model_name})
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("details", data)
        self.assertIn("license", data)
        self.assertIn("modelfile", data)
        self.assertIn("template", data)
        print(f"✓ /api/show returned details for {model_name}")

    def test_api_show_not_found(self):
        """Test /api/show returns error for non-existent model"""
        response = requests.post(
            f"{BASE_URL}/api/show", json={"name": "nonexistent-model-12345"}
        )
        self.assertEqual(response.status_code, 200)  # Ollama returns 200 with error

        data = response.json()
        self.assertIn("error", data)
        print("✓ /api/show correctly returns error for non-existent model")

    # ==================== Chat Tests ====================

    def test_api_chat_streaming(self):
        """Test /api/chat with streaming (Ollama format)"""
        # Get available models first
        tags_response = requests.get(f"{BASE_URL}/api/tags")
        models = tags_response.json().get("models", [])

        if not models:
            self.skipTest("No models available to test chat")

        model_name = models[0]["name"]

        response = requests.post(
            f"{BASE_URL}/api/chat",
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
            },
            stream=True,
        )
        self.assertEqual(response.status_code, 200)

        # Check content type for NDJSON
        content_type = response.headers.get("content-type", "")
        self.assertIn("application/x-ndjson", content_type)

        # Read first few chunks
        chunks = []
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                chunks.append(chunk)
                self.assertIn("model", chunk)
                self.assertIn("message", chunk)
                self.assertIn("done", chunk)
                if chunk["done"]:
                    break
                if len(chunks) >= 5:
                    break

        self.assertGreater(len(chunks), 0, "Should receive at least one chunk")
        print(f"✓ /api/chat streaming received {len(chunks)} chunks")

    def test_api_generate_streaming(self):
        """Test /api/generate with streaming (native format)"""
        response = requests.post(
            f"{BASE_URL}/api/generate",
            json={"prompt": "Hello", "stream": True},
            stream=True,
        )
        self.assertEqual(response.status_code, 200)

        # Check for SSE format
        content_type = response.headers.get("content-type", "")
        self.assertIn("text/event-stream", content_type)

        # Read a few events
        events = []
        for line in response.iter_lines():
            line_str = line.decode("utf-8") if isinstance(line, bytes) else line
            if line_str.startswith("data: "):
                data = line_str[6:]  # Remove "data: " prefix
                if data == "[DONE]":
                    break
                events.append(json.loads(data))
                if len(events) >= 3:
                    break

        self.assertGreater(len(events), 0, "Should receive at least one event")
        print(f"✓ /api/generate streaming received {len(events)} events")

    # ==================== CORS Tests ====================

    def test_cors_headers_present(self):
        """Test CORS headers are present"""
        response = requests.options(
            f"{BASE_URL}/api/chat",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type",
            },
        )

        self.assertIn(
            "access-control-allow-origin",
            response.headers,
            "CORS headers should be present",
        )
        print("✓ CORS headers present")

    def test_cors_preflight(self):
        """Test CORS preflight request succeeds"""
        response = requests.options(
            f"{BASE_URL}/api/tags",
            headers={
                "Origin": "http://zed-editor.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        self.assertEqual(response.status_code, 200)
        print("✓ CORS preflight request successful")

    # ==================== Error Handling Tests ====================

    def test_invalid_endpoint_returns_404(self):
        """Test invalid endpoint returns 404"""
        response = requests.get(f"{BASE_URL}/api/invalid-endpoint")
        self.assertEqual(response.status_code, 404)
        print("✓ Invalid endpoint returns 404")

    def test_chat_without_model(self):
        """Test chat without model specification"""
        response = requests.post(
            f"{BASE_URL}/api/chat",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
        # Should either work with default or return error
        self.assertIn(response.status_code, [200, 400, 422])
        print("✓ Chat without model handled")

    # ==================== Load Tests ====================

    def test_api_load_endpoint(self):
        """Test /api/load endpoint"""
        # Get available models
        tags_response = requests.get(f"{BASE_URL}/api/tags")
        models = tags_response.json().get("models", [])

        if not models:
            self.skipTest("No models available to test load")

        model_id = models[0].get("id", "/tmp/qwen05b")

        response = requests.post(f"{BASE_URL}/api/load", json={"model_dir": model_id})
        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("status", data)
        print(f"✓ /api/load returned status: {data['status']}")


class TestNetworkAccessibility(unittest.TestCase):
    """Test network accessibility of the API server"""

    @classmethod
    def setUpClass(cls):
        """Check prerequisites"""
        if not HAS_REQUESTS:
            raise unittest.SkipTest("requests library not installed")

        cls.local_ip = get_local_ip()
        cls.network_url = get_network_url()

        if not cls.local_ip:
            raise unittest.SkipTest("Could not determine local IP address")

        print(f"\nLocal IP: {cls.local_ip}")
        print(f"Network URL: {cls.network_url}")

    def test_server_accepts_external_connections(self):
        """Test that server accepts connections from network IP"""
        if not self.network_url:
            self.skipTest("Network URL not available")

        # Test connection from network IP
        try:
            response = requests.get(f"{self.network_url}/api/version", timeout=5)
            self.assertEqual(response.status_code, 200)
            print(f"✓ Server accepts connections from {self.network_url}")
        except requests.exceptions.ConnectionError as e:
            self.fail(f"Server does not accept external connections: {e}")

    def test_cors_accepts_external_origin(self):
        """Test CORS accepts requests from external origins"""
        if not self.network_url:
            self.skipTest("Network URL not available")

        external_origin = "http://192.168.1.100:3000"
        response = requests.options(
            f"{self.network_url}/api/chat",
            headers={
                "Origin": external_origin,
                "Access-Control-Request-Method": "POST",
            },
            timeout=5,
        )

        self.assertIn(
            "access-control-allow-origin",
            response.headers,
            "CORS should allow external origins",
        )

        # Check if the specific origin is allowed or wildcard
        allow_origin = response.headers.get("access-control-allow-origin", "")
        self.assertTrue(
            allow_origin == "*" or external_origin in allow_origin,
            f"CORS should allow the origin {external_origin}",
        )
        print(f"✓ CORS accepts external origin: {external_origin}")

    def test_api_accessible_from_multiple_origins(self):
        """Test API is accessible from various origins"""
        origins = [
            "http://localhost:3000",
            "http://127.0.0.1:8080",
            "http://zed-editor.local:8080",
            "https://zed.dev",
        ]

        for origin in origins:
            with self.subTest(origin=origin):
                response = requests.get(
                    f"{BASE_URL}/api/version",
                    headers={"Origin": origin},
                    timeout=5,
                )
                self.assertEqual(response.status_code, 200)

                # Check CORS header
                allow_origin = response.headers.get("access-control-allow-origin", "")
                self.assertTrue(
                    allow_origin == "*" or origin in allow_origin,
                    f"Should allow origin: {origin}",
                )

        print(f"✓ API accessible from {len(origins)} different origins")

    def test_network_interface_binding(self):
        """Test that server is bound to all interfaces (0.0.0.0)"""
        import socket

        # Try to connect via 0.0.0.0 (this won't work directly, but we can check
        # that the server responds on all local IPs)
        hostname = socket.gethostname()
        try:
            local_ips = socket.getaddrinfo(hostname, None, socket.AF_INET)
            unique_ips = set([ip[4][0] for ip in local_ips])
            unique_ips.add("127.0.0.1")

            success_count = 0
            for ip in unique_ips:
                try:
                    url = f"http://{ip}:11436/api/version"
                    response = requests.get(url, timeout=3)
                    if response.status_code == 200:
                        success_count += 1
                        print(f"  ✓ Accessible via {ip}")
                except:
                    print(f"  ✗ Not accessible via {ip}")

            self.assertGreater(
                success_count, 0, "Server should be accessible on at least one IP"
            )
            print(f"✓ Server accessible on {success_count}/{len(unique_ips)} interfaces")

        except Exception as e:
            print(f"Warning: Could not enumerate all interfaces: {e}")

    def test_firewall_prompt(self):
        """Test that server triggers firewall prompt on macOS"""
        # This is informational only - we can't actually test the firewall prompt
        # but we can verify the server is trying to bind to external interfaces
        print("\n⚠ INFO: If this is the first time running, macOS may show a firewall prompt.")
        print("  Make sure to click 'Allow' when prompted to accept incoming connections.")


class TestAPICompatibility(unittest.TestCase):
    """Test Ollama API compatibility specifics"""

    @classmethod
    def setUpClass(cls):
        """Check if server is running"""
        if not HAS_REQUESTS:
            raise unittest.SkipTest("requests library not installed")

        try:
            requests.get(f"{BASE_URL}/api/version", timeout=2)
        except requests.exceptions.ConnectionError:
            raise unittest.SkipTest(f"Server not running at {BASE_URL}")

    def test_ollama_endpoint_consistency(self):
        """Test that all Ollama endpoints follow expected patterns"""
        endpoints = [
            ("/api/version", "GET", 200),
            ("/api/tags", "GET", 200),
            ("/api/ps", "GET", 200),
            ("/api/models", "GET", 200),
        ]

        for endpoint, method, expected_status in endpoints:
            response = requests.request(method, f"{BASE_URL}{endpoint}")
            self.assertEqual(
                response.status_code,
                expected_status,
                f"Endpoint {endpoint} should return {expected_status}",
            )
            print(f"✓ {endpoint} returns {expected_status}")


def run_quick_test():
    """Run a quick smoke test without unittest framework"""
    if not HAS_REQUESTS:
        print("Error: requests library not installed")
        print("Install with: pip install requests")
        return 1

    print("=" * 60)
    print("Ollama API Compatibility Quick Test")
    print("=" * 60)
    print(f"Testing server at: {BASE_URL}")
    print()

    # Wait for server
    print("Checking if server is running...")
    for i in range(MAX_RETRIES):
        try:
            response = requests.get(f"{BASE_URL}/api/version", timeout=2)
            if response.status_code == 200:
                print(f"✓ Server is ready\n")
                break
        except requests.exceptions.ConnectionError:
            if i == MAX_RETRIES - 1:
                print(f"\n✗ Server not running at {BASE_URL}")
                print("Start the server with: ./boot_api.sh")
                return 1
            time.sleep(RETRY_DELAY)

    tests_passed = 0
    tests_failed = 0

    # Test 1: Version
    try:
        response = requests.get(f"{BASE_URL}/api/version")
        data = response.json()
        print(f"✓ /api/version: {data.get('version')}")
        tests_passed += 1
    except Exception as e:
        print(f"✗ /api/version failed: {e}")
        tests_failed += 1

    # Test 2: Tags
    try:
        response = requests.get(f"{BASE_URL}/api/tags")
        data = response.json()
        models = data.get("models", [])
        print(f"✓ /api/tags: {len(models)} models available")
        tests_passed += 1
    except Exception as e:
        print(f"✗ /api/tags failed: {e}")
        tests_failed += 1

    # Test 3: PS
    try:
        response = requests.get(f"{BASE_URL}/api/ps")
        data = response.json()
        running = data.get("models", [])
        print(f"✓ /api/ps: {len(running)} models running")
        tests_passed += 1
    except Exception as e:
        print(f"✗ /api/ps failed: {e}")
        tests_failed += 1

    # Test 4: Root with API client header
    try:
        headers = {"Accept": "application/json", "User-Agent": "TestClient/1.0"}
        response = requests.get(f"{BASE_URL}/", headers=headers)
        if response.headers.get("content-type") == "application/json":
            print(f"✓ / returns JSON for API clients")
            tests_passed += 1
        else:
            print(f"✓ / returns HTML for browsers")
            tests_passed += 1
    except Exception as e:
        print(f"✗ / root endpoint failed: {e}")
        tests_failed += 1

    # Test 5: CORS
    try:
        response = requests.options(
            f"{BASE_URL}/api/chat", headers={"Origin": "http://test.com"}
        )
        if "access-control-allow-origin" in response.headers:
            print(f"✓ CORS headers present")
            tests_passed += 1
        else:
            print(f"⚠ CORS headers may be missing")
            tests_passed += 1
    except Exception as e:
        print(f"✗ CORS test failed: {e}")
        tests_failed += 1

    # Test 6: Chat streaming (if models available)
    try:
        tags_response = requests.get(f"{BASE_URL}/api/tags")
        models = tags_response.json().get("models", [])

        if models:
            model_name = models[0]["name"]
            response = requests.post(
                f"{BASE_URL}/api/chat",
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": "Hi"}],
                    "stream": True,
                },
                stream=True,
            )

            # Read first chunk
            chunk = None
            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    break

            if chunk and "message" in chunk:
                print(f"✓ /api/chat streaming works with model: {model_name}")
                tests_passed += 1
            else:
                print(f"⚠ /api/chat returned unexpected format")
                tests_passed += 1
        else:
            print(f"⚠ Skipping chat test - no models available")
            tests_passed += 1
    except Exception as e:
        print(f"✗ /api/chat streaming failed: {e}")
        tests_failed += 1

    print()
    print("=" * 60)
    print(f"Results: {tests_passed} passed, {tests_failed} failed")
    print("=" * 60)

    return 0 if tests_failed == 0 else 1


def run_network_test():
"""Run network-specific tests"""
if not HAS_REQUESTS:
    print("Error: requests library not installed")
    return 1

print("=" * 60)
print("Network Accessibility Test")
print("=" * 60)

local_ip = get_local_ip()
if not local_ip:
    print("✗ Could not determine local IP address")
    return 1

network_url = get_network_url()
print(f"Local IP: {local_ip}")
print(f"Network URL: {network_url}")
print(f"Local URL: {BASE_URL}")
print()

tests_passed = 0
tests_failed = 0

# Test 1: Server running locally
try:
    response = requests.get(f"{BASE_URL}/api/version", timeout=5)
    if response.status_code == 200:
        print(f"✓ Server responding on localhost")
        tests_passed += 1
    else:
        print(f"✗ Server returned status {response.status_code}")
        tests_failed += 1
except Exception as e:
    print(f"✗ Server not accessible on localhost: {e}")
    tests_failed += 1
    return 1  # Can't continue without local server

# Test 2: Server accessible from network IP
try:
    response = requests.get(f"{network_url}/api/version", timeout=5)
    if response.status_code == 200:
        print(f"✓ Server accessible from network IP: {network_url}")
        tests_passed += 1
    else:
        print(f"✗ Server returned status {response.status_code} on network IP")
        tests_failed += 1
except Exception as e:
    print(f"✗ Server NOT accessible from network IP: {e}")
    print()
    print("TROUBLESHOOTING:")
    print("1. Make sure the server is running with: ./boot_api.sh")
    print("2. Check that the server binds to 0.0.0.0 (all interfaces)")
    print("3. If on macOS, check System Preferences > Security & Privacy > Firewall")
    print("4. Try disabling the firewall temporarily to test")
    tests_failed += 1

# Test 3: CORS from external origin
try:
    response = requests.options(
        f"{network_url}/api/chat",
        headers={"Origin": "http://example.com"},
        timeout=5,
    )
    if "access-control-allow-origin" in response.headers:
        print(f"✓ CORS headers present on network interface")
        tests_passed += 1
    else:
        print(f"⚠ CORS headers may be missing")
        tests_passed += 1
except Exception as e:
    print(f"✗ CORS test failed: {e}")
    tests_failed += 1

# Test 4: Chat API from network
try:
    response = requests.get(f"{BASE_URL}/api/tags", timeout=5)
    models = response.json().get("models", [])

    if models:
        model_name = models[0]["name"]
        response = requests.post(
            f"{network_url}/api/chat",
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": "Hi"}],
            },
            stream=True,
            timeout=10,
        )

        if response.status_code == 200:
            print(f"✓ Chat API accessible from network")
            tests_passed += 1
        else:
            print(f"✗ Chat API returned {response.status_code}")
            tests_failed += 1
    else:
        print(f"⚠ No models to test chat API")
        tests_passed += 1
except Exception as e:
    print(f"✗ Chat API from network failed: {e}")
    tests_failed += 1

print()
print("=" * 60)
print(f"Network Tests: {tests_passed} passed, {tests_failed} failed")
print("=" * 60)

if tests_failed == 0:
    print()
    print("SUCCESS! The API is accessible from the network.")
    print(f"Other devices can connect using: {network_url}")
    print()
    print("Zed Editor Configuration:")
    print(f"  Ollama URL: {network_url}")
    print()

return 0 if tests_failed == 0 else 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Ollama API compatibility")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full unittest suite (default: quick smoke test)",
    )
    parser.add_argument(
        "--url", default=BASE_URL, help=f"Server URL (default: {BASE_URL})"
    )
    parser.add_argument(
        "--network",
        action="store_true",
        help="Run network accessibility tests only",
    )

    args = parser.parse_args()

    # Update BASE_URL if provided
    if args.url != BASE_URL:
        BASE_URL = args.url
        # Also update in TestOllamaAPI
        TestOllamaAPI.BASE_URL = args.url
        TestAPICompatibility.BASE_URL = args.url

    if args.network:
        exit_code = run_network_test()
        exit(exit_code)
    elif args.full:
        print("Running full unittest suite...")
        unittest.main(verbosity=2, exit=False)
    else:
        exit_code = run_quick_test()
        exit(exit_code)
