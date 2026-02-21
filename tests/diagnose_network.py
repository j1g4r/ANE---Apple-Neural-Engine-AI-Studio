#!/usr/bin/env python3
"""
Network Diagnostics Tool for ANE Chat Server

This script diagnoses network connectivity issues for the Ollama-compatible API server.

Usage:
    python tests/diagnose_network.py

"""

import json
import os
import socket
import subprocess
import sys

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("Warning: requests library not installed. Install with: pip install requests")

DEFAULT_PORT = 11436


def get_local_ip():
    """Get the primary local IP address"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception as e:
        return None


def get_all_interfaces():
    """Get all network interfaces"""
    interfaces = []
    try:
        # Try to get all interfaces using ifconfig
        result = subprocess.run(["ifconfig"], capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "inet " in line and "127.0.0.1" not in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "inet":
                            ip = parts[i + 1]
                            interfaces.append(ip)
                            break
    except Exception:
        pass

    # Fallback to hostname
    try:
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        if ip and ip not in interfaces and ip != "127.0.0.1":
            interfaces.append(ip)
    except Exception:
        pass

    return interfaces


def check_port_binding(port):
    """Check if a port is already in use"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(("127.0.0.1", port))
        sock.close()
        return result == 0  # True if port is in use
    except Exception:
        return False


def test_server_response(url, timeout=3):
    """Test if server responds at URL"""
    if not HAS_REQUESTS:
        return None, "requests library not installed"

    try:
        response = requests.get(url, timeout=timeout)
        return response.status_code, None
    except requests.exceptions.ConnectionError as e:
        return None, f"Connection refused: {e}"
    except requests.exceptions.Timeout:
        return None, "Connection timed out"
    except Exception as e:
        return None, str(e)


def check_firewall_status():
    """Check macOS firewall status"""
    try:
        result = subprocess.run(
            [
                "sudo",
                "defaults",
                "read",
                "/Library/Preferences/com.apple.alf.plist",
                "globalstate",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            status = result.stdout.strip()
            if status == "0":
                return "OFF"
            elif status == "1":
                return "ON (essential services)"
            elif status == "2":
                return "ON (all connections)"
        return "Unknown"
    except Exception:
        return "Unknown (requires sudo)"


def main():
    print("=" * 70)
    print("ANE Chat Server - Network Diagnostics")
    print("=" * 70)
    print()

    # 1. Check Python and dependencies
    print("1. Python Environment")
    print("-" * 40)
    print(f"   Python version: {sys.version.split()[0]}")
    print(
        f"   Requests library: {'✓ Installed' if HAS_REQUESTS else '✗ Not installed'}"
    )
    print()

    # 2. Network Configuration
    print("2. Network Configuration")
    print("-" * 40)

    local_ip = get_local_ip()
    if local_ip:
        print(f"   Primary IP: {local_ip}")
    else:
        print("   Primary IP: ✗ Could not determine")

    all_ips = get_all_interfaces()
    if all_ips:
        print(f"   All interfaces: {', '.join(all_ips)}")
    else:
        print("   All interfaces: None found")

    hostname = socket.gethostname()
    print(f"   Hostname: {hostname}")
    print()

    # 3. Server Status
    print("3. Server Status")
    print("-" * 40)

    port = DEFAULT_PORT
    localhost_url = f"http://127.0.0.1:{port}"
    local_ip_url = f"http://{local_ip}:{port}" if local_ip else None

    # Check if port is in use
    port_in_use = check_port_binding(port)
    if port_in_use:
        print(f"   Port {port}: ✓ In use (server may be running)")
    else:
        print(f"   Port {port}: ✗ Not in use (server not running)")
        print()
        print("   ⚠ Start the server with: ./boot_api.sh")
        print()
        return 1

    # Test localhost
    if HAS_REQUESTS:
        status, error = test_server_response(f"{localhost_url}/api/version")
        if status == 200:
            print(f"   Localhost (127.0.0.1:{port}): ✓ Responding")
            try:
                response = requests.get(f"{localhost_url}/api/version", timeout=3)
                version = response.json().get("version", "unknown")
                print(f"      API Version: {version}")
            except:
                pass
        else:
            print(
                f"   Localhost (127.0.0.1:{port}): ✗ {error or 'Status ' + str(status)}"
            )

        # Test local IP
        if local_ip_url:
            status, error = test_server_response(f"{local_ip_url}/api/version")
            if status == 200:
                print(f"   Network ({local_ip}:{port}): ✓ Responding")
            else:
                print(
                    f"   Network ({local_ip}:{port}): ✗ {error or 'Status ' + str(status)}"
                )

                if "refused" in str(error).lower():
                    print()
                    print("   ⚠ POSSIBLE CAUSES:")
                    print("      1. Server not binding to 0.0.0.0 (all interfaces)")
                    print("      2. Firewall blocking incoming connections")
                    print("      3. Network interface not configured")
    else:
        print("   Cannot test (requests library not installed)")
    print()

    # 4. Firewall Status
    print("4. Firewall Status (macOS)")
    print("-" * 40)
    firewall_status = check_firewall_status()
    print(f"   Status: {firewall_status}")

    if "ON" in firewall_status:
        print()
        print("   ⚠ Firewall is enabled!")
        print("      - First run may prompt for network access")
        print("      - Click 'Allow' when macOS asks for incoming connections")
        print("      - Or add Python to Firewall exceptions in:")
        print("        System Preferences > Security & Privacy > Firewall")
    print()

    # 5. CORS Check
    print("5. CORS Configuration")
    print("-" * 40)

    if HAS_REQUESTS and port_in_use:
        try:
            response = requests.options(
                f"{localhost_url}/api/chat",
                headers={"Origin": "http://test.example.com"},
                timeout=3,
            )
            if "access-control-allow-origin" in response.headers:
                cors_value = response.headers["access-control-allow-origin"]
                print(f"   CORS Origin: {cors_value}")
                if cors_value == "*":
                    print("   ✓ Accepting all origins")
                else:
                    print(f"   Allowing specific origin: {cors_value}")
            else:
                print("   ✗ CORS headers missing!")
                print("      This will block browser/editor clients")
        except Exception as e:
            print(f"   Could not check CORS: {e}")
    else:
        print("   Cannot test (server not running)")
    print()

    # 6. Zed Editor Configuration
    print("6. Zed Editor Configuration")
    print("-" * 40)
    print("   Add this to your Zed settings.json:")
    print()
    print('   "assistant": {')
    print('     "default_model": {')
    print('       "provider": "ollama",')
    if local_ip:
        print(f'       "model": "qwen05b",')
    print("     },")
    print('     "version": "2"')
    print("   }")
    print()
    print("   Environment variable for Ollama URL:")
    if local_ip_url:
        print(f'   export OLLAMA_API_BASE="{local_ip_url}"')
    else:
        print('   export OLLAMA_API_BASE="http://YOUR_IP:11436"')
    print()

    # 7. Test Summary
    print("7. Summary")
    print("-" * 40)
    if port_in_use:
        if local_ip:
            print(f"   Local access:    http://127.0.0.1:{port}")
            print(f"   Network access:  http://{local_ip}:{port}")
            print()
            print("   ✓ Server is running")
            print("   ✓ API should be accessible from this machine")

            # Check if network access works
            if HAS_REQUESTS and local_ip_url:
                status, _ = test_server_response(f"{local_ip_url}/api/version")
                if status == 200:
                    print("   ✓ Server is accessible from network")
                    print()
                    print("=" * 70)
                    print("SUCCESS! The API should work with Zed and other clients.")
                    print("=" * 70)
                else:
                    print("   ✗ Server may not be accessible from network")
                    print()
                    print("=" * 70)
                    print("ISSUE DETECTED!")
                    print("=" * 70)
                    print()
                    print("The server responds on localhost but NOT on the network IP.")
                    print()
                    print("TROUBLESHOOTING STEPS:")
                    print("1. Stop the server and restart it")
                    print("2. When macOS prompts for firewall access, click 'Allow'")
                    print("3. Check the boot_api.sh script binds to 0.0.0.0:")
                    print("   uvicorn chat_server:app --host 0.0.0.0 --port 11436")
                    print("4. Temporarily disable firewall to test:")
                    print(
                        "   sudo /usr/libexec/ApplicationFirewall/socketfilterfw --setglobalstate off"
                    )
                    print()
                    return 1
        else:
            print("   Could not determine network configuration")
    else:
        print("   ✗ Server is not running")
        print("   Start with: ./boot_api.sh")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
