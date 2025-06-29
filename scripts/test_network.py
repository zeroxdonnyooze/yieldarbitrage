#!/usr/bin/env python3
"""Network diagnostic script for Railway deployment."""
import socket
import asyncio
import asyncpg
import os

async def test_supabase_connection():
    """Test connection to Supabase."""
    host = "db.eurnzugexarjxqkbulkr.supabase.co"
    port = 5432
    
    print(f"🔍 Testing connection to {host}:{port}")
    
    # Test DNS resolution
    try:
        addr_info = socket.getaddrinfo(host, port, family=socket.AF_UNSPEC)
        print(f"✅ DNS resolution successful:")
        for info in addr_info:
            family_name = "IPv4" if info[0] == socket.AF_INET else "IPv6"
            print(f"  - {family_name}: {info[4][0]}")
    except Exception as e:
        print(f"❌ DNS resolution failed: {e}")
        return False
    
    # Test IPv4 connection
    try:
        addr_info_v4 = socket.getaddrinfo(host, port, family=socket.AF_INET)
        if addr_info_v4:
            ipv4 = addr_info_v4[0][4][0]
            print(f"🔍 Testing IPv4 connection to {ipv4}:{port}")
            
            # Test basic socket connection
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)
            result = sock.connect_ex((ipv4, port))
            sock.close()
            
            if result == 0:
                print("✅ IPv4 socket connection successful")
                return await test_postgres_connection(ipv4)
            else:
                print(f"❌ IPv4 socket connection failed: {result}")
        else:
            print("❌ No IPv4 address found")
    except Exception as e:
        print(f"❌ IPv4 test failed: {e}")
    
    # Test IPv6 connection
    try:
        addr_info_v6 = socket.getaddrinfo(host, port, family=socket.AF_INET6)
        if addr_info_v6:
            ipv6 = addr_info_v6[0][4][0]
            print(f"🔍 Testing IPv6 connection to [{ipv6}]:{port}")
            
            sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
            sock.settimeout(10)
            result = sock.connect_ex((ipv6, port))
            sock.close()
            
            if result == 0:
                print("✅ IPv6 socket connection successful")
                return await test_postgres_connection(f"[{ipv6}]")
            else:
                print(f"❌ IPv6 socket connection failed: {result}")
    except Exception as e:
        print(f"❌ IPv6 test failed: {e}")
    
    return False

async def test_postgres_connection(host):
    """Test PostgreSQL connection."""
    try:
        print(f"🔍 Testing PostgreSQL connection to {host}")
        conn = await asyncpg.connect(
            host=host.strip('[]'),  # Remove brackets for IPv6
            port=5432,
            user='postgres',
            password='cpDSoucIWcP4RffV',
            database='postgres',
            command_timeout=10
        )
        result = await conn.fetchval('SELECT version()')
        await conn.close()
        print(f"✅ PostgreSQL connection successful!")
        print(f"   Version: {result}")
        return True
    except Exception as e:
        print(f"❌ PostgreSQL connection failed: {e}")
        return False

if __name__ == "__main__":
    asyncio.run(test_supabase_connection())