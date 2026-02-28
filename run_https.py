"""
run_https.py

Utility script to automatically generate a self-signed certificate
and run the PULSE backend over HTTPS.

This is required for smartphones to allow access to camera and sensors (DeviceMotionEvent, getUserMedia).
"""

import sys
import socket

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

def generate_certs():
    try:
        import trustme
    except ImportError:
        print("Error: 'trustme' is not installed. Please run: pip install trustme")
        sys.exit(1)
        
    local_ip = get_local_ip()
    print(f"Generating self-signed cert for {local_ip}...")
    
    ca = trustme.CA()
    server_cert = ca.issue_cert("localhost", "127.0.0.1", local_ip)
    
    server_cert.private_key_pem.write_to_path("key.pem")
    for b in server_cert.cert_chain_pems:
        b.write_to_path("cert.pem")
        
    print("Certificates written to key.pem and cert.pem")
    return local_ip

if __name__ == "__main__":
    local_ip = generate_certs()
    print(f"==================================================")
    print(f"Starting PULSE Backend over HTTPS!")
    print(f"Access the frontend on your smartphone at:")
    print(f"https://{local_ip}:8000/app/index.html")
    print(f"==================================================")
    print(f"(Note: You will need to click 'Advanced' > 'Proceed' on the security warning in your browser since it is self-signed.)")
    print(f"==================================================\n")
    
    import uvicorn
    uvicorn.run(
        "backend.main:app", 
        host="0.0.0.0", 
        port=8000, 
        ssl_keyfile="key.pem", 
        ssl_certfile="cert.pem"
    )
