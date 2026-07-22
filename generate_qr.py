"""
Generate a QR code pointing at your deployed app, to put on a slide.

    python generate_qr.py https://your-app.up.railway.app
"""
import sys
import qrcode

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python generate_qr.py <url>")
    url = sys.argv[1]
    img = qrcode.make(url, box_size=10, border=2)
    out = "qr_code.png"
    img.save(out)
    print(f"Saved {out} for {url} — drop it into your slides.")
