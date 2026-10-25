
import os

def identify_pdf(filepath):
    print(f"\nScanning: {os.path.basename(filepath)}")
    try:
        with open(filepath, 'rb') as f:
            # Read more bytes to find the title inside the PDF content
            content = f.read(150000)
            text = ""
            for byte in content:
                if 32 <= byte <= 126:
                    text += chr(byte)
                elif byte == 10 or byte == 13:
                    text += "\n"
            
            # Look for "Jurnal Nasional Teknik Elektro dan Teknologi Informasi"
            jnteti_idx = text.find("Jurnal Nasional Teknik Elektro")
            if jnteti_idx != -1:
                print(f"  - JNTETI found at: {text[jnteti_idx:jnteti_idx+200]}")
            
            # Look for Volume/Number/Year
            vol_idx = text.find("Vol.")
            if vol_idx != -1:
                print(f"  - Vol info: {text[vol_idx:vol_idx+100]}")
            
            # Look for Title metadata tag or first large-ish text block
            title_idx = text.find("/Title")
            if title_idx != -1:
                print(f"  - PDF Title Tag: {text[title_idx:title_idx+150]}")
            
            # If it's a JNTETI article, it often has "Available at:" or similar
            avail_idx = text.find("Available at:")
            if avail_idx != -1:
                print(f"  - Available link: {text[avail_idx:avail_idx+100]}")
    except Exception as e:
        print(f"  - Error reading file: {e}")

folder = r"c:\TA Deteksi Kelelahan\Jurnal JNTETI"
files = ["document (1).pdf", "document (2).pdf", "document.pdf", "document3.pdf", "18701-Article Text-99324-2-10-20250702.pdf", "23855-Article Text-131220-1-10-20260227.pdf"]

for file in files:
    path = os.path.join(folder, file)
    if os.path.exists(path):
        identify_pdf(path)
    else:
        print(f"File not found: {file}")
