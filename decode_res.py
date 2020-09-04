import sys
import base64 as b 
import zlib as z

s = input("Please paste the encoded blob: ")
with open("rebuilt.py", "wb") as fil:
    fil.write(z.decompress(b.b64decode(s.replace('`','/'))))
    print("------------------------------")
    print("Wrote solution to ./rebuilt.py")
