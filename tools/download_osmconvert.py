import httpx, pathlib
pathlib.Path("tools").mkdir(exist_ok=True)
ua = "madrid-workzone-research/0.1 (research; +https://github.com/albertomoreno04/madrid-workzone)"
urls = [
    "https://wiki.openstreetmap.org/w/images/9/93/Osmconvert.exe",
    "https://m.m.i24.cc/osmconvert64-0.8.8p.exe",
]
for url in urls:
    try:
        with httpx.Client(follow_redirects=True, timeout=60.0, headers={"User-Agent": ua}) as c:
            r = c.get(url)
        print(url, "->", r.status_code, "bytes:", len(r.content))
        if r.status_code == 200 and len(r.content) > 100000:
            pathlib.Path("tools/osmconvert.exe").write_bytes(r.content)
            print("saved")
            break
    except Exception as e:
        print(url, "err:", e)
