from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
# from starlette.middleware.sessions import SessionMiddleware  # ❌ ไม่ต้องใช้แล้ว ถ้าไม่ล็อกอิน
import pymysql
import os
from pathlib import Path

app = FastAPI()

from dotenv import load_dotenv
load_dotenv()  # โหลดค่าจาก .env

# (ถ้าจะเก็บ session ไว้ใช้ภายหลัง ค่อยเปิด middleware อีกที)
# app.add_middleware(SessionMiddleware, secret_key="my-super-secret-key")

# ใช้ absolute path กันหยิบ static/templates โปรเจคอื่น
BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# ---------- DB CONNECTION HELPER (ยังเก็บไว้ใช้ /dbtest ได้) ----------
def get_conn():
    return pymysql.connect(
        host=os.getenv("DB_HOST", "127.0.0.1"),
        port=int(os.getenv("DB_PORT", "3308")),
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASS", ""),
        db=os.getenv("DB_NAME", "mypro"),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )

# ---------- OPTIONAL: เสิร์ฟ favicon โดยตรง ----------
@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse(str(BASE_DIR / "static" / "favicon.ico"))

# =============== ROUTES =================

# ✔️ หน้าแรกเข้าหน้า Dashboard เลย
@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

# ✔️ Dashboard ไม่เช็ค session แล้ว
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

# ❌ เอา /login และ /logout ออก (ไม่ต้องมีแล้ว)

# (ยังคง /dbtest ไว้สำหรับลองเชื่อมฐานข้อมูล)
@app.get("/dbtest")
def db_test():
    try:
        with get_conn() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SHOW TABLES")
                tables = cursor.fetchall()
        return {"status": "ok", "tables": tables}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

# (ยังใช้ได้สำหรับตรวจ path)
@app.get("/_where")
def where():
    return {
        "base_dir": str(BASE_DIR),
        "static_dir": str(BASE_DIR / "static"),
        "templates": str(BASE_DIR / "templates"),
    }
