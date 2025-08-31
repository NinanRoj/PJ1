from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pymysql
import os
from pathlib import Path
from dotenv import load_dotenv

# === Path หลักของโปรเจ็กต์ ===
BASE_DIR = Path(__file__).resolve().parent

# === โหลด .env ที่อยู่ข้างไฟล์นี้โดยตรง (กันกรณีรันจากโฟลเดอร์อื่น) ===
load_dotenv(BASE_DIR / ".env")

app = FastAPI()

# === Static / Templates ผูกกับ BASE_DIR เพื่อกันหลงโปรเจ็กต์ ===
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# ---------- DB CONNECTION HELPER ----------
def get_conn():
    return pymysql.connect(
        host=os.getenv("DB_HOST", "127.0.0.1"),
        port=int(os.getenv("DB_PORT", "3306")),   # ปกติ MySQL คือ 3306 (ถ้าใช้ XAMPP/พอร์ตอื่นปรับใน .env)
        user=os.getenv("DB_USER", "root"),
        password=os.getenv("DB_PASS", ""),
        db=os.getenv("DB_NAME", "mypro"),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True,
    )

# ---------- Favicon (ส่ง 204 ถ้าไม่มีไฟล์) ----------
@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    path = BASE_DIR / "static" / "favicon.ico"
    if path.exists():
        return FileResponse(str(path))
    return HTMLResponse(status_code=204)

# =============== ROUTES =================

# หน้าแรก -> dashboard
@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

# ทดสอบเชื่อม DB แบบง่าย
@app.get("/dbtest")
def db_test():
    try:
        with get_conn() as conn:
            conn.ping(reconnect=True)
            with conn.cursor() as cursor:
                cursor.execute("SHOW TABLES")
                tables = cursor.fetchall()
        return {"status": "ok", "tables": tables}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})

# ตรวจ path ที่ FastAPI ใช้อยู่
@app.get("/_where")
def where():
    return {
        "base_dir": str(BASE_DIR),
        "static_dir": str(BASE_DIR / "static"),
        "templates": str(BASE_DIR / "templates"),
        "env_loaded_from": str(BASE_DIR / ".env"),
    }

# รันด้วย python main.py ก็ได้
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", reload=True)