from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select
from ..db import get_session
from ..db_models import User
from ..schemas import AuthRequest, TokenResponse
from ..auth import get_password_hash, verify_password, create_access_token
import jwt
import httpx
import os
import time
from urllib.parse import urlencode
from ..auth import SECRET_KEY, ALGORITHM
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

router = APIRouter(prefix="/api/auth", tags=["auth"])
security = HTTPBearer()


def _google_redirect_uri() -> str:
    return os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/api/auth/google/callback")

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security), session: Session = Depends(get_session)) -> User:
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        sub = payload.get("sub")
        if sub is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        user_id = int(sub)
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    except (TypeError, ValueError):
        raise HTTPException(status_code=401, detail="Invalid token")
    
    user = session.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

@router.post("/register", response_model=TokenResponse)
def register(req: AuthRequest, session: Session = Depends(get_session)):
    user = session.exec(select(User).where(User.email == req.email)).first()
    if user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    new_user = User(
        email=req.email,
        hashed_password=get_password_hash(req.password)
    )
    session.add(new_user)
    session.commit()
    session.refresh(new_user)
    
    from datetime import timedelta
    access_token = create_access_token(data={"sub": str(new_user.id)}, expires_delta=timedelta(days=7))
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/login", response_model=TokenResponse)
def login(req: AuthRequest, session: Session = Depends(get_session)):
    user = session.exec(select(User).where(User.email == req.email)).first()
    if not user or not verify_password(req.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    
    from datetime import timedelta
    access_token = create_access_token(data={"sub": str(user.id)}, expires_delta=timedelta(days=7))
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/reset-password")
def reset_password(req: AuthRequest, session: Session = Depends(get_session)):
    user = session.exec(select(User).where(User.email == req.email)).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    user.hashed_password = get_password_hash(req.password)
    session.add(user)
    session.commit()
    return {"status": "ok", "message": "Password reset successfully"}

@router.get("/google/login")
def google_login(token: str):
    # Verify token
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        if not payload.get("sub"):
            raise HTTPException(status_code=401)
    except Exception as e:
        print(f"Token decode error: {e}")
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")

    client_id = os.getenv("GOOGLE_CLIENT_ID")
    if not client_id or client_id == "PASTE_YOUR_CLIENT_ID_HERE":
        raise HTTPException(status_code=500, detail="Google Client ID not configured")

    redirect_uri = _google_redirect_uri()
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid https://www.googleapis.com/auth/userinfo.email https://www.googleapis.com/auth/userinfo.profile https://www.googleapis.com/auth/gmail.readonly",
        "access_type": "offline",
        "prompt": "consent",
        "state": token # pass the user's jwt as state to identify them on callback
    }
    
    auth_url = "https://accounts.google.com/o/oauth2/v2/auth?" + urlencode(params)
    from fastapi.responses import RedirectResponse
    return RedirectResponse(auth_url)

@router.get("/google/callback")
async def google_callback(code: str, state: str, session: Session = Depends(get_session)):
    try:
        payload = jwt.decode(state, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = int(payload.get("sub"))
    except Exception:
        from fastapi.responses import RedirectResponse
        return RedirectResponse("http://localhost:5173/?error=invalid_state")
        
    user = session.get(User, user_id)
    if not user:
        from fastapi.responses import RedirectResponse
        return RedirectResponse("http://localhost:5173/?error=user_not_found")
        
    client_id = os.getenv("GOOGLE_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
    redirect_uri = _google_redirect_uri()
    
    token_url = "https://oauth2.googleapis.com/token"
    data = {
        "code": code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code"
    }
    
    async with httpx.AsyncClient() as client:
        resp = await client.post(token_url, data=data)
        if resp.status_code != 200:
            print(f"OAuth token exchange failed: {resp.status_code} - {resp.text}")
            error_type = "oauth_failed"
            try:
                error_type = resp.json().get("error", "oauth_failed")
            except:
                pass
            from fastapi.responses import RedirectResponse
            return RedirectResponse(f"http://localhost:5173/?error={error_type}")
            
        tokens = resp.json()
        
    user.google_access_token = tokens.get("access_token")
    if tokens.get("refresh_token"):
        user.google_refresh_token = tokens.get("refresh_token")
    user.token_expiry = int(time.time()) + tokens.get("expires_in", 3600)
    
    # Optional: fetch user's Gmail address for UI status and worker context.
    try:
        async with httpx.AsyncClient() as client:
            profile_resp = await client.get(
                "https://gmail.googleapis.com/gmail/v1/users/me/profile",
                headers={"Authorization": f"Bearer {user.google_access_token}"}
            )
            if profile_resp.status_code == 200:
                user.connected_email = profile_resp.json().get("emailAddress")
    except Exception:
        pass
        
    session.add(user)
    session.commit()
    
    from fastapi.responses import RedirectResponse
    return RedirectResponse("http://localhost:5173/?oauth_success=1")
