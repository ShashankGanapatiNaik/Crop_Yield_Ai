from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ..core.config import get_settings
from ..core.limiter import limiter
from ..core.security import (
    create_access_token,
    get_current_user,
    hash_password,
    verify_password,
)
from ..db import get_db
from ..models.orm import User
from ..schemas import LoginIn, RegisterIn, TokenOut, UserOut

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=TokenOut, status_code=201)
@limiter.limit(get_settings().rate_limit_auth)
def register(request: Request, body: RegisterIn, db: Session = Depends(get_db)):
    if db.scalar(select(User).where(User.email == body.email.lower())):
        raise HTTPException(status.HTTP_409_CONFLICT, "An account with this email already exists")
    user = User(name=body.name.strip(), email=body.email.lower(), hashed_password=hash_password(body.password), role="farmer")
    db.add(user); db.commit(); db.refresh(user)
    return TokenOut(access_token=create_access_token(user), user=UserOut.model_validate(user))


@router.post("/login", response_model=TokenOut)
@limiter.limit(get_settings().rate_limit_auth)
def login(request: Request, body: LoginIn, db: Session = Depends(get_db)):
    user = db.scalar(select(User).where(User.email == body.email.lower()))
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect email or password")
    return TokenOut(access_token=create_access_token(user), user=UserOut.model_validate(user))


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)):
    return user
