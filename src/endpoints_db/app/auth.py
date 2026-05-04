from fastapi import Header


async def require_user(x_user_id: str = Header(..., alias="X-User-Id")) -> str:
    """Extract authenticated user from PRISM header (placeholder)."""
    return x_user_id
