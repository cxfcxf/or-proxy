from pydantic import BaseModel


class ModelObject(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "openrouter"
    context_length: int | None = None


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelObject]


class ErrorDetail(BaseModel):
    message: str
    type: str = "proxy_error"
    code: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
