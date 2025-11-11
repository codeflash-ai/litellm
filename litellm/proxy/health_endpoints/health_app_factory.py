from fastapi import FastAPI
from litellm.proxy.health_endpoints._health_endpoints import router as health_router

# Move FastAPI instance and router inclusion to module scope for reuse.
_health_app: FastAPI = FastAPI(title="LiteLLM Health Endpoints")
_health_app.include_router(health_router)


def build_health_app():
    # Return the pre-built FastAPI instance to minimize repeated instantiation and router setup.
    return _health_app
