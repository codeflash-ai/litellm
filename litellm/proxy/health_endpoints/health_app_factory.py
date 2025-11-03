from fastapi import FastAPI
from litellm.proxy.health_endpoints._health_endpoints import router as health_router

_health_app = FastAPI(title="LiteLLM Health Endpoints")

_health_app.include_router(health_router)


def build_health_app():
    # Return the cached FastAPI app to avoid repeatedly re-creating and re-including the router
    return _health_app
