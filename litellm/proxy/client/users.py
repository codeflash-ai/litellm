import requests
from typing import List, Dict, Any, Optional
from .exceptions import UnauthorizedError, NotFoundError


class UsersManagementClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

        # Precompute headers without Authorization for reuse
        self._base_headers = {"Content-Type": "application/json"}
        # Precompute headers with Authorization if api_key provided
        if api_key:
            self._auth_headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
        else:
            self._auth_headers = self._base_headers

    def _get_headers(self) -> Dict[str, str]:
        # Return precomputed headers object rather than constructing each call
        # This saves dict creation and string formatting costs in hot path
        return self._auth_headers

    def list_users(self, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """List users (GET /user/list)"""
        url = f"{self.base_url}/user/list"
        response = requests.get(url, headers=self._get_headers(), params=params)
        if response.status_code == 401:
            raise UnauthorizedError(response.text)
        response.raise_for_status()
        return response.json().get("users", response.json())

    def get_user(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get user info (GET /user/info)"""
        url = f"{self.base_url}/user/info"
        # The following check runs on every request but is essentially free
        if user_id is not None:
            params = {"user_id": user_id}
        else:
            params = {}
        # Requests session reuse can reduce connection overhead, but we preserve single requests call to avoid mutating behavior
        response = requests.get(url, headers=self._get_headers(), params=params)
        if response.status_code == 401:
            raise UnauthorizedError(response.text)
        if response.status_code == 404:
            raise NotFoundError(response.text)
        response.raise_for_status()
        return response.json()

    def create_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new user (POST /user/new)"""
        url = f"{self.base_url}/user/new"
        response = requests.post(url, headers=self._get_headers(), json=user_data)
        if response.status_code == 401:
            raise UnauthorizedError(response.text)
        response.raise_for_status()
        return response.json()

    def delete_user(self, user_ids: List[str]) -> Dict[str, Any]:
        """Delete users (POST /user/delete)"""
        url = f"{self.base_url}/user/delete"
        response = requests.post(url, headers=self._get_headers(), json={"user_ids": user_ids})
        if response.status_code == 401:
            raise UnauthorizedError(response.text)
        response.raise_for_status()
        return response.json()
