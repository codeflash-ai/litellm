import ast
import base64
import binascii
import os
import traceback
from typing import Any, Optional, Union

import httpx

import litellm
from litellm._logging import print_verbose, verbose_logger
from litellm.caching.caching import DualCache
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.secret_managers.get_azure_ad_token_provider import \
    get_azure_ad_token_provider
from litellm.types.secret_managers.main import KeyManagementSystem

oidc_cache = DualCache()


######### Secret Manager ############################
# checks if user has passed in a secret manager client
# if passed in then checks the secret there
def _is_base64(s):
    try:
        return base64.b64encode(base64.b64decode(s)).decode() == s
    except binascii.Error:
        return False


def str_to_bool(value: Optional[str]) -> Optional[bool]:
    """
    Converts a string to a boolean if it's a recognized boolean string.
    Returns None if the string is not a recognized boolean value.

    :param value: The string to be checked.
    :return: True or False if the string is a recognized boolean, otherwise None.
    """
    if value is None:
        return None

    true_values = {"true"}
    false_values = {"false"}

    value_lower = value.strip().lower()

    if value_lower in true_values:
        return True
    elif value_lower in false_values:
        return False
    else:
        return None


def get_secret_str(
    secret_name: str,
    default_value: Optional[Union[str, bool]] = None,
) -> Optional[str]:
    """
    Guarantees response from 'get_secret' is either string or none. Used for fixing linting errors.
    """
    value = get_secret(secret_name=secret_name, default_value=default_value)
    if value is not None and not isinstance(value, str):
        return None

    return value


def get_secret_bool(
    secret_name: str,
    default_value: Optional[bool] = None,
) -> Optional[bool]:
    """
    Guarantees response from 'get_secret' is either boolean or none. Used for fixing linting errors.

    Args:
        secret_name: The name of the secret to get.
        default_value: The default value to return if the secret is not found.

    Returns:
        The secret value as a boolean or None if the secret is not found.
    """
    _secret_value = get_secret(secret_name, default_value)
    if _secret_value is None:
        return None
    elif isinstance(_secret_value, bool):
        return _secret_value
    else:
        return str_to_bool(_secret_value)


def get_secret(  # noqa: PLR0915
    secret_name: str,
    default_value: Optional[Union[str, bool]] = None,
):
    key_management_system = litellm._key_management_system
    key_management_settings = litellm._key_management_settings
    secret = None

    # Optimize: Quick early-out if secret_name is os.environ/ and environment contains it
    if secret_name.startswith("os.environ/"):
        env_var = secret_name[11:]
        secret = os.environ.get(env_var)
        secret_value_as_bool = str_to_bool(secret) if secret is not None else None
        if secret_value_as_bool is not None and isinstance(secret_value_as_bool, bool):
            return secret_value_as_bool
        return secret

    # OIDC logic
    if secret_name.startswith("oidc/"):
        secret_name_split = secret_name[5:]
        first_slash = secret_name_split.find("/")
        oidc_provider = secret_name_split[:first_slash]
        oidc_aud = secret_name_split[first_slash+1:]
        # TODO: Add caching for HTTP requests
        if oidc_provider == "google":
            oidc_token = oidc_cache.get_cache(key=secret_name)
            if oidc_token is not None:
                return oidc_token

            oidc_client = HTTPHandler(timeout=httpx.Timeout(timeout=600.0, connect=5.0))
            response = oidc_client.get(
                "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/identity",
                params={"audience": oidc_aud},
                headers={"Metadata-Flavor": "Google"},
            )
            if response.status_code == 200:
                oidc_token = response.text
                oidc_cache.set_cache(key=secret_name, value=oidc_token, ttl=3600 - 60)
                return oidc_token
            raise ValueError("Google OIDC provider failed")
        elif oidc_provider == "circleci":
            env_secret = os.getenv("CIRCLE_OIDC_TOKEN")
            if env_secret is None:
                raise ValueError("CIRCLE_OIDC_TOKEN not found in environment")
            return env_secret
        elif oidc_provider == "circleci_v2":
            env_secret = os.getenv("CIRCLE_OIDC_TOKEN_V2")
            if env_secret is None:
                raise ValueError("CIRCLE_OIDC_TOKEN_V2 not found in environment")
            return env_secret
        elif oidc_provider == "github":
            actions_id_token_request_url = os.getenv("ACTIONS_ID_TOKEN_REQUEST_URL")
            actions_id_token_request_token = os.getenv("ACTIONS_ID_TOKEN_REQUEST_TOKEN")
            if actions_id_token_request_url is None or actions_id_token_request_token is None:
                raise ValueError(
                    "ACTIONS_ID_TOKEN_REQUEST_URL or ACTIONS_ID_TOKEN_REQUEST_TOKEN not found in environment"
                )
            oidc_token = oidc_cache.get_cache(key=secret_name)
            if oidc_token is not None:
                return oidc_token

            oidc_client = HTTPHandler(timeout=httpx.Timeout(timeout=600.0, connect=5.0))
            response = oidc_client.get(
                actions_id_token_request_url,
                params={"audience": oidc_aud},
                headers={
                    "Authorization": f"Bearer {actions_id_token_request_token}",
                    "Accept": "application/json; api-version=2.0",
                },
            )
            if response.status_code == 200:
                oidc_token = response.json().get("value", None)
                oidc_cache.set_cache(key=secret_name, value=oidc_token, ttl=300 - 5)
                return oidc_token
            raise ValueError("Github OIDC provider failed")
        elif oidc_provider == "azure":
            azure_federated_token_file = os.getenv("AZURE_FEDERATED_TOKEN_FILE")
            if azure_federated_token_file is None:
                verbose_logger.warning(
                    "AZURE_FEDERATED_TOKEN_FILE not found in environment will use Azure AD token provider"
                )
                azure_token_provider = get_azure_ad_token_provider(azure_scope=oidc_aud)
                try:
                    oidc_token = azure_token_provider()
                    if oidc_token is None:
                        raise ValueError("Azure OIDC provider returned None token")
                    return oidc_token
                except Exception as e:
                    error_msg = f"Azure OIDC provider failed: {str(e)}"
                    verbose_logger.error(error_msg)
                    raise ValueError(error_msg)
            with open(azure_federated_token_file, "r") as f:
                return f.read()
        elif oidc_provider == "file":
            with open(oidc_aud, "r") as f:
                return f.read()
        elif oidc_provider == "env":
            oidc_token = os.getenv(oidc_aud)
            if oidc_token is None:
                raise ValueError(f"Environment variable {oidc_aud} not found")
            return oidc_token
        elif oidc_provider == "env_path":
            token_file_path = os.getenv(oidc_aud)
            if token_file_path is None:
                raise ValueError(f"Environment variable {oidc_aud} not found")
            with open(token_file_path, "r") as f:
                return f.read()
        raise ValueError("Unsupported OIDC provider")

    try:
        if _should_read_secret_from_secret_manager() and litellm.secret_manager_client is not None:
            try:
                client = litellm.secret_manager_client
                key_manager = "local"
                if key_management_system is not None:
                    key_manager = key_management_system.value

                if key_management_settings is not None:
                    hosted_keys = key_management_settings.hosted_keys
                    if (
                        hosted_keys is not None
                        and secret_name not in hosted_keys
                    ):
                        key_manager = "local"

                if (
                    key_manager == KeyManagementSystem.AZURE_KEY_VAULT.value
                    or type(client).__module__ + "." + type(client).__name__
                    == "azure.keyvault.secrets._client.SecretClient"
                ):
                    secret = client.get_secret(secret_name).value
                elif (
                    key_manager == KeyManagementSystem.GOOGLE_KMS.value
                    or client.__class__.__name__ == "KeyManagementServiceClient"
                ):
                    encrypted_secret: Any = os.getenv(secret_name)
                    if encrypted_secret is None:
                        raise ValueError("Google KMS requires the encrypted secret to be in the environment!")
                    b64_flag = _is_base64(encrypted_secret)
                    if b64_flag:
                        ciphertext = base64.b64decode(encrypted_secret)
                    else:
                        raise ValueError(
                            "Google KMS requires the encrypted secret to be encoded in base64"
                        )
                    response = client.decrypt(
                        request={
                            "name": litellm._google_kms_resource_name,
                            "ciphertext": ciphertext,
                        }
                    )
                    secret = response.plaintext.decode("utf-8")
                elif key_manager == KeyManagementSystem.AWS_KMS.value:
                    encrypted_value = os.getenv(secret_name, None)
                    if encrypted_value is None:
                        raise Exception(f"AWS KMS - Encrypted Value of Key={secret_name} is None")
                    ciphertext_blob = base64.b64decode(encrypted_value)
                    params = {"CiphertextBlob": ciphertext_blob}
                    response = client.decrypt(**params)
                    plaintext = response["Plaintext"]
                    secret = plaintext.decode("utf-8")
                    if isinstance(secret, str):
                        secret = secret.strip()
                elif key_manager == KeyManagementSystem.AWS_SECRET_MANAGER.value:
                    from litellm.secret_managers.aws_secret_manager_v2 import \
                        AWSSecretsManagerV2

                    if isinstance(client, AWSSecretsManagerV2):
                        secret = client.sync_read_secret(
                            secret_name=secret_name,
                            primary_secret_name=key_management_settings.primary_secret_name,
                        )
                        print_verbose(f"get_secret_value_response: {secret}")
                elif key_manager == KeyManagementSystem.GOOGLE_SECRET_MANAGER.value:
                    try:
                        secret = client.get_secret_from_google_secret_manager(secret_name)
                        print_verbose(f"secret from google secret manager:  {secret}")
                        if secret is None:
                            raise ValueError(f"No secret found in Google Secret Manager for {secret_name}")
                    except Exception as e:
                        print_verbose(f"An error occurred - {str(e)}")
                        raise e
                elif key_manager == KeyManagementSystem.HASHICORP_VAULT.value:
                    try:
                        secret = client.sync_read_secret(secret_name=secret_name)
                        if secret is None:
                            raise ValueError(f"No secret found in Hashicorp Secret Manager for {secret_name}")
                    except Exception as e:
                        print_verbose(f"An error occurred - {str(e)}")
                        raise e
                elif key_manager == "local":
                    secret = os.getenv(secret_name)
                else:  # assume the default is infisicial client
                    secret = client.get_secret(secret_name).secret_value
            except Exception as e:
                verbose_logger.error(
                    f"Defaulting to os.environ value for key={secret_name}. An exception occurred - {str(e)}.\n\n{traceback.format_exc()}"
                )
                secret = os.getenv(secret_name)
            try:
                if isinstance(secret, str):
                    secret_value_as_bool = ast.literal_eval(secret)
                    if isinstance(secret_value_as_bool, bool):
                        return secret_value_as_bool
                    return secret
            except Exception:
                return secret
        else:
            secret = os.environ.get(secret_name)
            secret_value_as_bool = str_to_bool(secret) if secret is not None else None
            if secret_value_as_bool is not None and isinstance(secret_value_as_bool, bool):
                return secret_value_as_bool
            return secret
    except Exception as e:
        if default_value is not None:
            return default_value
        raise e


def _should_read_secret_from_secret_manager() -> bool:
    """
    Returns True if the secret manager should be used to read the secret, False otherwise

    - If the secret manager client is not set, return False
    - If the `_key_management_settings` access mode is "read_only" or "read_and_write", return True
    - Otherwise, return False
    """
    if litellm.secret_manager_client is not None:
        if litellm._key_management_settings is not None:
            if (
                litellm._key_management_settings.access_mode == "read_only"
                or litellm._key_management_settings.access_mode == "read_and_write"
            ):
                return True
    return False
