"""Keep Cookbook training restores and policy samplers on one hosted context.

Only Cookbook's module reference is adapted; the SDK and installed packages are
untouched. All checkpoint/optimizer operations delegate to the original client.
"""
from contextlib import contextmanager
import uuid


class TrainingClient:
    def __init__(self, client, service, base_model):
        self._client, self._service, self._base_model = client, service, base_model

    def __getattr__(self, name):
        return getattr(self._client, name)

    def create_sampling_client(self, model_path, **kwargs):
        return self._service.create_sampling_client(model_path=model_path,
                                                   base_model=self._base_model, **kwargs)

    async def create_sampling_client_async(self, model_path, **kwargs):
        return await self._service.create_sampling_client_async(model_path=model_path,
                                                               base_model=self._base_model, **kwargs)

    async def save_weights_and_get_sampling_client_async(self, name=None, **kwargs):
        # Public APIs only. A bounded-TTL snapshot replaces the SDK's ephemeral
        # shortcut, which offers no base_model override for sampling.
        future = await self._client.save_weights_for_sampler_async(
            name=name or 'artsy-policy-' + uuid.uuid4().hex, ttl_seconds=86400)
        saved = await future.result_async()
        return await self.create_sampling_client_async(saved.path, **kwargs)


class ServiceClient:
    def __init__(self, client, base_model):
        self._client, self._base_model = client, base_model

    def __getattr__(self, name):
        return getattr(self._client, name)

    async def create_training_client_from_state_async(self, path, **kwargs):
        client = await self._client.create_training_client_from_state_async(
            path, **{**kwargs, 'base_model': self._base_model})
        return TrainingClient(client, self._client, self._base_model)

    async def create_training_client_from_state_with_optimizer_async(self, path, **kwargs):
        client = await self._client.create_training_client_from_state_with_optimizer_async(
            path, **{**kwargs, 'base_model': self._base_model})
        return TrainingClient(client, self._client, self._base_model)

    async def create_lora_training_client_async(self, base_model, **kwargs):
        client = await self._client.create_lora_training_client_async(self._base_model, **kwargs)
        return TrainingClient(client, self._client, self._base_model)


@contextmanager
def cookbook_hosted_context(cookbook, base_model):
    original = cookbook.tinker
    class SDKProxy:
        def __getattr__(self, name):
            return getattr(original, name)

        def ServiceClient(self, *args, **kwargs):
            return ServiceClient(original.ServiceClient(*args, **kwargs), base_model)
    cookbook.tinker = SDKProxy()
    try:
        yield
    finally:
        cookbook.tinker = original
