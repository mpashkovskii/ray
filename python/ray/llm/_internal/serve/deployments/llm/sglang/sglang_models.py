import os
from typing import Any, Dict, List, Optional

from pydantic import ConfigDict, Field

from ray.llm._internal.common.base_pydantic import BaseModelExtended
from ray.llm._internal.common.utils.cloud_utils import CloudMirrorConfig
from ray.llm._internal.common.utils.import_utils import try_import
from ray.llm._internal.serve.configs.constants import (
    ALLOW_NEW_PLACEMENT_GROUPS_IN_DEPLOYMENT,
    ENV_VARS_TO_PROPAGATE,
)
from ray.llm._internal.serve.configs.server_models import (
    GPUType,
    LLMConfig,
)
from ray.llm._internal.serve.observability.logging import get_logger
from ray.util.placement_group import (
    get_current_placement_group,
)

sglang = try_import("sglang")
logger = get_logger(__name__)


class SGLangEngineConfig(BaseModelExtended):
    model_config = ConfigDict(
        use_enum_values=True,
        extra="forbid",
    )

    model_id: str = Field(
        description="The identifier for the model. This is the id that will be used to query the model.",
    )
    hf_model_id: Optional[str] = Field(
        None, description="The Hugging Face model identifier."
    )
    mirror_config: Optional[CloudMirrorConfig] = Field(
        None,
        description="Configuration for cloud storage mirror. This is for where the weights are downloaded from.",
    )
    resources_per_bundle: Optional[Dict[str, float]] = Field(
        default=None,
        description="This overrides the SGLang engine worker's default resource configuration, "
        "the number of resources returned by `placement_bundles`.",
    )
    accelerator_type: Optional[GPUType] = Field(
        None,
        description="The type of accelerator to use. This is used to determine the placement group strategy.",
    )
    actual_hf_model_id: Optional[str] = Field(
        default=None, description="The actual hugging face model identifier."
    )
    use_gpu: bool = True
    initialization_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional keyword arguments to pass to the SGLang engine initialization.",
    )
    runtime_env: Dict[str, Any] = Field(
        default_factory=dict,
        description="Runtime environment for SGLang workers.",
    )
    trust_remote_code: bool = Field(
        default=False,
        description="Whether to trust remote code in SGLang",
    )

    def get_initialization_kwargs(self) -> Dict[str, Any]:
        """Get the initialization kwargs for SGLang engine."""
        kwargs = self.initialization_kwargs.copy()

        # Add standard parameters
        if self.hf_model_id:
            kwargs["model"] = self.hf_model_id
        elif self.model_id:
            kwargs["model"] = self.model_id

        kwargs["trust_remote_code"] = self.trust_remote_code

        return kwargs

    def get_runtime_env(self) -> Dict[str, Any]:
        """Get the runtime environment for SGLang workers."""
        runtime_env = self.runtime_env.copy()

        # Add environment variables that should be propagated
        env_vars = runtime_env.get("env_vars", {})
        for env_var in ENV_VARS_TO_PROPAGATE:
            if env_var in os.environ:
                env_vars[env_var] = os.environ[env_var]

        if env_vars:
            runtime_env["env_vars"] = env_vars

        return runtime_env

    def get_placement_bundles(self) -> List[Dict[str, float]]:
        """Get placement group bundles for SGLang workers."""
        if self.resources_per_bundle:
            return [self.resources_per_bundle]

        # Default resource allocation
        bundle = {"CPU": 1}
        if self.use_gpu:
            bundle["GPU"] = 1

        return [bundle]

    def build_pg_config(self) -> Dict[str, Any]:
        """Build placement group configuration."""
        bundles = self.get_placement_bundles()

        # Check if we can create new placement groups
        if not ALLOW_NEW_PLACEMENT_GROUPS_IN_DEPLOYMENT:
            current_pg = get_current_placement_group()
            if current_pg:
                logger.info(f"Using existing placement group: {current_pg}")
                return {"placement_group": current_pg}

        # Create new placement group configuration
        pg_config = {
            "bundles": bundles,
            "strategy": "STRICT_PACK",
        }

        if self.accelerator_type:
            pg_config[
                "_custom_resource_prefix"
            ] = f"accelerator_type:{self.accelerator_type.value}"

        return pg_config

    def validate_config(self) -> None:
        """Validate the SGLang engine configuration."""
        if not self.model_id and not self.hf_model_id:
            raise ValueError("Either model_id or hf_model_id must be specified")

        # Check if SGLang is available
        if sglang is None:
            raise ImportError(
                "SGLang is not installed. Please install it with `pip install sglang`."
            )

    @classmethod
    def from_llm_config(cls, llm_config: LLMConfig) -> "SGLangEngineConfig":
        """Create SGLangEngineConfig from LLMConfig."""
        # This would be implemented to convert from the general LLMConfig
        # to SGLang-specific configuration
        return cls(
            model_id=llm_config.model_id,
            hf_model_id=llm_config.hf_model_id,
            accelerator_type=llm_config.accelerator_type,
            trust_remote_code=getattr(llm_config, "trust_remote_code", False),
            initialization_kwargs=getattr(llm_config, "initialization_kwargs", {}),
        )
